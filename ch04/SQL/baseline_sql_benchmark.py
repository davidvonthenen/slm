# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

#!/usr/bin/env python3
"""Benchmark the baseline Qwen2.5-7B-Instruct model on deterministic SQL tasks.

The benchmark uses the same chat structure as 1_finetune.py:

  system: You are a database engineer. Generate valid SQL for the given schema.
  user:   Schema: <schema>\nQuestion: <question>

Each generated query is executed against a seeded in-memory SQLite database and
compared with a reference query. The script writes:

- baseline_scorecard.json
- baseline_scorecard.csv
- baseline_failure_matrix.csv
- baseline_report.md

Usage:
  python baseline_sql_benchmark.py
  python baseline_sql_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema."
BENCHMARK_VERSION = "sql-specialist-v1"
MAX_NEW_TOKENS = 256
SEED = 3407
BASELINE_OUTPUT_DIR = Path("./benchmark_results/baseline")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    category: str
    difficulty: str
    critical: bool
    schema_sql: str
    seed_sql: str
    question: str
    reference_sql: str
    order_matters: bool = True

    @property
    def request(self) -> str:
        return f"Schema: {self.schema_sql.strip()}\nQuestion: {self.question.strip()}"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    time_to_first_token_seconds: float | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.elapsed_seconds <= 0:
            return None
        return self.completion_tokens / self.elapsed_seconds


class SqlGenerator(Protocol):
    model_identifier: str
    load_time_seconds: float

    def generate(self, schema: str, question: str) -> GenerationResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


def benchmark_cases() -> list[BenchmarkCase]:
    """Return the fixed, held-out benchmark suite."""

    return [
        BenchmarkCase(
            case_id="B01_FILTER_ORDER",
            title="Filter and deterministic ordering",
            category="filtering",
            difficulty="basic",
            critical=False,
            schema_sql="""
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    city TEXT NOT NULL,
                    active INTEGER NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO customers VALUES
                    (1, 'Alice', 'Seattle', 1),
                    (2, 'Bob', 'Portland', 1),
                    (3, 'Carla', 'Seattle', 0),
                    (4, 'Dinesh', 'Seattle', 1),
                    (5, 'Eve', 'Seattle', 0);
            """,
            question=(
                "Using SQLite syntax, list customer_name for active customers in Seattle. "
                "Sort alphabetically. Return only customer_name using one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT customer_name
                FROM customers
                WHERE active = 1 AND city = 'Seattle'
                ORDER BY customer_name;
            """,
        ),
        BenchmarkCase(
            case_id="B02_JOIN_SORT",
            title="Join with filtering and tie-safe ordering",
            category="joins",
            difficulty="basic",
            critical=False,
            schema_sql="""
                CREATE TABLE departments (
                    department_id INTEGER PRIMARY KEY,
                    department_name TEXT NOT NULL
                );
                CREATE TABLE employees (
                    employee_id INTEGER PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    department_id INTEGER NOT NULL,
                    salary REAL NOT NULL,
                    FOREIGN KEY (department_id) REFERENCES departments(department_id)
                );
            """,
            seed_sql="""
                INSERT INTO departments VALUES
                    (10, 'Engineering'),
                    (20, 'Sales');
                INSERT INTO employees VALUES
                    (1, 'Mina', 'Chen', 10, 142000),
                    (2, 'Omar', 'Diaz', 10, 126000),
                    (3, 'Priya', 'Shah', 20, 131000),
                    (4, 'Ravi', 'Singh', 10, 126000);
            """,
            question=(
                "Using SQLite syntax, return first_name, last_name, and salary for employees in the "
                "Engineering department. Sort by salary descending, then last_name ascending. "
                "Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT e.first_name, e.last_name, e.salary
                FROM employees AS e
                JOIN departments AS d
                  ON d.department_id = e.department_id
                WHERE d.department_name = 'Engineering'
                ORDER BY e.salary DESC, e.last_name ASC;
            """,
        ),
        BenchmarkCase(
            case_id="B03_AGG_HAVING",
            title="Join, aggregation, and HAVING",
            category="aggregation",
            difficulty="intermediate",
            critical=False,
            schema_sql="""
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    customer_name TEXT NOT NULL
                );
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    total REAL NOT NULL,
                    status TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO customers VALUES
                    (1, 'Ada'), (2, 'Ben'), (3, 'Cara'), (4, 'Diego');
                INSERT INTO orders VALUES
                    (101, 1, 120.00, 'completed'),
                    (102, 1, 130.00, 'completed'),
                    (103, 1, 999.00, 'pending'),
                    (104, 2, 210.00, 'completed'),
                    (105, 3, 50.00, 'completed'),
                    (106, 4, 500.00, 'cancelled');
            """,
            question=(
                "Using SQLite syntax, find customers whose completed-order revenue is greater than 200. "
                "Return customer_name and total_revenue rounded to two decimal places. Sort by "
                "total_revenue descending, then customer_name. Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT c.customer_name, ROUND(SUM(o.total), 2) AS total_revenue
                FROM customers AS c
                JOIN orders AS o
                  ON o.customer_id = c.customer_id
                WHERE o.status = 'completed'
                GROUP BY c.customer_id, c.customer_name
                HAVING SUM(o.total) > 200
                ORDER BY total_revenue DESC, c.customer_name ASC;
            """,
        ),
        BenchmarkCase(
            case_id="B04_LEFT_JOIN_ZERO",
            title="LEFT JOIN with zero-valued groups",
            category="joins",
            difficulty="intermediate",
            critical=False,
            schema_sql="""
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL
                );
                CREATE TABLE order_items (
                    order_item_id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO products VALUES
                    (1, 'Adapter'), (2, 'Cable'), (3, 'Widget');
                INSERT INTO order_items VALUES
                    (1, 2, 4),
                    (2, 2, 3),
                    (3, 3, 5),
                    (4, 3, 2);
            """,
            question=(
                "Using SQLite syntax, return every product_name and total units sold, including products "
                "with no sales as zero. Sort by units_sold descending, then product_name ascending. "
                "Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT p.product_name, COALESCE(SUM(oi.quantity), 0) AS units_sold
                FROM products AS p
                LEFT JOIN order_items AS oi
                  ON oi.product_id = p.product_id
                GROUP BY p.product_id, p.product_name
                ORDER BY units_sold DESC, p.product_name ASC;
            """,
        ),
        BenchmarkCase(
            case_id="B05_NULL_SAFE_ANTI_JOIN",
            title="NULL-safe anti-join",
            category="null_semantics",
            difficulty="advanced",
            critical=True,
            schema_sql="""
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    customer_name TEXT NOT NULL
                );
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    status TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO customers VALUES
                    (1, 'Alpha'), (2, 'Beta'), (3, 'Gamma'), (4, 'Delta');
                INSERT INTO orders VALUES
                    (1, 1, 'completed'),
                    (2, 2, 'pending'),
                    (3, 4, 'completed'),
                    (4, NULL, 'completed');
            """,
            question=(
                "Using SQLite syntax, list customers who have never placed a completed order. "
                "The orders.customer_id column can contain NULL. Return customer_name alphabetically. "
                "Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT c.customer_name
                FROM customers AS c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM orders AS o
                    WHERE o.customer_id = c.customer_id
                      AND o.status = 'completed'
                )
                ORDER BY c.customer_name;
            """,
        ),
        BenchmarkCase(
            case_id="B06_SQLITE_DATE_GROUP",
            title="Dialect-specific date grouping",
            category="date_logic",
            difficulty="intermediate",
            critical=False,
            schema_sql="""
                CREATE TABLE events (
                    event_id INTEGER PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO events VALUES
                    (1, '2025-01-03 10:00:00', 'login'),
                    (2, '2025-01-28 12:00:00', 'purchase'),
                    (3, '2025-02-14 09:00:00', 'login'),
                    (4, '2025-03-01 08:30:00', 'login'),
                    (5, '2025-03-31 23:59:59', 'purchase'),
                    (6, '2025-04-01 00:00:00', 'login');
            """,
            question=(
                "Using SQLite syntax, count events by calendar month for the first quarter of 2025. "
                "Return month in YYYY-MM form and event_count. Include only months containing events and "
                "sort by month. Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT strftime('%Y-%m', occurred_at) AS month, COUNT(*) AS event_count
                FROM events
                WHERE occurred_at >= '2025-01-01'
                  AND occurred_at < '2025-04-01'
                GROUP BY month
                ORDER BY month;
            """,
        ),
        BenchmarkCase(
            case_id="B07_WINDOW_TOP_TWO",
            title="Top two rows per group",
            category="window_functions",
            difficulty="advanced",
            critical=True,
            schema_sql="""
                CREATE TABLE departments (
                    department_id INTEGER PRIMARY KEY,
                    department_name TEXT NOT NULL
                );
                CREATE TABLE employees (
                    employee_id INTEGER PRIMARY KEY,
                    employee_name TEXT NOT NULL,
                    department_id INTEGER NOT NULL,
                    salary REAL NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO departments VALUES
                    (1, 'Engineering'), (2, 'Sales'), (3, 'Support');
                INSERT INTO employees VALUES
                    (1, 'Asha', 1, 150000),
                    (2, 'Bruno', 1, 140000),
                    (3, 'Chen', 1, 140000),
                    (4, 'Dara', 2, 130000),
                    (5, 'Elena', 2, 120000),
                    (6, 'Farah', 2, 110000),
                    (7, 'Gabe', 3, 100000),
                    (8, 'Hana', 3, 95000),
                    (9, 'Ivan', 3, 90000);
            """,
            question=(
                "Using SQLite syntax, return the two highest-paid employees in each department. Break salary "
                "ties by employee_name ascending before selecting the top two. Return department_name, "
                "employee_name, and salary. Sort final rows by department_name, salary descending, then "
                "employee_name. Use one read-only query and no explanation."
            ),
            reference_sql="""
                WITH ranked AS (
                    SELECT
                        d.department_name,
                        e.employee_name,
                        e.salary,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.department_id
                            ORDER BY e.salary DESC, e.employee_name ASC
                        ) AS row_num
                    FROM employees AS e
                    JOIN departments AS d
                      ON d.department_id = e.department_id
                )
                SELECT department_name, employee_name, salary
                FROM ranked
                WHERE row_num <= 2
                ORDER BY department_name, salary DESC, employee_name;
            """,
        ),
        BenchmarkCase(
            case_id="B08_CONDITIONAL_PERCENTAGE",
            title="Conditional percentage with decimal arithmetic",
            category="aggregation",
            difficulty="intermediate",
            critical=False,
            schema_sql="""
                CREATE TABLE teams (
                    team_id INTEGER PRIMARY KEY,
                    team_name TEXT NOT NULL
                );
                CREATE TABLE tickets (
                    ticket_id INTEGER PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    status TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO teams VALUES
                    (1, 'Blue'), (2, 'Green'), (3, 'Red');
                INSERT INTO tickets VALUES
                    (1, 1, 'resolved'),
                    (2, 1, 'resolved'),
                    (3, 1, 'resolved'),
                    (4, 1, 'open'),
                    (5, 2, 'resolved'),
                    (6, 2, 'open'),
                    (7, 2, 'open'),
                    (8, 3, 'open');
            """,
            question=(
                "Using SQLite syntax, calculate the percentage of tickets resolved for each team. Return "
                "team_name and resolved_pct rounded to one decimal place. Sort by team_name. Avoid integer "
                "division. Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT
                    t.team_name,
                    ROUND(100.0 * SUM(CASE WHEN k.status = 'resolved' THEN 1 ELSE 0 END) / COUNT(*), 1)
                        AS resolved_pct
                FROM teams AS t
                JOIN tickets AS k
                  ON k.team_id = t.team_id
                GROUP BY t.team_id, t.team_name
                ORDER BY t.team_name;
            """,
        ),
        BenchmarkCase(
            case_id="B09_DISTINCT_NULL",
            title="COUNT DISTINCT with NULL and duplicate values",
            category="null_semantics",
            difficulty="intermediate",
            critical=False,
            schema_sql="""
                CREATE TABLE users (
                    user_id INTEGER PRIMARY KEY,
                    user_name TEXT NOT NULL
                );
                CREATE TABLE sessions (
                    session_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    device TEXT
                );
            """,
            seed_sql="""
                INSERT INTO users VALUES
                    (1, 'Ada'), (2, 'Ben'), (3, 'Cara'), (4, 'Dan');
                INSERT INTO sessions VALUES
                    (1, 1, 'mobile'),
                    (2, 1, 'desktop'),
                    (3, 1, NULL),
                    (4, 2, 'mobile'),
                    (5, 2, 'mobile'),
                    (6, 3, 'tablet'),
                    (7, 3, 'desktop'),
                    (8, 3, 'mobile');
            """,
            question=(
                "Using SQLite syntax, find users who used at least two distinct non-NULL device values. "
                "Return user_name and device_count. Sort by device_count descending, then user_name. "
                "Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT u.user_name, COUNT(DISTINCT s.device) AS device_count
                FROM users AS u
                JOIN sessions AS s
                  ON s.user_id = u.user_id
                WHERE s.device IS NOT NULL
                GROUP BY u.user_id, u.user_name
                HAVING COUNT(DISTINCT s.device) >= 2
                ORDER BY device_count DESC, u.user_name;
            """,
        ),
        BenchmarkCase(
            case_id="B10_CONDITIONAL_SUM",
            title="Conditional sums while retaining empty groups",
            category="aggregation",
            difficulty="advanced",
            critical=False,
            schema_sql="""
                CREATE TABLE accounts (
                    account_id INTEGER PRIMARY KEY,
                    account_name TEXT NOT NULL
                );
                CREATE TABLE payments (
                    payment_id INTEGER PRIMARY KEY,
                    account_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO accounts VALUES
                    (1, 'Acme'), (2, 'Beacon'), (3, 'Cobalt');
                INSERT INTO payments VALUES
                    (1, 1, 100.00, 'approved'),
                    (2, 1, 50.00, 'approved'),
                    (3, 1, 20.00, 'declined'),
                    (4, 2, 40.00, 'declined'),
                    (5, 2, 500.00, 'pending');
            """,
            question=(
                "Using SQLite syntax, return every account_name with approved_total and declined_total. "
                "Ignore pending payments and show zero when an account has no amount for a status. "
                "Sort by account_name. Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT
                    a.account_name,
                    COALESCE(SUM(CASE WHEN p.status = 'approved' THEN p.amount ELSE 0 END), 0) AS approved_total,
                    COALESCE(SUM(CASE WHEN p.status = 'declined' THEN p.amount ELSE 0 END), 0) AS declined_total
                FROM accounts AS a
                LEFT JOIN payments AS p
                  ON p.account_id = a.account_id
                GROUP BY a.account_id, a.account_name
                ORDER BY a.account_name;
            """,
        ),
        BenchmarkCase(
            case_id="B11_LATEST_ROW_TIE",
            title="Latest row per entity with deterministic tie-breaking",
            category="window_functions",
            difficulty="advanced",
            critical=True,
            schema_sql="""
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL
                );
                CREATE TABLE price_history (
                    history_id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    price REAL NOT NULL,
                    effective_at TEXT NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO products VALUES
                    (1, 'Alpha'), (2, 'Beta'), (3, 'Gamma');
                INSERT INTO price_history VALUES
                    (1, 1, 10.00, '2025-01-01'),
                    (2, 1, 12.50, '2025-02-01'),
                    (3, 1, 13.00, '2025-02-01'),
                    (4, 2, 8.00, '2025-01-15'),
                    (5, 2, 9.00, '2025-03-01'),
                    (6, 3, 20.00, '2024-12-01');
            """,
            question=(
                "Using SQLite syntax, return each product_name and its latest price. Latest means the greatest "
                "effective_at; when timestamps tie, choose the row with the greatest history_id. Sort by "
                "product_name. Use one read-only query and no explanation."
            ),
            reference_sql="""
                WITH ranked AS (
                    SELECT
                        p.product_name,
                        h.price,
                        ROW_NUMBER() OVER (
                            PARTITION BY h.product_id
                            ORDER BY h.effective_at DESC, h.history_id DESC
                        ) AS row_num
                    FROM price_history AS h
                    JOIN products AS p
                      ON p.product_id = h.product_id
                )
                SELECT product_name, price
                FROM ranked
                WHERE row_num = 1
                ORDER BY product_name;
            """,
        ),
        BenchmarkCase(
            case_id="B12_RELATIONAL_DIVISION",
            title="Relational division with duplicates and active filtering",
            category="set_logic",
            difficulty="advanced",
            critical=True,
            schema_sql="""
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    customer_name TEXT NOT NULL
                );
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    active INTEGER NOT NULL
                );
                CREATE TABLE purchases (
                    purchase_id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL
                );
            """,
            seed_sql="""
                INSERT INTO customers VALUES
                    (1, 'Ada'), (2, 'Ben'), (3, 'Cara'), (4, 'Dan');
                INSERT INTO products VALUES
                    (1, 'Firewall', 'Security', 1),
                    (2, 'Scanner', 'Security', 1),
                    (3, 'Legacy AV', 'Security', 0),
                    (4, 'Monitor', 'Operations', 1);
                INSERT INTO purchases VALUES
                    (1, 1, 1),
                    (2, 1, 2),
                    (3, 1, 1),
                    (4, 2, 1),
                    (5, 3, 2),
                    (6, 4, 1),
                    (7, 4, 2),
                    (8, 4, 4);
            """,
            question=(
                "Using SQLite syntax, list customers who purchased every active product in the Security "
                "category. Duplicate purchases must not change the answer, inactive Security products do not "
                "count, and products in other categories do not count. Return customer_name alphabetically. "
                "Use one read-only query and no explanation."
            ),
            reference_sql="""
                SELECT c.customer_name
                FROM customers AS c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM products AS p
                    WHERE p.category = 'Security'
                      AND p.active = 1
                      AND NOT EXISTS (
                          SELECT 1
                          FROM purchases AS x
                          WHERE x.customer_id = c.customer_id
                            AND x.product_id = p.product_id
                      )
                )
                ORDER BY c.customer_name;
            """,
        ),
    ]


def _compact_sql(value: str) -> str:
    return " ".join(value.split())


def benchmark_suite_hash(cases: list[BenchmarkCase]) -> str:
    payload = [asdict(case) for case in cases]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_statement(text: str) -> tuple[str, str]:
    """Split at the first semicolon outside quoted strings and comments."""

    quote: str | None = None
    line_comment = False
    block_comment = False
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue

        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
                continue
            index += 1
            continue

        if quote:
            if char == quote:
                if next_char == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "[":
            quote = "]"
            index += 1
            continue
        if char == "-" and next_char == "-":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char == ";":
            return text[: index + 1].strip(), text[index + 1 :].strip()

        index += 1

    return text.strip(), ""


def extract_sql(response: str) -> dict[str, Any]:
    raw = response.strip()
    if not raw:
        return {
            "sql": "",
            "format_violation": False,
            "multiple_statements": False,
            "extraction_source": "empty",
        }

    cleaned = re.sub(r"<\|(?:im_end|endoftext)\|>", "", raw, flags=re.IGNORECASE).strip()
    fenced_blocks = re.findall(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)

    source = cleaned
    extraction_source = "plain_text"
    outside_fence = ""
    if fenced_blocks:
        source = next(
            (block for block in fenced_blocks if re.search(r"\b(?:SELECT|WITH)\b", block, re.IGNORECASE)),
            fenced_blocks[0],
        ).strip()
        extraction_source = "markdown_fence"
        outside_fence = re.sub(r"```(?:sql)?\s*.*?```", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    start_match = re.search(r"\b(?:WITH|SELECT)\b", source, flags=re.IGNORECASE)
    if not start_match:
        return {
            "sql": "",
            "format_violation": bool(cleaned),
            "multiple_statements": False,
            "extraction_source": extraction_source,
        }

    prefix = source[: start_match.start()].strip()
    statement, remainder = _first_statement(source[start_match.start() :])
    statement_pattern = re.compile(
        r"\b(?:SELECT|WITH|INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|ATTACH|DETACH|PRAGMA|VACUUM)\b",
        flags=re.IGNORECASE,
    )
    multiple_statements = bool(
        statement_pattern.search(remainder)
        or statement_pattern.search(prefix)
        or statement_pattern.search(outside_fence)
    )
    format_violation = bool(prefix or remainder or outside_fence or extraction_source == "markdown_fence")

    return {
        "sql": statement.strip(),
        "format_violation": format_violation,
        "multiple_statements": multiple_statements,
        "extraction_source": extraction_source,
    }


def _strip_leading_comments(sql: str) -> str:
    value = sql.lstrip()
    while True:
        previous = value
        value = re.sub(r"^--[^\n]*(?:\n|$)", "", value).lstrip()
        value = re.sub(r"^/\*.*?\*/", "", value, count=1, flags=re.DOTALL).lstrip()
        if value == previous:
            return value


def is_read_only_query(sql: str) -> bool:
    value = _strip_leading_comments(sql)
    if not re.match(r"^(?:SELECT|WITH)\b", value, flags=re.IGNORECASE):
        return False

    forbidden = re.compile(
        r"\b(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
        flags=re.IGNORECASE,
    )
    return forbidden.search(value) is None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return round(value, 6)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_rows(rows: list[tuple[Any, ...]], order_matters: bool) -> list[list[Any]]:
    normalized = [[_normalize_value(value) for value in row] for row in rows]
    if not order_matters:
        normalized.sort(key=lambda row: json.dumps(row, sort_keys=True, default=str))
    return normalized


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
    return left == right


def rows_equal(expected: list[list[Any]], actual: list[list[Any]]) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_row, actual_row in zip(expected, actual):
        if len(expected_row) != len(actual_row):
            return False
        if any(not _values_equal(left, right) for left, right in zip(expected_row, actual_row)):
            return False
    return True


def _classify_sqlite_error(message: str) -> str:
    value = message.lower()
    if any(token in value for token in ("no such table", "no such column", "ambiguous column", "has no column")):
        return "schema_reference_error"
    if any(token in value for token in ("syntax error", "incomplete input", "unrecognized token", "near \"")):
        return "syntax_error"
    if "readonly" in value or "not authorized" in value:
        return "non_read_only_sql"
    if "interrupted" in value:
        return "query_timeout"
    return "execution_error"


def execute_case(case: BenchmarkCase, candidate_sql: str) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    deadline = time.perf_counter() + 2.0

    def stop_long_query() -> int:
        return 1 if time.perf_counter() > deadline else 0

    try:
        connection.executescript(case.schema_sql)
        connection.executescript(case.seed_sql)

        expected_rows = connection.execute(case.reference_sql).fetchall()
        expected = _normalize_rows(expected_rows, case.order_matters)

        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(stop_long_query, 1000)
        actual_rows = connection.execute(candidate_sql).fetchall()
        actual = _normalize_rows(actual_rows, case.order_matters)

        return {
            "executed": True,
            "result_correct": rows_equal(expected, actual),
            "expected_rows": expected,
            "actual_rows": actual,
            "error": "",
            "failure_category": "",
        }
    except sqlite3.Error as exc:
        expected: list[list[Any]] = []
        try:
            connection.set_progress_handler(None, 0)
            connection.execute("PRAGMA query_only = OFF")
            expected_rows = connection.execute(case.reference_sql).fetchall()
            expected = _normalize_rows(expected_rows, case.order_matters)
        except sqlite3.Error:
            pass

        message = str(exc)
        return {
            "executed": False,
            "result_correct": False,
            "expected_rows": expected,
            "actual_rows": [],
            "error": message,
            "failure_category": _classify_sqlite_error(message),
        }
    finally:
        connection.close()


def evaluate_response(case: BenchmarkCase, response: str) -> dict[str, Any]:
    extraction = extract_sql(response)
    candidate_sql = extraction["sql"]
    categories: list[str] = []
    quality_score = 0

    if not response.strip():
        return {
            "extracted_sql": "",
            "primary_failure_category": "empty_response",
            "failure_categories": ["empty_response"],
            "format_compliant": False,
            "executed": False,
            "result_correct": False,
            "strict_pass": False,
            "quality_score": 0,
            "expected_rows": [],
            "actual_rows": [],
            "error": "Model returned an empty response.",
            "extraction_source": extraction["extraction_source"],
        }

    quality_score += 10

    if not candidate_sql:
        return {
            "extracted_sql": "",
            "primary_failure_category": "no_sql_detected",
            "failure_categories": ["no_sql_detected"],
            "format_compliant": False,
            "executed": False,
            "result_correct": False,
            "strict_pass": False,
            "quality_score": quality_score,
            "expected_rows": [],
            "actual_rows": [],
            "error": "No SELECT or WITH query was detected in the response.",
            "extraction_source": extraction["extraction_source"],
        }

    quality_score += 15

    if extraction["multiple_statements"]:
        return {
            "extracted_sql": candidate_sql,
            "primary_failure_category": "multiple_statements",
            "failure_categories": ["multiple_statements"],
            "format_compliant": False,
            "executed": False,
            "result_correct": False,
            "strict_pass": False,
            "quality_score": quality_score,
            "expected_rows": [],
            "actual_rows": [],
            "error": "More than one SQL statement was detected.",
            "extraction_source": extraction["extraction_source"],
        }

    if not is_read_only_query(candidate_sql):
        return {
            "extracted_sql": candidate_sql,
            "primary_failure_category": "non_read_only_sql",
            "failure_categories": ["non_read_only_sql"],
            "format_compliant": False,
            "executed": False,
            "result_correct": False,
            "strict_pass": False,
            "quality_score": quality_score,
            "expected_rows": [],
            "actual_rows": [],
            "error": "The response was not a single read-only SELECT/CTE query.",
            "extraction_source": extraction["extraction_source"],
        }

    quality_score += 15
    execution = execute_case(case, candidate_sql)

    if execution["executed"]:
        quality_score += 20
        if execution["result_correct"]:
            quality_score += 40
        else:
            categories.append("incorrect_result")
    else:
        categories.append(execution["failure_category"])

    if extraction["format_violation"]:
        categories.append("format_violation")
        quality_score = max(0, quality_score - 5)

    if not categories:
        categories = ["pass"]

    result_correct = bool(execution["result_correct"])
    format_compliant = not extraction["format_violation"]
    strict_pass = result_correct and format_compliant

    return {
        "extracted_sql": candidate_sql,
        "primary_failure_category": categories[0],
        "failure_categories": categories,
        "format_compliant": format_compliant,
        "executed": bool(execution["executed"]),
        "result_correct": result_correct,
        "strict_pass": strict_pass,
        "quality_score": quality_score,
        "expected_rows": execution["expected_rows"],
        "actual_rows": execution["actual_rows"],
        "error": execution["error"],
        "extraction_source": extraction["extraction_source"],
    }


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    result_correct = sum(1 for row in case_results if row["result_correct"])
    strict_passes = sum(1 for row in case_results if row["strict_pass"])
    executed = sum(1 for row in case_results if row["executed"])
    format_compliant = sum(1 for row in case_results if row["format_compliant"])

    critical = [row for row in case_results if row["critical"]]
    critical_correct = sum(1 for row in critical if row["result_correct"])

    latencies = [float(row["latency_seconds"]) for row in case_results if row["latency_seconds"] is not None]
    throughput = [float(row["tokens_per_second"]) for row in case_results if row["tokens_per_second"] is not None]
    scores = [float(row["quality_score"]) for row in case_results]

    failure_counts: dict[str, int] = {}
    for row in case_results:
        category = row["primary_failure_category"]
        if category != "pass":
            failure_counts[category] = failure_counts.get(category, 0) + 1

    return {
        "total_cases": total,
        "executed_cases": executed,
        "result_correct_cases": result_correct,
        "strict_pass_cases": strict_passes,
        "format_compliant_cases": format_compliant,
        "execution_accuracy": result_correct / total if total else 0.0,
        "strict_accuracy": strict_passes / total if total else 0.0,
        "format_compliance_rate": format_compliant / total if total else 0.0,
        "critical_cases": len(critical),
        "critical_correct_cases": critical_correct,
        "critical_accuracy": critical_correct / len(critical) if critical else 0.0,
        "average_quality_score": statistics.fmean(scores) if scores else 0.0,
        "average_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_tokens_per_second": statistics.fmean(throughput) if throughput else None,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def path_size_bytes(value: str | Path) -> int | None:
    path = Path(value).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
    }


def run_benchmark(
    runner: SqlGenerator,
    *,
    run_kind: str,
    model_label: str,
    cases: list[BenchmarkCase] | None = None,
) -> dict[str, Any]:
    selected_cases = cases or benchmark_cases()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index:02d}/{len(selected_cases):02d}] {case.case_id}: {case.title}")
        generation_error = ""
        try:
            generated = runner.generate(case.schema_sql, case.question)
            response = generated.text
            evaluation = evaluate_response(case, response)
        except Exception as exc:  # keep the remaining benchmark cases observable
            generation_error = f"{type(exc).__name__}: {exc}"
            generated = GenerationResult(text="", elapsed_seconds=0.0)
            response = ""
            evaluation = {
                "extracted_sql": "",
                "primary_failure_category": "generation_error",
                "failure_categories": ["generation_error"],
                "format_compliant": False,
                "executed": False,
                "result_correct": False,
                "strict_pass": False,
                "quality_score": 0,
                "expected_rows": [],
                "actual_rows": [],
                "error": generation_error,
                "extraction_source": "generation_error",
            }

        row = {
            "case_id": case.case_id,
            "title": case.title,
            "category": case.category,
            "difficulty": case.difficulty,
            "critical": case.critical,
            "request": case.request,
            "question": case.question.strip(),
            "schema_sql": case.schema_sql.strip(),
            "reference_sql": case.reference_sql.strip(),
            "model_response": response,
            **evaluation,
            "latency_seconds": None if generation_error else generated.elapsed_seconds,
            "time_to_first_token_seconds": generated.time_to_first_token_seconds,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "tokens_per_second": generated.tokens_per_second,
            "generation_error": generation_error,
        }
        results.append(row)
        latency_text = "n/a" if row["latency_seconds"] is None else f"{row['latency_seconds']:.2f}s"
        print(
            f"    {row['primary_failure_category']} | result_correct={row['result_correct']} "
            f"| score={row['quality_score']} | {latency_text}"
        )

    summary = summarize_cases(results)
    return {
        "artifact_type": "sql_model_scorecard",
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_suite_hash": benchmark_suite_hash(selected_cases),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "model_label": model_label,
        "model_identifier": runner.model_identifier,
        "system_prompt": SYSTEM_PROMPT,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "seed": SEED,
        },
        "load_time_seconds": runner.load_time_seconds,
        "model_size_bytes": path_size_bytes(runner.model_identifier),
        "hardware": hardware_metadata(),
        "runtime": runner.metadata(),
        "summary": summary,
        "cases": results,
    }


class TransformersSqlRunner:
    """Run Qwen chat inference through Transformers."""

    def __init__(self, model_identifier: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for the baseline benchmark") from exc

        self._torch = torch
        self.model_identifier = model_identifier
        self._device_kind = "cpu"
        torch.manual_seed(SEED)

        load_start = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if torch.cuda.is_available():
            self._device_kind = "cuda"
            self._model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device_kind = "mps"
            self._model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self._model.to("mps")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )

        self._model.eval()
        self._input_device = self._model.get_input_embeddings().weight.device
        self.load_time_seconds = time.perf_counter() - load_start

    def _synchronize(self) -> None:
        if self._device_kind == "cuda":
            self._torch.cuda.synchronize()
        elif self._device_kind == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.synchronize()

    def generate(self, schema: str, question: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema: {schema.strip()}\nQuestion: {question.strip()}"},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self._input_device) for name, tensor in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        self._synchronize()
        start = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        self._synchronize()
        elapsed = time.perf_counter() - start

        completion_ids = output[0, prompt_tokens:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_ids.shape[-1]),
        )

    def metadata(self) -> dict[str, Any]:
        torch = self._torch
        metadata: dict[str, Any] = {
            "backend": "transformers",
            "torch_version": torch.__version__,
            "device": self._device_kind,
            "input_device": str(self._input_device),
            "model_dtype": str(next(self._model.parameters()).dtype),
        }
        if self._device_kind == "cuda":
            metadata["accelerator_name"] = torch.cuda.get_device_name(0)
            metadata["accelerator_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        elif self._device_kind == "mps":
            metadata["accelerator_name"] = "Apple Metal Performance Shaders"
        return metadata

    def close(self) -> None:
        del self._model
        del self._tokenizer
        gc.collect()
        if self._device_kind == "cuda":
            self._torch.cuda.empty_cache()
        elif self._device_kind == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.empty_cache()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _csv_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "critical": case["critical"],
        "request": case["request"],
        "model_response": case["model_response"],
        "extracted_sql": case["extracted_sql"],
        "primary_failure_category": case["primary_failure_category"],
        "failure_categories": ",".join(case["failure_categories"]),
        "result_correct": case["result_correct"],
        "strict_pass": case["strict_pass"],
        "format_compliant": case["format_compliant"],
        "quality_score": case["quality_score"],
        "latency_seconds": case["latency_seconds"],
        "prompt_tokens": case["prompt_tokens"],
        "completion_tokens": case["completion_tokens"],
        "tokens_per_second": case["tokens_per_second"],
        "error": case["error"],
        "expected_rows": _json_text(case["expected_rows"]),
        "actual_rows": _json_text(case["actual_rows"]),
        "reference_sql": case["reference_sql"],
    }


CSV_FIELDNAMES = [
    "case_id",
    "title",
    "category",
    "difficulty",
    "critical",
    "request",
    "model_response",
    "extracted_sql",
    "primary_failure_category",
    "failure_categories",
    "result_correct",
    "strict_pass",
    "format_compliant",
    "quality_score",
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
    "error",
    "expected_rows",
    "actual_rows",
    "reference_sql",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown_cell(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def scorecard_markdown(scorecard: dict[str, Any], title: str) -> str:
    summary = scorecard["summary"]
    runtime = scorecard["runtime"]
    hardware = scorecard["hardware"]

    lines = [
        f"# {title}",
        "",
        "## Run",
        "",
        f"- Model: `{scorecard['model_identifier']}`",
        f"- Benchmark: `{scorecard['benchmark_version']}`",
        f"- Suite hash: `{scorecard['benchmark_suite_hash']}`",
        f"- Backend: `{runtime.get('backend', 'unknown')}`",
        f"- Device: `{runtime.get('device', 'unknown')}`",
        f"- Hardware: `{hardware.get('platform', 'unknown')}`",
        f"- Model load time: {_number(scorecard.get('load_time_seconds'))} seconds",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Execution accuracy | {_percent(summary['execution_accuracy'])} |",
        f"| Strict accuracy | {_percent(summary['strict_accuracy'])} |",
        f"| Critical-case accuracy | {_percent(summary['critical_accuracy'])} |",
        f"| Format compliance | {_percent(summary['format_compliance_rate'])} |",
        f"| Average quality score | {_number(summary['average_quality_score'], 1)} / 100 |",
        f"| Mean generation latency | {_number(summary['average_latency_seconds'])} s |",
        f"| P95 generation latency | {_number(summary['p95_latency_seconds'])} s |",
        f"| Mean generation throughput | {_number(summary['average_tokens_per_second'])} tokens/s |",
        "",
        "## Failure counts",
        "",
    ]

    if summary["failure_counts"]:
        lines.extend(["| Failure category | Count |", "|---|---:|"])
        for category, count in summary["failure_counts"].items():
            lines.append(f"| `{category}` | {count} |")
    else:
        lines.append("No failures or format violations were recorded.")

    lines.extend(
        [
            "",
            "## Case scorecard",
            "",
            "| Case | Category | Critical | Result | Failure category | Score | Latency |",
            "|---|---|:---:|:---:|---|---:|---:|",
        ]
    )
    for case in scorecard["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['case_id']}`",
                    _markdown_cell(case["category"]),
                    "yes" if case["critical"] else "no",
                    "pass" if case["result_correct"] else "fail",
                    f"`{case['primary_failure_category']}`",
                    str(case["quality_score"]),
                    f"{case['latency_seconds']:.2f}s",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "The CSV and JSON scorecards contain each full request, model response, extracted SQL, expected rows, actual rows, and failure classification.",
            "",
        ]
    )
    return "\n".join(lines)


def write_scorecard_artifacts(
    scorecard: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
    title: str,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / f"{prefix}_scorecard.json"
    csv_path = destination / f"{prefix}_scorecard.csv"
    failure_path = destination / f"{prefix}_failure_matrix.csv"
    report_path = destination / f"{prefix}_report.md"

    json_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, scorecard["cases"])
    failures = [case for case in scorecard["cases"] if case["primary_failure_category"] != "pass"]
    _write_csv(failure_path, failures)
    report_path.write_text(scorecard_markdown(scorecard, title), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "failure_matrix": failure_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the baseline Qwen2.5 model on SQL tasks.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model ID or local model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("SQL baseline benchmark")
    print("======================")
    print(f"Model      : {args.model}")
    print(f"Output dir : {BASELINE_OUTPUT_DIR.resolve()}")
    print(f"Cases      : {len(benchmark_cases())}")
    print()

    runner = TransformersSqlRunner(args.model)
    try:
        scorecard = run_benchmark(
            runner,
            run_kind="baseline",
            model_label="Qwen2.5-7B baseline",
        )
    finally:
        runner.close()

    paths = write_scorecard_artifacts(
        scorecard,
        BASELINE_OUTPUT_DIR,
        prefix="baseline",
        title="Baseline SQL Benchmark Report",
    )

    summary = scorecard["summary"]
    print("\nBaseline complete")
    print("-----------------")
    print(f"Execution accuracy : {_percent(summary['execution_accuracy'])}")
    print(f"Strict accuracy    : {_percent(summary['strict_accuracy'])}")
    print(f"Critical accuracy  : {_percent(summary['critical_accuracy'])}")
    for name, path in paths.items():
        print(f"{name:15}: {path.resolve()}")


if __name__ == "__main__":
    main()
