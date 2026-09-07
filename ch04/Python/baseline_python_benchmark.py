# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

#!/usr/bin/env python3
"""Benchmark the baseline Qwen2.5-7B-Instruct model on fixed Python tasks.

The benchmark uses the same system prompt and ChatML request structure as
1_finetune.py. Each response is reduced to candidate Python code, checked for
syntax and dependency compliance, then executed against deterministic unit
tests in a resource-limited subprocess.

The script writes:

- benchmark_results/baseline/baseline_scorecard.json
- benchmark_results/baseline/baseline_scorecard.csv
- benchmark_results/baseline/baseline_failure_matrix.csv
- benchmark_results/baseline/baseline_report.md

Usage:
  python baseline_python_benchmark.py
  python baseline_python_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


SYSTEM_PROMPT = (
    "You are a senior Python engineer and code assistant. "
    "Answer the user's question accurately. When helpful, include Python code."
)
BENCHMARK_VERSION = "python-specialist-v1"
MAX_NEW_TOKENS = 512
SEED = 3407
TEST_TIMEOUT_SECONDS = 6
BASELINE_OUTPUT_DIR = Path("./benchmark_results/baseline")
RESULT_MARKER = "__PYTHON_BENCHMARK_RESULT__="

DANGEROUS_IMPORTS = {
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
DANGEROUS_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "quit",
    "exit",
    "__import__",
}
DANGEROUS_ATTRIBUTES = {
    "__bases__",
    "__builtins__",
    "__code__",
    "__func__",
    "__globals__",
    "__mro__",
    "__subclasses__",
}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    category: str
    difficulty: str
    critical: bool
    question: str
    expected_outcome: str
    required_symbols: tuple[str, ...]
    allowed_imports: tuple[str, ...]
    tests_source: str
    reference_code: str

    @property
    def request(self) -> str:
        allowed = ", ".join(self.allowed_imports) if self.allowed_imports else "none"
        return (
            f"{self.question.strip()}\n\n"
            f"Allowed imports: {allowed}. Do not use third-party packages.\n"
            "Return only executable Python code. Do not include Markdown fences, examples, or explanation."
        )


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


class PythonGenerator(Protocol):
    model_identifier: str
    load_time_seconds: float

    def generate(self, question: str) -> GenerationResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


def benchmark_cases() -> list[BenchmarkCase]:
    """Return the fixed Python evaluation suite."""

    return [
        BenchmarkCase(
            case_id="P01_COPY_APPEND",
            title="Mutable-default avoidance and input immutability",
            category="function_semantics",
            difficulty="basic",
            critical=False,
            question="""
Write a function `append_item(item, items=None) -> list`.

Requirements:
- When `items` is `None`, start from a new empty list.
- Return a new list containing the existing values followed by `item`.
- Never mutate a caller-provided list.
- Calls that omit `items` must never share state.
""",
            expected_outcome="All calls return independent lists, and caller-owned inputs remain unchanged.",
            required_symbols=("append_item",),
            allowed_imports=(),
            tests_source="""
def test_default_state_is_not_shared():
    assert candidate.append_item(1) == [1]
    assert candidate.append_item(2) == [2]


def test_input_is_not_mutated():
    source = [1, 2]
    result = candidate.append_item(3, source)
    assert source == [1, 2]
    assert result == [1, 2, 3]
    assert result is not source


def test_empty_input_is_copied():
    source = []
    result = candidate.append_item("x", source)
    assert result == ["x"]
    assert source == []
    assert isinstance(result, list)


check("default_state_is_not_shared", test_default_state_is_not_shared)
check("input_is_not_mutated", test_input_is_not_mutated)
check("empty_input_is_copied", test_empty_input_is_copied)
""",
            reference_code="""
def append_item(item, items=None):
    result = [] if items is None else list(items)
    result.append(item)
    return result
""",
        ),
        BenchmarkCase(
            case_id="P02_STABLE_UNIQUE",
            title="Stable deduplication with unhashable values",
            category="collections",
            difficulty="intermediate",
            critical=False,
            question="""
Write a function `stable_unique(items) -> list` that removes duplicates while preserving the first occurrence.

Requirements:
- Accept any iterable, including a one-shot generator.
- Values may be unhashable, including lists and dictionaries.
- Determine duplicates using equality, not object identity.
- Preserve input order.
""",
            expected_outcome="The result contains the first equal value in order and supports unhashable inputs.",
            required_symbols=("stable_unique",),
            allowed_imports=(),
            tests_source="""
def test_hashable_values():
    assert candidate.stable_unique([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_unhashable_values():
    values = [[1], [1], {"a": 1}, {"a": 1}, [2]]
    assert candidate.stable_unique(values) == [[1], {"a": 1}, [2]]


def test_one_shot_iterable():
    consumed = []

    def source():
        for value in ["a", "b", "a", "c"]:
            consumed.append(value)
            yield value

    assert candidate.stable_unique(source()) == ["a", "b", "c"]
    assert consumed == ["a", "b", "a", "c"]


def test_equality_not_identity():
    class Value:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Value) and self.value == other.value

    first = Value(7)
    duplicate = Value(7)
    result = candidate.stable_unique([first, duplicate])
    assert result == [first]
    assert result[0] is first


check("hashable_values", test_hashable_values)
check("unhashable_values", test_unhashable_values)
check("one_shot_iterable", test_one_shot_iterable)
check("equality_not_identity", test_equality_not_identity)
""",
            reference_code="""
def stable_unique(items):
    result = []
    for item in items:
        if not any(item == existing for existing in result):
            result.append(item)
    return result
""",
        ),
        BenchmarkCase(
            case_id="P03_CHUNKED_ITERATOR",
            title="Lazy chunking of one-shot iterators",
            category="iterators",
            difficulty="intermediate",
            critical=True,
            question="""
Write a function `chunked(iterable, size)` that returns an iterator of tuples.

Requirements:
- Each tuple contains at most `size` values.
- The final tuple may be shorter.
- Consume the input lazily and support one-shot iterators.
- Raise `ValueError` when `size < 1`.
- Do not materialize the full input.
""",
            expected_outcome=(
                "Chunks are produced lazily, in order, without losing values or consuming "
                "the full input early."
            ),
            required_symbols=("chunked",),
            allowed_imports=("itertools",),
            tests_source="""
def test_chunk_shapes():
    assert list(candidate.chunked([1, 2, 3, 4, 5], 2)) == [(1, 2), (3, 4), (5,)]
    assert list(candidate.chunked([], 3)) == []


def test_lazy_one_shot_iterator():
    class TrackingIterator:
        def __init__(self):
            self.current = 0
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.current >= 5:
                raise StopIteration
            value = self.current
            self.current += 1
            return value

    source = TrackingIterator()
    chunks = candidate.chunked(source, 2)
    assert iter(chunks) is chunks
    assert source.next_calls == 0
    assert next(chunks) == (0, 1)
    assert source.next_calls == 2
    assert list(chunks) == [(2, 3), (4,)]


def test_invalid_size():
    try:
        list(candidate.chunked([1], 0))
    except ValueError:
        return
    raise AssertionError("size=0 did not raise ValueError")


check("chunk_shapes", test_chunk_shapes)
check("lazy_one_shot_iterator", test_lazy_one_shot_iterator)
check("invalid_size", test_invalid_size)
""",
            reference_code="""
from itertools import islice


def chunked(iterable, size):
    if size < 1:
        raise ValueError("size must be positive")
    iterator = iter(iterable)
    while True:
        chunk = tuple(islice(iterator, size))
        if not chunk:
            return
        yield chunk
""",
        ),
        BenchmarkCase(
            case_id="P04_DEEP_MERGE",
            title="Recursive mapping merge without shared mutable state",
            category="data_structures",
            difficulty="advanced",
            critical=True,
            question="""
Write a function `deep_merge(left, right) -> dict` for mapping objects.

Requirements:
- Recursively merge values only when both values are mappings.
- Otherwise, the value from `right` replaces the value from `left`.
- Lists are replaced, not concatenated.
- Preserve keys that appear in only one input.
- Do not mutate either input.
- The returned nested dictionaries and lists must not share mutable objects with either input.
- Support non-string keys.
""",
            expected_outcome=(
                "Nested mappings merge recursively, replacements follow right-hand precedence, "
                "and outputs are independent copies."
            ),
            required_symbols=("deep_merge",),
            allowed_imports=("collections", "copy"),
            tests_source="""
from types import MappingProxyType


def test_recursive_merge():
    left = {"db": {"host": "a", "ports": [1, 2]}, "enabled": True}
    right = {"db": {"host": "b", "user": "sam"}, "enabled": False}
    assert candidate.deep_merge(left, right) == {
        "db": {"host": "b", "ports": [1, 2], "user": "sam"},
        "enabled": False,
    }


def test_inputs_and_nested_values_are_independent():
    left = {"nested": {"items": [1], "left": {"x": 1}}}
    right = {"nested": {"right": {"y": 2}}}
    result = candidate.deep_merge(left, right)
    result["nested"]["items"].append(9)
    result["nested"]["left"]["x"] = 8
    result["nested"]["right"]["y"] = 7
    assert left == {"nested": {"items": [1], "left": {"x": 1}}}
    assert right == {"nested": {"right": {"y": 2}}}


def test_mapping_and_non_string_keys():
    left = MappingProxyType({1: {"a": 1}, 2: "left"})
    right = MappingProxyType({1: {"b": 2}, 3: "right"})
    assert candidate.deep_merge(left, right) == {1: {"a": 1, "b": 2}, 2: "left", 3: "right"}


def test_lists_are_replaced():
    left = {"values": [1, 2]}
    right = {"values": [3]}
    result = candidate.deep_merge(left, right)
    assert result == {"values": [3]}
    assert result["values"] is not right["values"]


check("recursive_merge", test_recursive_merge)
check("inputs_and_nested_values_are_independent", test_inputs_and_nested_values_are_independent)
check("mapping_and_non_string_keys", test_mapping_and_non_string_keys)
check("lists_are_replaced", test_lists_are_replaced)
""",
            reference_code="""
from collections.abc import Mapping
from copy import deepcopy


def deep_merge(left, right):
    result = deepcopy(dict(left))
    for key, right_value in right.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(right_value, Mapping):
            result[key] = deep_merge(result[key], right_value)
        else:
            result[key] = deepcopy(right_value)
    return result
""",
        ),
        BenchmarkCase(
            case_id="P05_TOPOLOGICAL_SORT",
            title="Deterministic dependency ordering and cycle detection",
            category="algorithms",
            difficulty="advanced",
            critical=True,
            question="""
Write `topological_sort(graph) -> list[str]`.

`graph` maps a node name to an iterable of node names that it depends on.

Requirements:
- Include nodes that appear only as dependencies.
- Return dependencies before dependents.
- When multiple nodes are available, choose the lexicographically smallest name.
- Ignore duplicate dependency entries.
- Raise `ValueError` when the graph contains a cycle.
- Do not mutate the input.
""",
            expected_outcome="The output is a deterministic dependency-first ordering, with cycles rejected.",
            required_symbols=("topological_sort",),
            allowed_imports=("collections", "heapq"),
            tests_source="""
def test_dependency_order_and_tie_breaking():
    graph = {
        "deploy": ["package", "test"],
        "package": ["build"],
        "test": ["build"],
        "build": ["fetch"],
    }
    assert candidate.topological_sort(graph) == ["fetch", "build", "package", "test", "deploy"]


def test_independent_and_dependency_only_nodes():
    assert candidate.topological_sort({"b": [], "a": [], "c": ["z"]}) == ["a", "b", "z", "c"]


def test_duplicate_dependencies_and_input_immutability():
    graph = {"b": ["a", "a"], "a": []}
    snapshot = {key: list(value) for key, value in graph.items()}
    assert candidate.topological_sort(graph) == ["a", "b"]
    assert graph == snapshot


def test_cycle_detection():
    for graph in ({"a": ["a"]}, {"a": ["b"], "b": ["a"]}):
        try:
            candidate.topological_sort(graph)
        except ValueError:
            continue
        raise AssertionError(f"cycle was not rejected: {graph}")


check("dependency_order_and_tie_breaking", test_dependency_order_and_tie_breaking)
check("independent_and_dependency_only_nodes", test_independent_and_dependency_only_nodes)
check("duplicate_dependencies_and_input_immutability", test_duplicate_dependencies_and_input_immutability)
check("cycle_detection", test_cycle_detection)
""",
            reference_code="""
import heapq


def topological_sort(graph):
    nodes = set(graph)
    dependencies = {}
    for node, raw_dependencies in graph.items():
        dependencies[node] = set(raw_dependencies)
        nodes.update(dependencies[node])
    for node in nodes:
        dependencies.setdefault(node, set())

    dependents = {node: set() for node in nodes}
    indegree = {node: len(dependencies[node]) for node in nodes}
    for node, node_dependencies in dependencies.items():
        for dependency in node_dependencies:
            dependents[dependency].add(node)

    ready = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for dependent in dependents[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(result) != len(nodes):
        raise ValueError("graph contains a cycle")
    return result
""",
        ),
        BenchmarkCase(
            case_id="P06_RETRY_DECORATOR",
            title="Selective retry decorator with metadata preservation",
            category="decorators",
            difficulty="advanced",
            critical=False,
            question="""
Write a decorator factory `retry(attempts, exceptions=(Exception,))`.

Requirements:
- `attempts` is the total number of calls, not the number of retries.
- Raise `ValueError` when `attempts < 1`.
- Retry only exceptions matched by `exceptions`.
- Return immediately after a successful call.
- Re-raise the final matching exception after all attempts fail.
- Preserve the wrapped function's name and docstring.
- Support positional and keyword arguments.
""",
            expected_outcome=(
                "Matching failures are retried exactly as configured, unrelated failures pass "
                "through, and function metadata is preserved."
            ),
            required_symbols=("retry",),
            allowed_imports=("functools",),
            tests_source="""
def test_success_after_retries():
    calls = []

    @candidate.retry(3, (ValueError,))
    def sometimes(value=5):
        calls.append(value)
        if len(calls) < 3:
            raise ValueError("try again")
        return value * 2

    assert sometimes(value=7) == 14
    assert calls == [7, 7, 7]


def test_non_matching_exception_is_not_retried():
    calls = []

    @candidate.retry(4, (ValueError,))
    def fail():
        calls.append(1)
        raise TypeError("wrong type")

    try:
        fail()
    except TypeError as exc:
        assert str(exc) == "wrong type"
    else:
        raise AssertionError("TypeError was swallowed")
    assert len(calls) == 1


def test_final_exception_and_attempt_validation():
    calls = []

    @candidate.retry(2, (RuntimeError,))
    def fail():
        calls.append(1)
        raise RuntimeError("final")

    try:
        fail()
    except RuntimeError as exc:
        assert str(exc) == "final"
    else:
        raise AssertionError("final exception was not raised")
    assert len(calls) == 2

    try:
        candidate.retry(0)
    except ValueError:
        return
    raise AssertionError("attempts=0 did not raise ValueError")


def test_metadata_is_preserved():
    @candidate.retry(1)
    def documented():
        '''kept docstring'''
        return 9

    assert documented() == 9
    assert documented.__name__ == "documented"
    assert documented.__doc__ == "kept docstring"


check("success_after_retries", test_success_after_retries)
check("non_matching_exception_is_not_retried", test_non_matching_exception_is_not_retried)
check("final_exception_and_attempt_validation", test_final_exception_and_attempt_validation)
check("metadata_is_preserved", test_metadata_is_preserved)
""",
            reference_code="""
from functools import wraps


def retry(attempts, exceptions=(Exception,)):
    if attempts < 1:
        raise ValueError("attempts must be at least one")

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return function(*args, **kwargs)
                except exceptions:
                    if attempt == attempts - 1:
                        raise
            raise RuntimeError("unreachable")

        return wrapped

    return decorate
""",
        ),
        BenchmarkCase(
            case_id="P07_TEMPORARY_ATTRIBUTE",
            title="Exception-safe temporary attribute context manager",
            category="context_managers",
            difficulty="intermediate",
            critical=False,
            question="""
Write a context manager function `temporary_attribute(obj, name, value)`.

Requirements:
- Set `obj.<name>` to `value` while inside the context.
- Yield `obj` from the context manager.
- Restore the exact previous value on exit.
- If the attribute did not exist before entry, remove it on exit.
- Restore state even when the context body raises.
- Nested uses for the same attribute must restore in stack order.
""",
            expected_outcome=(
                "The attribute is present only for the intended scope and is restored correctly "
                "after normal or exceptional exits."
            ),
            required_symbols=("temporary_attribute",),
            allowed_imports=("contextlib",),
            tests_source="""
def test_existing_attribute_restored():
    class Item:
        value = "class-value"

    item = Item()
    item.value = "original"
    with candidate.temporary_attribute(item, "value", "temporary") as yielded:
        assert yielded is item
        assert item.value == "temporary"
    assert item.value == "original"


def test_missing_attribute_removed():
    class Item:
        pass

    item = Item()
    assert not hasattr(item, "token")
    with candidate.temporary_attribute(item, "token", 7):
        assert item.token == 7
    assert not hasattr(item, "token")


def test_exception_restores_state():
    class Item:
        pass

    item = Item()
    item.value = 1
    try:
        with candidate.temporary_attribute(item, "value", 2):
            raise RuntimeError("body failed")
    except RuntimeError:
        pass
    assert item.value == 1


def test_nested_contexts_restore_in_order():
    class Item:
        pass

    item = Item()
    item.value = "base"
    with candidate.temporary_attribute(item, "value", "outer"):
        assert item.value == "outer"
        with candidate.temporary_attribute(item, "value", "inner"):
            assert item.value == "inner"
        assert item.value == "outer"
    assert item.value == "base"


check("existing_attribute_restored", test_existing_attribute_restored)
check("missing_attribute_removed", test_missing_attribute_removed)
check("exception_restores_state", test_exception_restores_state)
check("nested_contexts_restore_in_order", test_nested_contexts_restore_in_order)
""",
            reference_code="""
from contextlib import contextmanager


_MISSING = object()


@contextmanager
def temporary_attribute(obj, name, value):
    previous = getattr(obj, name, _MISSING)
    setattr(obj, name, value)
    try:
        yield obj
    finally:
        if previous is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, previous)
""",
        ),
        BenchmarkCase(
            case_id="P08_ASYNC_MAP_ORDERED",
            title="Bounded asynchronous mapping with ordered results",
            category="asyncio",
            difficulty="advanced",
            critical=True,
            question="""
Write `async def async_map_ordered(func, items, limit)`.

Requirements:
- `func(item)` returns an awaitable.
- Run at most `limit` calls concurrently.
- Preserve input order in the returned list, regardless of completion order.
- Support one-shot iterables.
- Return an empty list for empty input.
- Raise `ValueError` when `limit < 1`.
- If any call fails, cancel and await unfinished work before re-raising that exception.
""",
            expected_outcome=(
                "Async work respects the concurrency bound, returns ordered results, and cleans "
                "up unfinished tasks on failure."
            ),
            required_symbols=("async_map_ordered",),
            allowed_imports=("asyncio",),
            tests_source="""
import asyncio


def test_order_and_concurrency_limit():
    async def scenario():
        active = 0
        maximum = 0

        async def worker(value):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01 * (4 - value))
                return value * 10
            finally:
                active -= 1

        result = await candidate.async_map_ordered(worker, (value for value in [1, 2, 3]), 2)
        assert result == [10, 20, 30]
        assert maximum <= 2

    asyncio.run(scenario())


def test_empty_and_invalid_limit():
    async def scenario():
        async def worker(value):
            return value

        assert await candidate.async_map_ordered(worker, [], 3) == []
        try:
            await candidate.async_map_ordered(worker, [1], 0)
        except ValueError:
            return
        raise AssertionError("limit=0 did not raise ValueError")

    asyncio.run(scenario())


def test_failure_cleans_up_tasks():
    async def scenario():
        class MarkerError(RuntimeError):
            pass

        active = 0

        async def worker(value):
            nonlocal active
            active += 1
            try:
                if value == 2:
                    await asyncio.sleep(0.01)
                    raise MarkerError("failed")
                await asyncio.sleep(0.5)
                return value
            finally:
                active -= 1

        before = set(asyncio.all_tasks())
        try:
            await candidate.async_map_ordered(worker, [1, 2, 3], 3)
        except MarkerError as exc:
            assert str(exc) == "failed"
        else:
            raise AssertionError("worker failure was not propagated")

        await asyncio.sleep(0)
        leaked = [task for task in asyncio.all_tasks() if task not in before and not task.done()]
        assert leaked == []
        assert active == 0

    asyncio.run(scenario())


check("order_and_concurrency_limit", test_order_and_concurrency_limit)
check("empty_and_invalid_limit", test_empty_and_invalid_limit)
check("failure_cleans_up_tasks", test_failure_cleans_up_tasks)
""",
            reference_code="""
import asyncio


async def async_map_ordered(func, items, limit):
    if limit < 1:
        raise ValueError("limit must be positive")

    semaphore = asyncio.Semaphore(limit)

    async def run_one(index, item):
        async with semaphore:
            return index, await func(item)

    tasks = [asyncio.create_task(run_one(index, item)) for index, item in enumerate(items)]
    try:
        pairs = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [value for _, value in sorted(pairs)]
""",
        ),
        BenchmarkCase(
            case_id="P09_POSITIVE_INT_DESCRIPTOR",
            title="Per-instance validating descriptor",
            category="object_model",
            difficulty="advanced",
            critical=False,
            question="""
Write a descriptor class `PositiveInt`.

Requirements:
- Use `__set_name__` to create per-attribute storage on each instance.
- Accept positive integers only.
- Reject `bool` and non-integer values with `TypeError`.
- Reject zero and negative integers with `ValueError`.
- Reading through an instance returns its stored value.
- Reading through the owner class returns the descriptor itself.
- Values stored on different instances must remain independent.
""",
            expected_outcome=(
                "The descriptor validates assignments and stores independent values per instance "
                "while supporting class-level access."
            ),
            required_symbols=("PositiveInt",),
            allowed_imports=(),
            tests_source="""
def test_storage_and_class_access():
    class Inventory:
        quantity = candidate.PositiveInt()

        def __init__(self, quantity):
            self.quantity = quantity

    first = Inventory(2)
    second = Inventory(7)
    assert first.quantity == 2
    assert second.quantity == 7
    first.quantity = 4
    assert first.quantity == 4
    assert second.quantity == 7
    assert isinstance(Inventory.quantity, candidate.PositiveInt)


def test_type_validation():
    class Inventory:
        quantity = candidate.PositiveInt()

    item = Inventory()
    for value in (True, False, 1.5, "3", None):
        try:
            item.quantity = value
        except TypeError:
            continue
        raise AssertionError(f"non-integer value accepted: {value!r}")


def test_value_validation():
    class Inventory:
        quantity = candidate.PositiveInt()

    item = Inventory()
    for value in (0, -1, -100):
        try:
            item.quantity = value
        except ValueError:
            continue
        raise AssertionError(f"non-positive value accepted: {value!r}")


check("storage_and_class_access", test_storage_and_class_access)
check("type_validation", test_type_validation)
check("value_validation", test_value_validation)
""",
            reference_code="""
class PositiveInt:
    def __set_name__(self, owner, name):
        self.storage_name = f"_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("value must be an integer")
        if value <= 0:
            raise ValueError("value must be positive")
        setattr(instance, self.storage_name, value)
""",
        ),
        BenchmarkCase(
            case_id="P10_TTL_CACHE",
            title="Deterministic TTL cache with lazy expiration",
            category="stateful_classes",
            difficulty="advanced",
            critical=True,
            question="""
Write a class `TTLCache` with this interface:

- `TTLCache(ttl_seconds, clock=None)`
- `set(key, value) -> None`
- `get(key, default=None)`
- `purge_expired() -> int`
- `key in cache`

Requirements:
- Raise `ValueError` when `ttl_seconds <= 0`.
- Use `time.monotonic` when `clock` is omitted; otherwise call the supplied zero-argument clock.
- A value expires when `clock() >= insertion_time + ttl_seconds`.
- `set` replaces the value and resets its expiration.
- `get` and membership checks lazily remove expired entries.
- Stored values may be `None`; membership must not confuse `None` with a missing key.
- `purge_expired` removes all expired entries and returns the number removed.
""",
            expected_outcome=(
                "Entries expire at the specified boundary, overwrite operations reset TTL, and "
                "lookup semantics distinguish missing keys from stored None."
            ),
            required_symbols=("TTLCache",),
            allowed_imports=("time",),
            tests_source="""
class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_expiration_boundary_and_default():
    clock = Clock()
    cache = candidate.TTLCache(5, clock=clock)
    cache.set("a", 1)
    clock.value = 4.999
    assert cache.get("a") == 1
    assert "a" in cache
    clock.value = 5.0
    assert cache.get("a", "missing") == "missing"
    assert "a" not in cache


def test_none_value_and_overwrite_reset():
    clock = Clock()
    cache = candidate.TTLCache(3, clock=clock)
    cache.set("none", None)
    assert "none" in cache
    assert cache.get("none", "missing") is None

    cache.set("key", "old")
    clock.value = 2.0
    cache.set("key", "new")
    clock.value = 4.9
    assert cache.get("key") == "new"
    clock.value = 5.0
    assert cache.get("key", "gone") == "gone"


def test_purge_and_validation():
    clock = Clock()
    cache = candidate.TTLCache(2, clock=clock)
    cache.set("a", 1)
    clock.value = 1.0
    cache.set("b", 2)
    clock.value = 2.0
    assert cache.purge_expired() == 1
    assert "a" not in cache
    assert "b" in cache
    clock.value = 3.0
    assert cache.purge_expired() == 1

    for ttl in (0, -1):
        try:
            candidate.TTLCache(ttl, clock=clock)
        except ValueError:
            continue
        raise AssertionError(f"invalid ttl accepted: {ttl}")


check("expiration_boundary_and_default", test_expiration_boundary_and_default)
check("none_value_and_overwrite_reset", test_none_value_and_overwrite_reset)
check("purge_and_validation", test_purge_and_validation)
""",
            reference_code="""
import time


class TTLCache:
    def __init__(self, ttl_seconds, clock=None):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.monotonic
        self.entries = {}

    def set(self, key, value):
        self.entries[key] = (value, self.clock() + self.ttl_seconds)

    def get(self, key, default=None):
        entry = self.entries.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        if self.clock() >= expires_at:
            del self.entries[key]
            return default
        return value

    def __contains__(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return False
        _, expires_at = entry
        if self.clock() >= expires_at:
            del self.entries[key]
            return False
        return True

    def purge_expired(self):
        now = self.clock()
        expired = [key for key, (_, expires_at) in self.entries.items() if now >= expires_at]
        for key in expired:
            del self.entries[key]
        return len(expired)
""",
        ),
        BenchmarkCase(
            case_id="P11_COALESCE_INTERVALS",
            title="Interval normalization with touching-boundary semantics",
            category="algorithms",
            difficulty="intermediate",
            critical=False,
            question="""
Write `coalesce_intervals(intervals) -> list[tuple]`.

Each input item is a two-value `(start, end)` half-open interval.

Requirements:
- Accept any iterable, including a generator.
- Raise `ValueError` when any `end < start`.
- Sort by start and then end.
- Merge intervals that overlap or touch, meaning `next_start <= current_end`.
- Preserve zero-length intervals unless they merge with another interval.
- Do not mutate caller-owned interval objects.
- Return a new list of tuples.
""",
            expected_outcome=(
                "Intervals are validated, sorted, and coalesced with deterministic treatment of "
                "touching and zero-length ranges."
            ),
            required_symbols=("coalesce_intervals",),
            allowed_imports=(),
            tests_source="""
def test_overlap_and_touching():
    values = [(5, 7), (1, 3), (3, 4), (6, 8), (10, 11)]
    assert candidate.coalesce_intervals(values) == [(1, 4), (5, 8), (10, 11)]


def test_generator_and_zero_length():
    values = ((start, end) for start, end in [(2, 2), (5, 5), (2, 4), (8, 8)])
    assert candidate.coalesce_intervals(values) == [(2, 4), (5, 5), (8, 8)]


def test_numeric_values_and_input_immutability():
    values = [[2.5, 4.0], [1.0, 2.5]]
    snapshot = [list(value) for value in values]
    assert candidate.coalesce_intervals(values) == [(1.0, 4.0)]
    assert values == snapshot


def test_invalid_interval():
    try:
        candidate.coalesce_intervals([(3, 2)])
    except ValueError:
        return
    raise AssertionError("end < start did not raise ValueError")


check("overlap_and_touching", test_overlap_and_touching)
check("generator_and_zero_length", test_generator_and_zero_length)
check("numeric_values_and_input_immutability", test_numeric_values_and_input_immutability)
check("invalid_interval", test_invalid_interval)
""",
            reference_code="""
def coalesce_intervals(intervals):
    ordered = []
    for start, end in intervals:
        if end < start:
            raise ValueError("interval end precedes start")
        ordered.append((start, end))
    ordered.sort(key=lambda interval: (interval[0], interval[1]))

    result = []
    for start, end in ordered:
        if not result or start > result[-1][1]:
            result.append((start, end))
        else:
            previous_start, previous_end = result[-1]
            result[-1] = (previous_start, max(previous_end, end))
    return result
""",
        ),
        BenchmarkCase(
            case_id="P12_TO_JSONABLE",
            title="Recursive standard-library object normalization",
            category="standard_library",
            difficulty="advanced",
            critical=True,
            question="""
Write `to_jsonable(value)` that recursively converts supported Python values into values accepted by `json.dumps`.

Support these values:
- `None`, `str`, `int`, `float`, and `bool` unchanged.
- `enum.Enum` instances by recursively converting `.value`.
- Dataclass instances as dictionaries of field names to converted values.
- `datetime.datetime`, `datetime.date`, and `datetime.time` using `isoformat()`.
- `decimal.Decimal` as a string without losing trailing zeros.
- Mapping objects as dictionaries, but raise `TypeError` for non-string keys.
- Lists and tuples as lists.
- Sets and frozensets as deterministically sorted lists, sorting converted values by `repr`.
- Raise `TypeError` for unsupported values.
""",
            expected_outcome=(
                "Supported standard-library objects become deterministic JSON-compatible "
                "structures, while unsupported values and non-string mapping keys are rejected."
            ),
            required_symbols=("to_jsonable",),
            allowed_imports=("collections", "dataclasses", "datetime", "decimal", "enum"),
            tests_source="""
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
import json


class Status(Enum):
    READY = "ready"


@dataclass
class Payload:
    status: Status
    amount: Decimal
    when: date
    tags: set


def test_nested_dataclass_and_json_compatibility():
    value = Payload(Status.READY, Decimal("1.20"), date(2026, 8, 16), {"z", "a"})
    result = candidate.to_jsonable(value)
    assert result == {
        "status": "ready",
        "amount": "1.20",
        "when": "2026-08-16",
        "tags": ["a", "z"],
    }
    json.dumps(result)


def test_datetime_time_tuple_and_frozenset():
    value = {
        "timestamp": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        "clock": time(6, 7, 8),
        "values": (1, frozenset({3, 2})),
    }
    assert candidate.to_jsonable(value) == {
        "timestamp": "2026-01-02T03:04:05+00:00",
        "clock": "06:07:08",
        "values": [1, [2, 3]],
    }


def test_bad_mapping_key():
    try:
        candidate.to_jsonable({1: "value"})
    except TypeError:
        return
    raise AssertionError("non-string mapping key was accepted")


def test_unsupported_value():
    try:
        candidate.to_jsonable(object())
    except TypeError:
        return
    raise AssertionError("unsupported object was accepted")


check("nested_dataclass_and_json_compatibility", test_nested_dataclass_and_json_compatibility)
check("datetime_time_tuple_and_frozenset", test_datetime_time_tuple_and_frozenset)
check("bad_mapping_key", test_bad_mapping_key)
check("unsupported_value", test_unsupported_value)
""",
            reference_code="""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum


def to_jsonable(value):
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")
            result[key] = to_jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=repr)
    raise TypeError(f"unsupported value: {type(value).__name__}")
""",
        ),
    ]


def benchmark_suite_hash(cases: list[BenchmarkCase]) -> str:
    payload = json.dumps([asdict(case) for case in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fenced_blocks(response: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"```\s*([A-Za-z0-9_+.-]*)\s*\n(.*?)```", flags=re.DOTALL)
    return [(match.group(1).strip().lower(), match.group(2).strip()) for match in pattern.finditer(response)]


def _required_symbol_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"(?:async\s+def|def|class)\s+{re.escape(symbol)}\b")


def _longest_parseable_suffix(lines: list[str], start: int) -> str | None:
    for end in range(len(lines), start, -1):
        candidate = "\n".join(lines[start:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return None


def extract_python_code(response: str, required_symbols: tuple[str, ...]) -> dict[str, Any]:
    value = response.strip()
    blocks = _fenced_blocks(value)
    if blocks:
        preferred = [block for block in blocks if block[0] in {"python", "py"}]
        pool = preferred or [block for block in blocks if block[0] == ""] or blocks

        def rank(block: tuple[str, str]) -> tuple[int, int]:
            code = block[1]
            symbol_hits = sum(1 for symbol in required_symbols if _required_symbol_pattern(symbol).search(code))
            return symbol_hits, len(code)

        language, code = max(pool, key=rank)
        return {
            "code": code.strip(),
            "source": f"fenced:{language or 'unlabeled'}",
            "had_fence": True,
            "block_count": len(blocks),
        }

    if value:
        try:
            ast.parse(value)
            return {"code": value, "source": "full_response", "had_fence": False, "block_count": 0}
        except SyntaxError:
            pass

    lines = value.splitlines()
    start_pattern = re.compile(r"^\s*(?:@|async\s+def\s+|def\s+|class\s+|from\s+|import\s+)")
    for start, line in enumerate(lines):
        if not start_pattern.match(line):
            continue
        candidate = _longest_parseable_suffix(lines, start)
        if candidate is not None:
            return {
                "code": candidate,
                "source": "unfenced_recovery",
                "had_fence": False,
                "block_count": 0,
            }

    return {"code": value, "source": "unparsed_response", "had_fence": False, "block_count": 0}


def _import_root(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def analyze_code(case: BenchmarkCase, code: str) -> dict[str, Any]:
    if not code.strip():
        return {
            "syntax_valid": False,
            "syntax_error": "No Python code was extracted.",
            "imported_modules": [],
            "dependency_compliant": False,
            "dependency_violations": [],
            "safety_compliant": False,
            "safety_violations": ["no_code"],
            "defined_symbols": [],
            "required_symbols_present": False,
        }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_valid": False,
            "syntax_error": f"{exc.msg} at line {exc.lineno}, column {exc.offset}",
            "imported_modules": [],
            "dependency_compliant": False,
            "dependency_violations": [],
            "safety_compliant": False,
            "safety_violations": [],
            "defined_symbols": [],
            "required_symbols_present": False,
        }

    imported_modules: list[str] = []
    dependency_violations: list[str] = []
    safety_violations: list[str] = []
    defined_symbols: list[str] = []
    allowed_roots = {_import_root(name) for name in case.allowed_imports}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                imported_modules.append(alias.name)
                if root not in allowed_roots:
                    dependency_violations.append(alias.name)
                if root in DANGEROUS_IMPORTS:
                    safety_violations.append(f"dangerous_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = _import_root(node.module)
            module_name = node.module or ""
            imported_modules.append(module_name)
            if node.level:
                dependency_violations.append(f"relative_import:{'.' * node.level}{module_name}")
            elif root not in allowed_roots:
                dependency_violations.append(module_name)
            if root in DANGEROUS_IMPORTS:
                safety_violations.append(f"dangerous_import:{module_name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.col_offset == 0:
            defined_symbols.append(node.name)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in DANGEROUS_CALLS:
                safety_violations.append(f"dangerous_call:{name}")
        elif isinstance(node, ast.Attribute) and node.attr in DANGEROUS_ATTRIBUTES:
            safety_violations.append(f"dangerous_attribute:{node.attr}")
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            safety_violations.append("dangerous_name:__builtins__")

    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        safety_violations.append(f"top_level_execution:{type(statement).__name__}")

    required_present = all(symbol in defined_symbols for symbol in case.required_symbols)
    return {
        "syntax_valid": True,
        "syntax_error": "",
        "imported_modules": sorted(set(imported_modules)),
        "dependency_compliant": not dependency_violations,
        "dependency_violations": sorted(set(dependency_violations)),
        "safety_compliant": not safety_violations,
        "safety_violations": sorted(set(safety_violations)),
        "defined_symbols": sorted(set(defined_symbols)),
        "required_symbols_present": required_present,
    }


def detect_cross_language_contamination(response: str) -> bool:
    patterns = [
        r"```(?:javascript|typescript|java|c\+\+|cpp|csharp|rust|go|sql)\b",
        r"\b(?:public\s+static\s+void|console\.log|SELECT\s+.+\s+FROM|#include\s*<|fn\s+\w+\s*\()",
        r"\b(?:const|let|var)\s+\w+\s*=.*;",
    ]
    return any(re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _test_harness(tests_source: str) -> str:
    header = f"""import importlib.util
import json
import sys
import traceback

RESULT_MARKER = {RESULT_MARKER!r}
candidate_path = sys.argv[1]
results = []


def apply_resource_limits():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
            memory_limit = 1_073_741_824
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except (ImportError, OSError, ValueError):
        pass


apply_resource_limits()


def check(name, function):
    try:
        function()
    except BaseException as exc:
        results.append({{
            "name": name,
            "passed": False,
            "error": f"{{type(exc).__name__}}: {{exc}}",
            "traceback": traceback.format_exc(limit=5),
        }})
    else:
        results.append({{"name": name, "passed": True, "error": "", "traceback": ""}})


try:
    spec = importlib.util.spec_from_file_location("candidate_solution", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create module spec")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
except BaseException as exc:
    payload = {{
        "import_error": f"{{type(exc).__name__}}: {{exc}}",
        "import_traceback": traceback.format_exc(limit=8),
        "tests": results,
    }}
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True))
    raise SystemExit(0)
"""
    footer = (
        'print(RESULT_MARKER + json.dumps({"import_error": "", '
        '"import_traceback": "", "tests": results}, sort_keys=True))\n'
    )
    return (
        header.strip()
        + "\n\n"
        + textwrap.dedent(tests_source).strip()
        + "\n\n"
        + footer.strip()
        + "\n"
    )


def execute_tests(case: BenchmarkCase, code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="python-benchmark-") as directory:
        root = Path(directory)
        candidate_path = root / "candidate_solution.py"
        harness_path = root / "test_harness.py"
        candidate_path.write_text(code + "\n", encoding="utf-8")
        harness_path.write_text(_test_harness(case.tests_source), encoding="utf-8")

        kwargs: dict[str, Any] = {
            "args": [sys.executable, "-I", str(harness_path), str(candidate_path)],
            "cwd": str(root),
            "capture_output": True,
            "text": True,
            "timeout": TEST_TIMEOUT_SECONDS,
            "check": False,
        }
        started = time.perf_counter()
        try:
            completed = subprocess.run(**kwargs)
        except subprocess.TimeoutExpired as exc:
            return {
                "executed": False,
                "timed_out": True,
                "runtime_seconds": time.perf_counter() - started,
                "return_code": None,
                "tests": [],
                "tests_passed": 0,
                "tests_total": 0,
                "test_pass_rate": 0.0,
                "import_error": "",
                "error": f"Test process exceeded {TEST_TIMEOUT_SECONDS} seconds.",
                "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }

        elapsed = time.perf_counter() - started
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        payload: dict[str, Any] | None = None
        for line in reversed(stdout.splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER) :])
                except json.JSONDecodeError:
                    payload = None
                break

        if payload is None:
            error = f"Test harness returned code {completed.returncode} without a result payload."
            if stderr.strip():
                error += f" stderr: {stderr.strip()[-1000:]}"
            return {
                "executed": False,
                "timed_out": False,
                "runtime_seconds": elapsed,
                "return_code": completed.returncode,
                "tests": [],
                "tests_passed": 0,
                "tests_total": 0,
                "test_pass_rate": 0.0,
                "import_error": "",
                "error": error,
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
            }

        tests = payload.get("tests", []) if isinstance(payload, dict) else []
        passed = sum(1 for test in tests if test.get("passed") is True)
        total = len(tests)
        import_error = str(payload.get("import_error", ""))
        return {
            "executed": not import_error,
            "timed_out": False,
            "runtime_seconds": elapsed,
            "return_code": completed.returncode,
            "tests": tests,
            "tests_passed": passed,
            "tests_total": total,
            "test_pass_rate": passed / total if total else 0.0,
            "import_error": import_error,
            "error": import_error,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }


def _format_compliant(response: str, extraction: dict[str, Any]) -> bool:
    return (
        bool(response.strip())
        and not extraction["had_fence"]
        and extraction["source"] == "full_response"
        and response.strip() == extraction["code"].strip()
    )


def evaluate_response(case: BenchmarkCase, response: str) -> dict[str, Any]:
    extraction = extract_python_code(response, case.required_symbols)
    code = extraction["code"]
    analysis = analyze_code(case, code)
    format_compliant = _format_compliant(response, extraction)
    contamination = detect_cross_language_contamination(response)

    execution = {
        "executed": False,
        "timed_out": False,
        "runtime_seconds": None,
        "return_code": None,
        "tests": [],
        "tests_passed": 0,
        "tests_total": 0,
        "test_pass_rate": 0.0,
        "import_error": "",
        "error": "",
        "stdout": "",
        "stderr": "",
    }
    if (
        analysis["syntax_valid"]
        and analysis["dependency_compliant"]
        and analysis["safety_compliant"]
        and analysis["required_symbols_present"]
    ):
        execution = execute_tests(case, code)

    result_correct = (
        execution["executed"]
        and execution["tests_total"] > 0
        and execution["tests_passed"] == execution["tests_total"]
    )
    strict_pass = result_correct and format_compliant

    failures: list[str] = []
    if not code.strip():
        failures.append("no_python_code")
    if contamination:
        failures.append("cross_language_contamination")
    if not analysis["syntax_valid"]:
        failures.append("syntax_error")
    if analysis["syntax_valid"] and not analysis["dependency_compliant"]:
        failures.append("dependency_violation")
    if analysis["syntax_valid"] and not analysis["safety_compliant"]:
        failures.append("unsafe_code")
    if analysis["syntax_valid"] and not analysis["required_symbols_present"]:
        failures.append("missing_required_symbol")
    if execution["timed_out"]:
        failures.append("test_timeout")
    elif analysis["syntax_valid"] and analysis["dependency_compliant"] and analysis["safety_compliant"]:
        if execution["import_error"]:
            failures.append("runtime_error")
        elif execution["executed"] and not result_correct:
            failures.append("unit_test_failure")
        elif not execution["executed"] and analysis["required_symbols_present"]:
            failures.append("runtime_error")
    if not format_compliant:
        failures.append("format_violation")

    failures = list(dict.fromkeys(failures))
    primary_failure = failures[0] if failures else "pass"

    quality_score = 0.0
    quality_score += 5.0 if code.strip() else 0.0
    quality_score += 10.0 if analysis["syntax_valid"] else 0.0
    quality_score += 10.0 if analysis["dependency_compliant"] else 0.0
    quality_score += 10.0 if analysis["safety_compliant"] else 0.0
    quality_score += 10.0 if analysis["required_symbols_present"] else 0.0
    quality_score += 50.0 * float(execution["test_pass_rate"])
    quality_score += 5.0 if format_compliant else 0.0

    if execution["timed_out"]:
        actual_outcome = "Test execution timed out."
    elif execution["import_error"]:
        actual_outcome = f"Candidate import failed: {execution['import_error']}"
    elif execution["tests_total"]:
        actual_outcome = f"{execution['tests_passed']} of {execution['tests_total']} unit tests passed."
    elif analysis["syntax_error"]:
        actual_outcome = analysis["syntax_error"]
    else:
        actual_outcome = execution["error"] or "Candidate was not executed."

    return {
        "extracted_code": code,
        "extraction_source": extraction["source"],
        "format_compliant": format_compliant,
        "cross_language_contamination": contamination,
        **analysis,
        **execution,
        "result_correct": result_correct,
        "strict_pass": strict_pass,
        "quality_score": round(quality_score, 1),
        "primary_failure_category": primary_failure,
        "failure_categories": failures or ["pass"],
        "actual_outcome": actual_outcome,
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
    correct = sum(1 for row in case_results if row["result_correct"])
    strict = sum(1 for row in case_results if row["strict_pass"])
    executed = sum(1 for row in case_results if row["executed"])
    syntax_valid = sum(1 for row in case_results if row["syntax_valid"])
    dependency_compliant = sum(1 for row in case_results if row["dependency_compliant"])
    safety_compliant = sum(1 for row in case_results if row["safety_compliant"])
    symbol_compliant = sum(1 for row in case_results if row["required_symbols_present"])
    format_compliant = sum(1 for row in case_results if row["format_compliant"])
    contamination = sum(1 for row in case_results if row["cross_language_contamination"])

    critical = [row for row in case_results if row["critical"]]
    critical_correct = sum(1 for row in critical if row["result_correct"])

    tests_passed = sum(int(row["tests_passed"]) for row in case_results)
    tests_total = sum(int(row["tests_total"]) for row in case_results)
    latencies = [float(row["latency_seconds"]) for row in case_results if row["latency_seconds"] is not None]
    first_tokens = [
        float(row["time_to_first_token_seconds"])
        for row in case_results
        if row["time_to_first_token_seconds"] is not None
    ]
    throughput = [float(row["tokens_per_second"]) for row in case_results if row["tokens_per_second"] is not None]
    prompt_tokens = [int(row["prompt_tokens"]) for row in case_results if row["prompt_tokens"] is not None]
    completion_tokens = [int(row["completion_tokens"]) for row in case_results if row["completion_tokens"] is not None]
    scores = [float(row["quality_score"]) for row in case_results]

    failure_counts: dict[str, int] = {}
    for row in case_results:
        category = row["primary_failure_category"]
        if category != "pass":
            failure_counts[category] = failure_counts.get(category, 0) + 1

    return {
        "total_cases": total,
        "executed_cases": executed,
        "result_correct_cases": correct,
        "strict_pass_cases": strict,
        "syntax_valid_cases": syntax_valid,
        "dependency_compliant_cases": dependency_compliant,
        "safety_compliant_cases": safety_compliant,
        "required_symbol_cases": symbol_compliant,
        "format_compliant_cases": format_compliant,
        "cross_language_contamination_cases": contamination,
        "case_accuracy": correct / total if total else 0.0,
        "strict_accuracy": strict / total if total else 0.0,
        "syntax_valid_rate": syntax_valid / total if total else 0.0,
        "dependency_compliance_rate": dependency_compliant / total if total else 0.0,
        "safety_compliance_rate": safety_compliant / total if total else 0.0,
        "required_symbol_rate": symbol_compliant / total if total else 0.0,
        "format_compliance_rate": format_compliant / total if total else 0.0,
        "cross_language_contamination_rate": contamination / total if total else 0.0,
        "critical_cases": len(critical),
        "critical_correct_cases": critical_correct,
        "critical_accuracy": critical_correct / len(critical) if critical else 0.0,
        "individual_tests_passed": tests_passed,
        "individual_tests_total": tests_total,
        "individual_test_pass_rate": tests_passed / tests_total if tests_total else 0.0,
        "average_quality_score": statistics.fmean(scores) if scores else 0.0,
        "average_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "average_time_to_first_token_seconds": statistics.fmean(first_tokens) if first_tokens else None,
        "average_tokens_per_second": statistics.fmean(throughput) if throughput else None,
        "average_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
        "average_completion_tokens": statistics.fmean(completion_tokens) if completion_tokens else None,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def current_rss_bytes() -> int | None:
    if platform.system() == "Linux":
        try:
            fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
        )
        value = completed.stdout.strip()
        return int(value) * 1024 if value else None
    except (OSError, ValueError):
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
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
    }


def run_benchmark(
    runner: PythonGenerator,
    *,
    run_kind: str,
    model_label: str,
    cases: list[BenchmarkCase] | None = None,
) -> dict[str, Any]:
    selected_cases = cases or benchmark_cases()
    results: list[dict[str, Any]] = []
    rss_after_load = current_rss_bytes()

    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index:02d}/{len(selected_cases):02d}] {case.case_id}: {case.title}")
        generation_error = ""
        try:
            generated = runner.generate(case.request)
            response = generated.text
            evaluation = evaluate_response(case, response)
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            generated = GenerationResult(text="", elapsed_seconds=0.0)
            response = ""
            evaluation = {
                "extracted_code": "",
                "extraction_source": "generation_error",
                "format_compliant": False,
                "cross_language_contamination": False,
                "syntax_valid": False,
                "syntax_error": "",
                "imported_modules": [],
                "dependency_compliant": False,
                "dependency_violations": [],
                "safety_compliant": False,
                "safety_violations": [],
                "defined_symbols": [],
                "required_symbols_present": False,
                "executed": False,
                "timed_out": False,
                "runtime_seconds": None,
                "return_code": None,
                "tests": [],
                "tests_passed": 0,
                "tests_total": 0,
                "test_pass_rate": 0.0,
                "import_error": "",
                "error": generation_error,
                "stdout": "",
                "stderr": "",
                "result_correct": False,
                "strict_pass": False,
                "quality_score": 0.0,
                "primary_failure_category": "generation_error",
                "failure_categories": ["generation_error"],
                "actual_outcome": generation_error,
            }

        row = {
            "case_id": case.case_id,
            "title": case.title,
            "category": case.category,
            "difficulty": case.difficulty,
            "critical": case.critical,
            "request": case.request,
            "question": case.question.strip(),
            "expected_outcome": case.expected_outcome,
            "required_symbols": list(case.required_symbols),
            "allowed_imports": list(case.allowed_imports),
            "reference_code": textwrap.dedent(case.reference_code).strip(),
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
            f"    {row['primary_failure_category']} | tests={row['tests_passed']}/{row['tests_total']} "
            f"| score={row['quality_score']:.1f} | {latency_text}"
        )

    summary = summarize_cases(results)
    return {
        "artifact_type": "python_model_scorecard",
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
        "execution_policy": {
            "test_timeout_seconds": TEST_TIMEOUT_SECONDS,
            "static_dependency_allowlist": True,
            "static_safety_screen": True,
            "isolated_python_flag": "-I",
            "secure_sandbox": False,
        },
        "load_time_seconds": runner.load_time_seconds,
        "model_size_bytes": path_size_bytes(runner.model_identifier),
        "rss_after_load_bytes": rss_after_load,
        "rss_after_benchmark_bytes": current_rss_bytes(),
        "hardware": hardware_metadata(),
        "runtime": runner.metadata(),
        "summary": summary,
        "cases": results,
    }


class TransformersPythonRunner:
    """Run Qwen chat inference through Transformers."""

    def __init__(self, model_identifier: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for the Transformers benchmark") from exc

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

    def generate(self, question: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
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
            metadata["accelerator_memory_allocated_bytes"] = torch.cuda.memory_allocated(0)
        elif self._device_kind == "mps":
            metadata["accelerator_name"] = "Apple Metal Performance Shaders"
            if hasattr(torch.mps, "current_allocated_memory"):
                metadata["accelerator_memory_allocated_bytes"] = torch.mps.current_allocated_memory()
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
        "extracted_code": case["extracted_code"],
        "primary_failure_category": case["primary_failure_category"],
        "failure_categories": ",".join(case["failure_categories"]),
        "result_correct": case["result_correct"],
        "strict_pass": case["strict_pass"],
        "format_compliant": case["format_compliant"],
        "syntax_valid": case["syntax_valid"],
        "dependency_compliant": case["dependency_compliant"],
        "safety_compliant": case["safety_compliant"],
        "required_symbols_present": case["required_symbols_present"],
        "cross_language_contamination": case["cross_language_contamination"],
        "tests_passed": case["tests_passed"],
        "tests_total": case["tests_total"],
        "test_pass_rate": case["test_pass_rate"],
        "quality_score": case["quality_score"],
        "latency_seconds": case["latency_seconds"],
        "time_to_first_token_seconds": case["time_to_first_token_seconds"],
        "prompt_tokens": case["prompt_tokens"],
        "completion_tokens": case["completion_tokens"],
        "tokens_per_second": case["tokens_per_second"],
        "expected_outcome": case["expected_outcome"],
        "actual_outcome": case["actual_outcome"],
        "error": case["error"],
        "syntax_error": case["syntax_error"],
        "imported_modules": _json_text(case["imported_modules"]),
        "dependency_violations": _json_text(case["dependency_violations"]),
        "safety_violations": _json_text(case["safety_violations"]),
        "test_results": _json_text(case["tests"]),
        "reference_code": case["reference_code"],
    }


CSV_FIELDNAMES = [
    "case_id",
    "title",
    "category",
    "difficulty",
    "critical",
    "request",
    "model_response",
    "extracted_code",
    "primary_failure_category",
    "failure_categories",
    "result_correct",
    "strict_pass",
    "format_compliant",
    "syntax_valid",
    "dependency_compliant",
    "safety_compliant",
    "required_symbols_present",
    "cross_language_contamination",
    "tests_passed",
    "tests_total",
    "test_pass_rate",
    "quality_score",
    "latency_seconds",
    "time_to_first_token_seconds",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
    "expected_outcome",
    "actual_outcome",
    "error",
    "syntax_error",
    "imported_modules",
    "dependency_violations",
    "safety_violations",
    "test_results",
    "reference_code",
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


def _bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{value} B"


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
        f"- Process RSS after load: {_bytes(scorecard.get('rss_after_load_bytes'))}",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Unit-test case accuracy | {_percent(summary['case_accuracy'])} |",
        f"| Individual unit-test pass rate | {_percent(summary['individual_test_pass_rate'])} |",
        f"| Strict accuracy | {_percent(summary['strict_accuracy'])} |",
        f"| Critical-case accuracy | {_percent(summary['critical_accuracy'])} |",
        f"| Syntax-valid rate | {_percent(summary['syntax_valid_rate'])} |",
        f"| Dependency compliance | {_percent(summary['dependency_compliance_rate'])} |",
        f"| Safety-screen compliance | {_percent(summary['safety_compliance_rate'])} |",
        f"| Required-symbol compliance | {_percent(summary['required_symbol_rate'])} |",
        f"| Format compliance | {_percent(summary['format_compliance_rate'])} |",
        f"| Cross-language contamination | {_percent(summary['cross_language_contamination_rate'])} |",
        f"| Average quality score | {_number(summary['average_quality_score'], 1)} / 100 |",
        f"| Mean generation latency | {_number(summary['average_latency_seconds'])} s |",
        f"| P95 generation latency | {_number(summary['p95_latency_seconds'])} s |",
        f"| Mean generation throughput | {_number(summary['average_tokens_per_second'])} tokens/s |",
        f"| Mean completion length | {_number(summary['average_completion_tokens'], 1)} tokens |",
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
            "| Case | Category | Critical | Unit tests | Result | Failure category | Score | Latency |",
            "|---|---|:---:|---:|:---:|---|---:|---:|",
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
                    f"{case['tests_passed']}/{case['tests_total']}",
                    "pass" if case["result_correct"] else "fail",
                    f"`{case['primary_failure_category']}`",
                    f"{case['quality_score']:.1f}",
                    "n/a" if case["latency_seconds"] is None else f"{case['latency_seconds']:.2f}s",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            (
                "The CSV and JSON scorecards contain every full request, model response, "
                "extracted code, unit-test result, expected outcome, actual outcome, and "
                "failure classification."
            ),
            "",
            (
                "Generated code is statically screened and run in an isolated, resource-limited "
                "subprocess. This reduces accidental damage but is not a secure sandbox; use a "
                "container or virtual machine for untrusted model output."
            ),
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
    parser = argparse.ArgumentParser(description="Benchmark the baseline Qwen2.5 model on Python coding tasks.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model ID or local model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Python baseline benchmark")
    print("=========================")
    print(f"Model      : {args.model}")
    print(f"Output dir : {BASELINE_OUTPUT_DIR.resolve()}")
    print(f"Cases      : {len(benchmark_cases())}")
    print()

    runner = TransformersPythonRunner(args.model)
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
        title="Baseline Python Benchmark Report",
    )

    summary = scorecard["summary"]
    print("\nBaseline complete")
    print("-----------------")
    print(f"Case accuracy      : {_percent(summary['case_accuracy'])}")
    print(f"Individual tests   : {_percent(summary['individual_test_pass_rate'])}")
    print(f"Strict accuracy    : {_percent(summary['strict_accuracy'])}")
    print(f"Critical accuracy  : {_percent(summary['critical_accuracy'])}")
    for name, path in paths.items():
        print(f"{name:15}: {path.resolve()}")


if __name__ == "__main__":
    main()
