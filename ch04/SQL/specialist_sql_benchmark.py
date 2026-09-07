# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

#!/usr/bin/env python3
"""Benchmark the merged SQL specialist against the frozen baseline scorecard.

The script reuses the benchmark suite and evaluator from
baseline_sql_benchmark.py. It runs the same held-out SQL cases against
./merged_qwen_sql, compares the results with the baseline scorecard, and
applies an explicit acceptance gate.

Run the baseline benchmark first:
  python baseline_sql_benchmark.py

Then run the specialist benchmark:
  python specialist_sql_benchmark.py
  python specialist_sql_benchmark.py --model ./merged_qwen_sql
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline_sql_benchmark import (
    BASELINE_OUTPUT_DIR,
    BENCHMARK_VERSION,
    TransformersSqlRunner,
    benchmark_cases,
    benchmark_suite_hash,
    run_benchmark,
    write_scorecard_artifacts,
)


BASELINE_SCORECARD = BASELINE_OUTPUT_DIR / "baseline_scorecard.json"
DEFAULT_SPECIALIST_MODEL = "./merged_qwen_sql"
SPECIALIST_OUTPUT_DIR = Path("./benchmark_results/specialist")


def load_baseline_scorecard() -> dict[str, Any]:
    """Load the baseline scorecard and verify that it uses this benchmark suite."""

    if not BASELINE_SCORECARD.is_file():
        raise FileNotFoundError(
            f"Baseline scorecard not found: {BASELINE_SCORECARD.resolve()}\n"
            "Run: python baseline_sql_benchmark.py"
        )

    payload = json.loads(BASELINE_SCORECARD.read_text(encoding="utf-8"))
    current_hash = benchmark_suite_hash(benchmark_cases())

    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(
            f"Baseline benchmark version is {payload.get('benchmark_version')!r}; "
            f"expected {BENCHMARK_VERSION!r}."
        )

    if payload.get("benchmark_suite_hash") != current_hash:
        raise ValueError(
            "Baseline scorecard used a different benchmark suite. "
            "Run the baseline benchmark again."
        )

    return payload


def _case_map(scorecard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in scorecard["cases"]}


def _delta(specialist_value: float | None, baseline_value: float | None) -> float | None:
    if specialist_value is None or baseline_value is None:
        return None
    return specialist_value - baseline_value


def compare_scorecards(
    specialist: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare the merged specialist with the frozen baseline case by case."""

    baseline_cases = _case_map(baseline)
    specialist_cases = _case_map(specialist)

    improvements: list[str] = []
    regressions: list[str] = []
    critical_regressions: list[str] = []
    case_comparison: list[dict[str, Any]] = []

    for case in benchmark_cases():
        baseline_case = baseline_cases[case.case_id]
        specialist_case = specialist_cases[case.case_id]

        if not baseline_case["result_correct"] and specialist_case["result_correct"]:
            improvements.append(case.case_id)

        if baseline_case["result_correct"] and not specialist_case["result_correct"]:
            regressions.append(case.case_id)
            if case.critical:
                critical_regressions.append(case.case_id)

        case_comparison.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "critical": case.critical,
                "baseline_result_correct": baseline_case["result_correct"],
                "specialist_result_correct": specialist_case["result_correct"],
                "baseline_failure_category": baseline_case["primary_failure_category"],
                "specialist_failure_category": specialist_case["primary_failure_category"],
                "baseline_quality_score": baseline_case["quality_score"],
                "specialist_quality_score": specialist_case["quality_score"],
            }
        )

    baseline_summary = baseline["summary"]
    specialist_summary = specialist["summary"]

    return {
        "execution_accuracy_delta": _delta(
            specialist_summary["execution_accuracy"],
            baseline_summary["execution_accuracy"],
        ),
        "critical_accuracy_delta": _delta(
            specialist_summary["critical_accuracy"],
            baseline_summary["critical_accuracy"],
        ),
        "quality_score_delta": _delta(
            specialist_summary["average_quality_score"],
            baseline_summary["average_quality_score"],
        ),
        "improvement_case_ids": improvements,
        "regression_case_ids": regressions,
        "critical_regression_case_ids": critical_regressions,
        "case_comparison": case_comparison,
    }


def apply_acceptance_gate(
    specialist: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Pass when the merged specialist improves execution accuracy."""

    baseline_accuracy = float(baseline["summary"]["execution_accuracy"])
    specialist_accuracy = float(specialist["summary"]["execution_accuracy"])
    accuracy_delta = specialist_accuracy - baseline_accuracy
    passed = accuracy_delta > 0.0

    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "criterion": "Specialist execution accuracy must exceed baseline execution accuracy.",
        "baseline_execution_accuracy": baseline_accuracy,
        "specialist_execution_accuracy": specialist_accuracy,
        "execution_accuracy_delta": accuracy_delta,
    }


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _number(value: float | None, digits: int = 2, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.{digits}f}"


def _case_list(case_ids: list[str]) -> str:
    if not case_ids:
        return "None."
    return ", ".join(f"`{case_id}`" for case_id in case_ids) + "."


def build_report_markdown(report: dict[str, Any]) -> str:
    """Build the baseline-versus-specialist report."""

    baseline = report["before"]
    specialist = report["specialist"]
    comparison = report["comparison"]
    gate = report["acceptance_gate"]

    baseline_summary = baseline["summary"]
    specialist_summary = specialist["summary"]

    lines = [
        "# SQL Specialist Benchmark Report",
        "",
        f"## Acceptance decision: **{gate['status']}**",
        "",
        f"- Specialist model: `{specialist['model_identifier']}`",
        f"- Baseline model: `{baseline['model_identifier']}`",
        f"- Benchmark: `{report['benchmark_version']}`",
        f"- Suite hash: `{report['benchmark_suite_hash']}`",
        "",
        "## Quality comparison",
        "",
        "| Metric | Baseline | Merged specialist | Change |",
        "|---|---:|---:|---:|",
        f"| Execution accuracy | {_percent(baseline_summary['execution_accuracy'])} | "
        f"{_percent(specialist_summary['execution_accuracy'])} | "
        f"{_percent(comparison['execution_accuracy_delta'], signed=True)} |",
        f"| Critical-case accuracy | {_percent(baseline_summary['critical_accuracy'])} | "
        f"{_percent(specialist_summary['critical_accuracy'])} | "
        f"{_percent(comparison['critical_accuracy_delta'], signed=True)} |",
        f"| Average quality score | {_number(baseline_summary['average_quality_score'], 1)} | "
        f"{_number(specialist_summary['average_quality_score'], 1)} | "
        f"{_number(comparison['quality_score_delta'], 1, signed=True)} |",
        f"| Format compliance | {_percent(baseline_summary['format_compliance_rate'])} | "
        f"{_percent(specialist_summary['format_compliance_rate'])} | "
        f"{_percent(_delta(specialist_summary['format_compliance_rate'], baseline_summary['format_compliance_rate']), signed=True)} |",
        "",
        "## Runtime measurements",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Model load time | {_number(specialist.get('load_time_seconds'))} s |",
        f"| Mean generation latency | {_number(specialist_summary['average_latency_seconds'])} s |",
        f"| P50 generation latency | {_number(specialist_summary['p50_latency_seconds'])} s |",
        f"| P95 generation latency | {_number(specialist_summary['p95_latency_seconds'])} s |",
        f"| Mean generation throughput | {_number(specialist_summary['average_tokens_per_second'])} tokens/s |",
        "",
        "## Pass criterion",
        "",
        "The merged specialist passes only when its execution accuracy is greater "
        "than the baseline execution accuracy. A tie does not pass.",
        "",
        "| Baseline | Merged specialist | Change | Result |",
        "|---:|---:|---:|:---:|",
        f"| {_percent(gate['baseline_execution_accuracy'])} | "
        f"{_percent(gate['specialist_execution_accuracy'])} | "
        f"{_percent(gate['execution_accuracy_delta'], signed=True)} | "
        f"{gate['status']} |",
        "",
    ]

    lines.extend(["", "## Merged specialist failures", ""])
    failures = [
        case
        for case in specialist["cases"]
        if case["primary_failure_category"] != "pass"
    ]

    if failures:
        lines.extend(
            [
                "| Case | Critical | Failure category | Result correct | Score | Error |",
                "|---|:---:|---|:---:|---:|---|",
            ]
        )
        for case in failures:
            error = str(case["error"]).replace("|", "\\|").replace("\n", " ")
            if len(error) > 100:
                error = error[:97] + "..."
            lines.append(
                f"| `{case['case_id']}` | {'yes' if case['critical'] else 'no'} | "
                f"`{case['primary_failure_category']}` | "
                f"{'yes' if case['result_correct'] else 'no'} | "
                f"{case['quality_score']} | {error or '-'} |"
            )
    else:
        lines.append("No specialist failures or format violations were recorded.")

    lines.extend(
        [
            "",
            "## Case-by-case changes",
            "",
            f"- Improved cases: {_case_list(comparison['improvement_case_ids'])}",
            f"- Regressed cases: {_case_list(comparison['regression_case_ids'])}",
            f"- Critical regressions: {_case_list(comparison['critical_regression_case_ids'])}",
            "",
            "The accompanying scorecards contain every request, full model response, "
            "extracted SQL, expected result, actual result, latency, token count, and "
            "failure classification.",
            "",
        ]
    )

    return "\n".join(lines)


def write_combined_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write the comparison report in JSON and Markdown formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "specialist_benchmark_report.json"
    markdown_path = output_dir / "specialist_benchmark_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_report_markdown(report), encoding="utf-8")

    return {
        "comparison_json": json_path,
        "comparison_report": markdown_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the merged SQL specialist against the baseline scorecard."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SPECIALIST_MODEL,
        help="Hugging Face model ID or local merged-model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = load_baseline_scorecard()

    print("SQL specialist benchmark")
    print("========================")
    print(f"Specialist model   : {args.model}")
    print(f"Baseline scorecard : {BASELINE_SCORECARD.resolve()}")
    print(f"Output dir         : {SPECIALIST_OUTPUT_DIR.resolve()}")
    print(f"Cases              : {len(benchmark_cases())}")
    print()

    runner = TransformersSqlRunner(args.model)
    try:
        after = run_benchmark(
            runner,
            run_kind="merged_specialist",
            model_label="Merged SQL specialist",
        )
    finally:
        runner.close()

    scorecard_paths = write_scorecard_artifacts(
        after,
        SPECIALIST_OUTPUT_DIR,
        prefix="merged_specialist",
        title="Merged SQL Specialist Scorecard",
    )

    comparison = compare_scorecards(before, after)
    gate = apply_acceptance_gate(before, after)

    report = {
        "artifact_type": "sql_specialist_benchmark_report",
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_suite_hash": benchmark_suite_hash(benchmark_cases()),
        "before": before,
        "specialist": after,
        "comparison": comparison,
        "acceptance_gate": gate,
    }
    report_paths = write_combined_report(report, SPECIALIST_OUTPUT_DIR)

    print("\nSpecialist benchmark complete")
    print("-----------------------------")
    print(f"Acceptance decision : {gate['status']}")
    print(
        "Specialization gain: "
        f"{comparison['execution_accuracy_delta']:+.1%}"
    )
    print("\nArtifacts")
    for name, path in {**scorecard_paths, **report_paths}.items():
        print(f"{name:20}: {path.resolve()}")


if __name__ == "__main__":
    main()
