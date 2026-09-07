# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

#!/usr/bin/env python3
"""Benchmark the merged Python specialist against the frozen baseline.

The benchmark reuses the fixed cases and evaluator from
baseline_python_benchmark.py. It compares two model states:

1. The baseline Qwen2.5-7B scorecard.
2. The merged Python specialist in ./merged_qwen_python.

Run the baseline benchmark first:
  python baseline_python_benchmark.py

Then run the specialist benchmark:
  python specialist_python_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline_python_benchmark import (
    BASELINE_OUTPUT_DIR,
    BENCHMARK_VERSION,
    TransformersPythonRunner,
    benchmark_cases,
    benchmark_suite_hash,
    run_benchmark,
    write_scorecard_artifacts,
)


BASELINE_SCORECARD = BASELINE_OUTPUT_DIR / "baseline_scorecard.json"
DEFAULT_MODEL = "./merged_qwen_python"
OUTPUT_DIR = Path("./benchmark_results/specialist")

def load_baseline_scorecard() -> dict[str, Any]:
    """Load the baseline and verify that both runs use the same test suite."""

    if not BASELINE_SCORECARD.is_file():
        raise FileNotFoundError(
            f"Baseline scorecard not found: {BASELINE_SCORECARD.resolve()}\n"
            "Run: python baseline_python_benchmark.py"
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
            "The baseline scorecard used a different benchmark suite. "
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
    """Compare the merged specialist with the frozen baseline."""

    baseline_cases = _case_map(baseline)
    specialist_cases = _case_map(specialist)

    case_comparison: list[dict[str, Any]] = []
    improvements: list[str] = []
    regressions: list[str] = []
    critical_regressions: list[str] = []

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
                "baseline_test_pass_rate": baseline_case["test_pass_rate"],
                "specialist_test_pass_rate": specialist_case["test_pass_rate"],
                "test_pass_rate_delta": _delta(
                    specialist_case["test_pass_rate"],
                    baseline_case["test_pass_rate"],
                ),
                "baseline_failure_category": baseline_case["primary_failure_category"],
                "specialist_failure_category": specialist_case["primary_failure_category"],
                "baseline_quality_score": baseline_case["quality_score"],
                "specialist_quality_score": specialist_case["quality_score"],
                "quality_score_delta": _delta(
                    specialist_case["quality_score"],
                    baseline_case["quality_score"],
                ),
            }
        )

    baseline_summary = baseline["summary"]
    specialist_summary = specialist["summary"]

    return {
        "quality_change_from_baseline": {
            "case_accuracy_delta": _delta(
                specialist_summary["case_accuracy"],
                baseline_summary["case_accuracy"],
            ),
            "individual_test_pass_rate_delta": _delta(
                specialist_summary["individual_test_pass_rate"],
                baseline_summary["individual_test_pass_rate"],
            ),
            "critical_accuracy_delta": _delta(
                specialist_summary["critical_accuracy"],
                baseline_summary["critical_accuracy"],
            ),
            "syntax_valid_rate_delta": _delta(
                specialist_summary["syntax_valid_rate"],
                baseline_summary["syntax_valid_rate"],
            ),
            "dependency_compliance_rate_delta": _delta(
                specialist_summary["dependency_compliance_rate"],
                baseline_summary["dependency_compliance_rate"],
            ),
            "format_compliance_rate_delta": _delta(
                specialist_summary["format_compliance_rate"],
                baseline_summary["format_compliance_rate"],
            ),
            "quality_score_delta": _delta(
                specialist_summary["average_quality_score"],
                baseline_summary["average_quality_score"],
            ),
        },
        "improvement_case_ids": improvements,
        "regression_case_ids": regressions,
        "critical_regression_case_ids": critical_regressions,
        "case_comparison": case_comparison,
    }


def apply_acceptance_gate(
    specialist: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Pass when the merged specialist improves case accuracy."""

    baseline_accuracy = float(baseline["summary"]["case_accuracy"])
    specialist_accuracy = float(specialist["summary"]["case_accuracy"])
    accuracy_change = specialist_accuracy - baseline_accuracy
    passed = specialist_accuracy > baseline_accuracy

    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "criterion": "Specialist case accuracy must be greater than baseline case accuracy.",
        "baseline_case_accuracy": baseline_accuracy,
        "specialist_case_accuracy": specialist_accuracy,
        "case_accuracy_delta": accuracy_change,
    }


def _percent(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


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


def _markdown_text(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _case_list(case_ids: list[str]) -> str:
    if not case_ids:
        return "None"
    return ", ".join(f"`{case_id}`" for case_id in case_ids)


def build_report_markdown(report: dict[str, Any]) -> str:
    baseline = report["before"]
    specialist = report["specialist"]
    baseline_summary = baseline["summary"]
    specialist_summary = specialist["summary"]
    comparison = report["comparison"]
    change = comparison["quality_change_from_baseline"]
    gate = report["acceptance_gate"]
    hardware = specialist["hardware"]
    runtime = specialist["runtime"]

    lines = [
        "# Merged Python Specialist Benchmark Report",
        "",
        f"## Acceptance decision: **{gate['status']}**",
        "",
        f"- Specialist model: `{specialist['model_identifier']}`",
        f"- Baseline model: `{baseline['model_identifier']}`",
        f"- Benchmark: `{report['benchmark_version']}`",
        f"- Suite hash: `{report['benchmark_suite_hash']}`",
        f"- Runtime: `{runtime.get('backend', 'unknown')}`",
        f"- Execution device: `{runtime.get('device', 'unknown')}`",
        f"- Host: `{hardware.get('platform', 'unknown')}`",
        f"- Logical CPUs: `{hardware.get('logical_cpu_count', 'unknown')}`",
        f"- Total memory: `{_bytes(hardware.get('total_memory_bytes'))}`",
        "",
        "## Quality comparison",
        "",
        "| Metric | Baseline | Merged specialist | Change |",
        "|---|---:|---:|---:|",
        (
            f"| Case accuracy | {_percent(baseline_summary['case_accuracy'])} | "
            f"{_percent(specialist_summary['case_accuracy'])} | "
            f"{_percent(change['case_accuracy_delta'], signed=True)} |"
        ),
        (
            "| Individual unit-test pass rate | "
            f"{_percent(baseline_summary['individual_test_pass_rate'])} | "
            f"{_percent(specialist_summary['individual_test_pass_rate'])} | "
            f"{_percent(change['individual_test_pass_rate_delta'], signed=True)} |"
        ),
        (
            f"| Critical-case accuracy | {_percent(baseline_summary['critical_accuracy'])} | "
            f"{_percent(specialist_summary['critical_accuracy'])} | "
            f"{_percent(change['critical_accuracy_delta'], signed=True)} |"
        ),
        (
            f"| Syntax-valid rate | {_percent(baseline_summary['syntax_valid_rate'])} | "
            f"{_percent(specialist_summary['syntax_valid_rate'])} | "
            f"{_percent(change['syntax_valid_rate_delta'], signed=True)} |"
        ),
        (
            "| Dependency compliance | "
            f"{_percent(baseline_summary['dependency_compliance_rate'])} | "
            f"{_percent(specialist_summary['dependency_compliance_rate'])} | "
            f"{_percent(change['dependency_compliance_rate_delta'], signed=True)} |"
        ),
        (
            f"| Format compliance | {_percent(baseline_summary['format_compliance_rate'])} | "
            f"{_percent(specialist_summary['format_compliance_rate'])} | "
            f"{_percent(change['format_compliance_rate_delta'], signed=True)} |"
        ),
        (
            "| Average quality score | "
            f"{_number(baseline_summary['average_quality_score'], 1)} | "
            f"{_number(specialist_summary['average_quality_score'], 1)} | "
            f"{_number(change['quality_score_delta'], 1)} points |"
        ),
        "",
        "## Case-level quality changes",
        "",
        f"- Improvements over baseline: {_case_list(comparison['improvement_case_ids'])}",
        f"- Regressions from baseline: {_case_list(comparison['regression_case_ids'])}",
        (
            "- Critical regressions from baseline: "
            f"{_case_list(comparison['critical_regression_case_ids'])}"
        ),
        "",
        "| Case | Critical | Baseline tests | Specialist tests | Change | Baseline result | Specialist result |",
        "|---|:---:|---:|---:|---:|---|---|",
    ]

    for case in comparison["case_comparison"]:
        lines.append(
            f"| `{case['case_id']}` | {'yes' if case['critical'] else 'no'} | "
            f"{_percent(case['baseline_test_pass_rate'])} | "
            f"{_percent(case['specialist_test_pass_rate'])} | "
            f"{_percent(case['test_pass_rate_delta'], signed=True)} | "
            f"`{case['baseline_failure_category']}` | "
            f"`{case['specialist_failure_category']}` |"
        )

    lines.extend(
        [
            "",
            "## Specialist runtime and hardware measurements",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Model size | {_bytes(specialist.get('model_size_bytes'))} |",
            f"| Model load time | {_number(specialist.get('load_time_seconds'))} s |",
            f"| RSS after model load | {_bytes(specialist.get('rss_after_load_bytes'))} |",
            f"| RSS after benchmark | {_bytes(specialist.get('rss_after_benchmark_bytes'))} |",
            f"| Mean generation latency | {_number(specialist_summary['average_latency_seconds'])} s |",
            f"| P95 generation latency | {_number(specialist_summary['p95_latency_seconds'])} s |",
            (
                "| Mean generation throughput | "
                f"{_number(specialist_summary['average_tokens_per_second'])} tokens/s |"
            ),
            (
                "| Mean completion length | "
                f"{_number(specialist_summary['average_completion_tokens'], 1)} tokens |"
            ),
            "",
            "## Pass criterion",
            "",
            gate["criterion"],
            "",
            "| Baseline case accuracy | Specialist case accuracy | Change | Result |",
            "|---:|---:|---:|:---:|",
            (
                f"| {_percent(gate['baseline_case_accuracy'])} | "
                f"{_percent(gate['specialist_case_accuracy'])} | "
                f"{_percent(gate['case_accuracy_delta'], signed=True)} | "
                f"**{gate['status']}** |"
            ),
            "",
            "## Specialist failures",
            "",
        ]
    )
    failures = [
        case
        for case in specialist["cases"]
        if case["primary_failure_category"] != "pass"
    ]

    if failures:
        lines.extend(
            [
                "| Case | Critical | Unit tests | Failure category | Score | Error |",
                "|---|:---:|---:|---|---:|---|",
            ]
        )
        for case in failures:
            lines.append(
                f"| `{case['case_id']}` | {'yes' if case['critical'] else 'no'} | "
                f"{case['tests_passed']}/{case['tests_total']} | "
                f"`{case['primary_failure_category']}` | {case['quality_score']:.1f} | "
                f"{_markdown_text(case['error']) or '-'} |"
            )
    else:
        lines.append("No specialist failures or format violations were recorded.")

    lines.extend(
        [
            "",
            "## Evaluator scope",
            "",
            (
                "The JSON and CSV scorecards retain every request, full model response, "
                "extracted Python code, expected behavior, unit-test result, latency, token "
                "count, and failure classification."
            ),
            "",
            (
                "Generated code is checked with AST-based dependency and safety rules, then run "
                "in an isolated, resource-limited subprocess. This is an evaluation boundary, "
                "not a secure sandbox. Run untrusted model output inside a container or virtual "
                "machine."
            ),
            "",
            (
                "The benchmark measures the fixed evaluation suite shared with the baseline. "
                "It does not establish universal Python competence, and format extraction may "
                "occasionally require human review when a response contains unusual prose or "
                "multiple code blocks."
            ),
            "",
            (
                "Before describing the suite as held out, compare these prompts and reference "
                "solutions against the exact fine-tuning corpus and replace any exact or near "
                "duplicates."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def write_combined_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "specialist_benchmark_report.json"
    markdown_path = output_dir / "specialist_benchmark_report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_report_markdown(report), encoding="utf-8")

    return {"json": json_path, "report": markdown_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the merged Python specialist against the frozen baseline."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local merged Hugging Face model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = load_baseline_scorecard()

    print("Merged Python specialist benchmark")
    print("==================================")
    print(f"Specialist model   : {args.model}")
    print(f"Baseline scorecard : {BASELINE_SCORECARD.resolve()}")
    print(f"Output dir         : {OUTPUT_DIR.resolve()}")
    print(f"Cases              : {len(benchmark_cases())}")
    print()

    runner = TransformersPythonRunner(args.model)
    try:
        after = run_benchmark(
            runner,
            run_kind="merged_specialist",
            model_label="Merged Python specialist",
        )
    finally:
        runner.close()

    scorecard_paths = write_scorecard_artifacts(
        after,
        OUTPUT_DIR,
        prefix="merged_specialist",
        title="Merged Python Specialist Scorecard",
    )

    comparison = compare_scorecards(before, after)
    gate = apply_acceptance_gate(before, after)

    report = {
        "artifact_type": "python_specialist_benchmark_report",
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_suite_hash": benchmark_suite_hash(benchmark_cases()),
        "before": before,
        "specialist": after,
        "comparison": comparison,
        "acceptance_gate": gate,
    }
    report_paths = write_combined_report(report, OUTPUT_DIR)

    case_accuracy_change = comparison["quality_change_from_baseline"][
        "case_accuracy_delta"
    ]

    print("\nSpecialist benchmark complete")
    print("-----------------------------")
    print(f"Acceptance decision : {gate['status']}")
    print(f"Accuracy change     : {case_accuracy_change:+.1%}")
    print(f"Improved cases      : {len(comparison['improvement_case_ids'])}")
    print("\nArtifacts")
    artifacts = {
        **scorecard_paths,
        **{f"combined_{name}": path for name, path in report_paths.items()},
    }
    for name, path in artifacts.items():
        print(f"{name:24}: {path.resolve()}")


if __name__ == "__main__":
    main()
