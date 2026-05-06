#!/usr/bin/env python3
"""Targeted IFS investigation for cycle availability and request parameters.

This script keeps the investigation narrow:
- sample date: 2023-01-01
- cycles: 00, 06, 12, 18
- tests: current MARS request, then minimal follow-up cases on failures

It writes a self-contained review bundle under
reports/audit_2026_04_29/run_03_ifs_investigation/review_bundle/.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import csv
import io
import json
import logging
import shutil
import sys
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasources.ifs import IfsMarsDataSource, IfsMarsConfig  # noqa: E402


RUN_ROOT = REPO_ROOT / "reports" / "audit_2026_04_29" / "run_03_ifs_investigation"
LOGS_ROOT = RUN_ROOT / "logs"
SUMMARY_ROOT = RUN_ROOT / "summary"
REQUESTS_ROOT = RUN_ROOT / "requests"
SAMPLE_ROOT = RUN_ROOT / "sample_downloads"
REVIEW_BUNDLE_ROOT = RUN_ROOT / "review_bundle"
DOCS_ROOT = REPO_ROOT / "docs"

SAMPLE_DATE = datetime(2023, 1, 1, 0, 0, 0)
CYCLE_HOURS = (0, 6, 12, 18)
SOURCE_NAME = "ifs_mars_conus"

CURRENT_PARAMS = {
    "class": "od",
    "stream": "oper",
    "type": "fc",
    "levtype": "sfc",
    "grid": "0.25/0.25",
    "area": "50/-126/24/-66",
    "param": "228.128/167.128/168.128/165.128/166.128/134.128/169.128",
    "step": "0/to/24/by/1",
}


@dataclass
class TestResult:
    cycle: int
    test_name: str
    success: bool
    expected_fields: int
    retrieved_fields: int
    output_bytes: int | None
    request: dict[str, Any]
    error_message: str | None = None
    output_file: str | None = None


def main() -> None:
    _clean_run_root()
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    REQUESTS_ROOT.mkdir(parents=True, exist_ok=True)
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)

    transcript_path = LOGS_ROOT / "run_stdout_stderr.txt"
    with transcript_path.open("w", encoding="utf-16") as transcript:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=transcript,
            force=True,
        )
        with redirect_stdout(transcript), redirect_stderr(transcript):
            print("=== IFS INVESTIGATION ===")
            print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
            print(f"Run root: {RUN_ROOT}")
            print(f"Sample date: {SAMPLE_DATE.date().isoformat()}")
            print(f"Cycles: {CYCLE_HOURS}")
            print()

            datasource = IfsMarsDataSource(max_lead_h=24, config=IfsMarsConfig())
            results: list[TestResult] = []
            cycle_logs: dict[str, list[dict[str, Any]]] = {}
            cycle_outcomes: dict[str, dict[str, Any]] = {}

            for hour in CYCLE_HOURS:
                cycle_dt = SAMPLE_DATE.replace(hour=hour)
                cycle_key = f"{hour:02d}"
                cycle_dir = SAMPLE_ROOT / cycle_key
                cycle_dir.mkdir(parents=True, exist_ok=True)
                logs: list[dict[str, Any]] = []
                cycle_logs[cycle_key] = logs

                print(f"--- Cycle {cycle_key} ---")
                print(f"Cycle datetime: {cycle_dt.isoformat()}Z")

                current_request = _build_request(datasource, cycle_dt)
                current_test = _run_request(
                    datasource=datasource,
                    cycle_dir=cycle_dir,
                    cycle=hour,
                    test_name="current_full",
                    request=current_request,
                    expected_fields=_expected_fields(current_request),
                )
                results.append(current_test)
                logs.append(_result_to_log_record(current_test))

                if current_test.success:
                    cycle_outcomes[cycle_key] = {"status": "success", "used_test": current_test.test_name}
                    continue

                minimal_tests: list[tuple[str, dict[str, Any], int]] = []
                base_minimal = dict(current_request)
                base_minimal["param"] = "2T"
                base_minimal["step"] = "0"
                minimal_tests.append(("minimal_2t_step0", base_minimal, 1))

                full_step0 = dict(current_request)
                full_step0["step"] = "0"
                minimal_tests.append(("all_vars_step0", full_step0, 7))

                temp_steps = dict(current_request)
                temp_steps["param"] = "2T"
                minimal_tests.append(("temp_2t_steps_0_24", temp_steps, 25))

                full_current = dict(current_request)
                minimal_tests.append(("all_vars_steps_0_24", full_current, 175))

                variant_tests: list[tuple[str, dict[str, Any], int]] = [
                    ("current_expver_1", {**current_request, "expver": "1"}, _expected_fields(current_request)),
                    ("current_expver_0001", {**current_request, "expver": "0001"}, _expected_fields(current_request)),
                ]

                for name, request, expected_fields in variant_tests:
                    test = _run_request(
                        datasource=datasource,
                        cycle_dir=cycle_dir,
                        cycle=hour,
                        test_name=name,
                        request=request,
                        expected_fields=expected_fields,
                    )
                    results.append(test)
                    logs.append(_result_to_log_record(test))
                    if test.success:
                        cycle_outcomes[cycle_key] = {"status": "success", "used_test": test.test_name}
                        break
                else:
                    for name, request, expected_fields in minimal_tests:
                        test = _run_request(
                            datasource=datasource,
                            cycle_dir=cycle_dir,
                            cycle=hour,
                            test_name=name,
                            request=request,
                            expected_fields=expected_fields,
                        )
                        results.append(test)
                        logs.append(_result_to_log_record(test))
                        if test.success:
                            cycle_outcomes[cycle_key] = {"status": "success", "used_test": test.test_name}
                            break
                    else:
                        cycle_outcomes[cycle_key] = {"status": "failed", "used_test": None}

                print()

            _write_summary(results, cycle_outcomes)
            _write_cycle_logs(cycle_logs)
            _write_review_bundle()
            _write_bundled_manifest(results, cycle_outcomes)

            print(f"Completed investigation run under {RUN_ROOT}")
            print(f"Review bundle: {REVIEW_BUNDLE_ROOT}")


def _clean_run_root() -> None:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)


def _build_request(datasource: IfsMarsDataSource, cycle_dt: datetime) -> dict[str, Any]:
    target_path = SAMPLE_ROOT / "_probe" / f"ifs_{cycle_dt.hour:02d}.grib"
    request = datasource._build_request(cycle_dt, target_path)
    request.pop("target", None)
    return request


def _expected_fields(request: dict[str, Any]) -> int:
    param_count = len([item for item in str(request.get("param", "")).split("/") if item])
    step_value = str(request.get("step", ""))
    if step_value == "0":
        step_count = 1
    elif step_value == "0/to/24/by/1":
        step_count = 25
    else:
        step_count = 1
    return param_count * step_count


def _run_request(
    datasource: IfsMarsDataSource,
    cycle_dir: Path,
    cycle: int,
    test_name: str,
    request: dict[str, Any],
    expected_fields: int,
) -> TestResult:
    from ecmwfapi import ECMWFService  # type: ignore[import-not-found]
    from eccodes import codes_count_in_file  # type: ignore[import-not-found]

    output_path = cycle_dir / f"{test_name}.grib"
    request_for_log = dict(request)
    request_for_log["target"] = str(output_path)
    request_for_execute = dict(request)
    request_path = REQUESTS_ROOT / f"cycle_{cycle:02d}_{test_name}.json"
    request_path.write_text(json.dumps({"cycle": cycle, "test_name": test_name, "request": request_for_log}, indent=2), encoding="utf-8")

    try:
        service = ECMWFService("mars")
        service.execute(request_for_execute, str(output_path))
        retrieved_fields = _count_grib_fields(output_path, codes_count_in_file)
        output_bytes = output_path.stat().st_size
        result = TestResult(
            cycle=cycle,
            test_name=test_name,
            success=True,
            expected_fields=expected_fields,
            retrieved_fields=retrieved_fields,
            output_bytes=output_bytes,
            request=request_for_log,
            output_file=str(output_path),
        )
        print(
            f"TEST PASS cycle={cycle:02d} test={test_name} expected_fields={expected_fields} "
            f"retrieved_fields={retrieved_fields} output_bytes={output_bytes}"
        )
        return result
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        print(
            f"TEST FAIL cycle={cycle:02d} test={test_name} expected_fields={expected_fields} "
            f"retrieved_fields=0 error={error_message}"
        )
        return TestResult(
            cycle=cycle,
            test_name=test_name,
            success=False,
            expected_fields=expected_fields,
            retrieved_fields=0,
            output_bytes=None,
            request=request_for_log,
            error_message=error_message,
        )


def _count_grib_fields(path: Path, codes_count_in_file) -> int:
    with path.open("rb") as handle:
        count = codes_count_in_file(handle)
    return int(count)


def _result_to_log_record(result: TestResult) -> dict[str, Any]:
    return {
        "cycle": f"{result.cycle:02d}",
        "test_name": result.test_name,
        "success": result.success,
        "expected_fields": result.expected_fields,
        "retrieved_fields": result.retrieved_fields,
        "output_bytes": result.output_bytes,
        "error_message": result.error_message,
        "output_file": result.output_file,
        "request": result.request,
    }


def _write_summary(results: list[TestResult], cycle_outcomes: dict[str, dict[str, Any]]) -> None:
    summary_rows = []
    for result in results:
        summary_rows.append({
            "cycle": f"{result.cycle:02d}",
            "test_name": result.test_name,
            "success": result.success,
            "expected_fields": result.expected_fields,
            "retrieved_fields": result.retrieved_fields,
            "output_bytes": result.output_bytes,
            "error_message": result.error_message,
        })

    csv_path = SUMMARY_ROOT / "ifs_investigation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["cycle", "test_name", "success", "expected_fields", "retrieved_fields", "output_bytes", "error_message"])
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = SUMMARY_ROOT / "ifs_investigation_summary.json"
    json_path.write_text(json.dumps({"results": summary_rows, "cycle_outcomes": cycle_outcomes}, indent=2), encoding="utf-8")

    md_path = SUMMARY_ROOT / "ifs_investigation_summary.md"
    lines = ["# IFS Investigation Summary", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    for cycle_key, outcome in cycle_outcomes.items():
        lines.append(f"- Cycle {cycle_key}: {outcome.get('status')} ({outcome.get('used_test')})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cycle_logs(cycle_logs: dict[str, list[dict[str, Any]]]) -> None:
    for cycle_key, entries in cycle_logs.items():
        log_path = LOGS_ROOT / f"ifs_cycle_{cycle_key}.json"
        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write_review_bundle() -> None:
    REVIEW_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "docs" / "data_pipeline_status.md", REVIEW_BUNDLE_ROOT / "data_pipeline_status.md")
    shutil.copytree(SUMMARY_ROOT, REVIEW_BUNDLE_ROOT / "summary", dirs_exist_ok=True)
    shutil.copytree(LOGS_ROOT, REVIEW_BUNDLE_ROOT / "logs", dirs_exist_ok=True)
    shutil.copytree(REQUESTS_ROOT, REVIEW_BUNDLE_ROOT / "requests", dirs_exist_ok=True)


def _write_bundled_manifest(results: list[TestResult], cycle_outcomes: dict[str, dict[str, Any]]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(RUN_ROOT),
        "review_bundle": str(REVIEW_BUNDLE_ROOT),
        "results": [asdict(result) for result in results],
        "cycle_outcomes": cycle_outcomes,
    }
    (REVIEW_BUNDLE_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()