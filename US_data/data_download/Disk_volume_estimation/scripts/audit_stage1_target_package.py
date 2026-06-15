"""
Flash-NH Stage 1 Milestone 2J-B — Target Package Auditor
=========================================================

Audits the output of build_stage1_target_package.py.

Checks per-basin NC:
  - qobs_m3s variable exists with correct units
  - date coordinate is hourly, monotonically increasing
  - no decoded -9999.0 sentinel values remain after xarray decode
  - no negative qobs values after cleaning
  - NaN counts are consistent with cleaning_report.csv

Checks package-level:
  - manifest.json, checksums.sha256, run_provenance.json all exist
  - SHA-256 checksums match actual NC files
  - operational-review basins are absent (if policy + status CSV provided)
  - special-review basins are surfaced

Exit code: 0 = PASS, 1 = FAIL

Usage:

  python scripts/audit_stage1_target_package.py \\
      --package-dir tmp/stage1_target_package_smoke \\
      --policy config/stage1_target_policy.yaml

  # Optional: provide audit status CSV to check for held-out basins
  python scripts/audit_stage1_target_package.py \\
      --package-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \\
      --policy config/stage1_target_policy.yaml \\
      --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time as _time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_NAME = pathlib.Path(__file__).name
CREATED_UTC = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

EXPECTED_UNITS = 'm3 s-1'
FILL_VALUE_SENTINEL = -9999.0


def _norm_staid(s: object) -> str:
    try:
        return f'{int(float(str(s).strip())):08d}'
    except Exception:
        return str(s).strip().zfill(8)


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_policy(policy_path: pathlib.Path) -> dict:
    with open(policy_path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _load_status_df(csv_path: pathlib.Path | None) -> pd.DataFrame | None:
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['STAID'] = df['STAID'].apply(_norm_staid)
    return df


def audit_one_basin(nc_path: pathlib.Path, cleaning_row: dict | None) -> tuple[bool, list[str]]:
    """
    Audit one target NC. Returns (ok: bool, issues: list[str]).
    """
    issues: list[str] = []
    staid = nc_path.stem  # e.g. '02073000'

    try:
        with xr.open_dataset(nc_path, mask_and_scale=True) as ds:
            # 1. Variable check
            if 'qobs_m3s' not in ds:
                issues.append(f'{staid}: missing variable qobs_m3s')
                return False, issues

            qv = ds['qobs_m3s']

            # 2. Units check
            actual_units = qv.attrs.get('units', '').strip()
            if actual_units != EXPECTED_UNITS:
                issues.append(f'{staid}: units={actual_units!r} expected {EXPECTED_UNITS!r}')

            # 3. Date coordinate check
            if 'date' not in ds.coords:
                issues.append(f'{staid}: missing date coordinate')
            else:
                dates = pd.DatetimeIndex(ds.coords['date'].values)
                n_steps = len(dates)

                if n_steps == 0:
                    issues.append(f'{staid}: date coordinate is empty')
                else:
                    # Monotonic hourly check
                    if not dates.is_monotonic_increasing:
                        issues.append(f'{staid}: date coordinate is not monotonically increasing')
                    diffs = pd.Series(dates).diff().dropna()
                    non_hourly = diffs[diffs != pd.Timedelta(hours=1)]
                    if not non_hourly.empty:
                        issues.append(
                            f'{staid}: {len(non_hourly)} non-hourly gaps in date coordinate '
                            f'(first at index {non_hourly.index[0]})'
                        )

            # 4. Decoded value checks — after xarray decode, NaN should be NaN not -9999
            vals = qv.values.astype(np.float64)
            n_sentinel_remaining = int(np.sum(vals == FILL_VALUE_SENTINEL))
            if n_sentinel_remaining > 0:
                issues.append(
                    f'{staid}: {n_sentinel_remaining} decoded values equal {FILL_VALUE_SENTINEL} '
                    f'(FillValue not properly decoded by xarray)'
                )

            # 5. No negative values after cleaning
            n_negative = int(np.sum((vals < 0) & ~np.isnan(vals)))
            if n_negative > 0:
                issues.append(f'{staid}: {n_negative} negative qobs values remain after cleaning')

            # 6. NaN count consistency with cleaning report
            n_nan = int(np.isnan(vals).sum())
            n_valid = int(np.sum(~np.isnan(vals)))
            if cleaning_row is not None:
                expected_nan = cleaning_row.get('n_nan_after', None)
                if expected_nan is not None:
                    try:
                        if int(float(expected_nan)) != n_nan:
                            issues.append(
                                f'{staid}: NaN count mismatch — cleaning_report says '
                                f'{int(float(expected_nan))}, NC has {n_nan}'
                            )
                    except (ValueError, TypeError):
                        pass

    except Exception as exc:
        issues.append(f'{staid}: failed to open NC: {exc}')
        return False, issues

    return len(issues) == 0, issues


def main() -> int:
    p = argparse.ArgumentParser(
        description='Flash-NH Stage 1 2J-B: audit target package.'
    )
    p.add_argument('--package-dir', required=True, type=pathlib.Path,
                   help='Output directory from build_stage1_target_package.py.')
    p.add_argument('--policy', required=True, type=pathlib.Path,
                   help='Path to stage1_target_policy.yaml.')
    p.add_argument('--status-csv', type=pathlib.Path, default=None,
                   help='Audit target_status.csv for held-out basin checks.')
    p.add_argument('--expected-basins', type=int, default=None,
                   help='Expected basin count (for smoke validation).')
    args = p.parse_args()

    t0 = _time.time()
    print('=' * 70)
    print('Flash-NH Stage 1 2J-B — Target Package Auditor')
    print(f'Started: {CREATED_UTC}')
    print(f'Package: {args.package_dir}')
    print('=' * 70)

    errors:   list[str] = []
    warnings: list[str] = []

    pkg_dir = args.package_dir
    ts_dir  = pkg_dir / 'time_series'

    # [1] Required files exist
    print('\n[1] Checking required package files ...')
    required_files = {
        'manifest.json':        pkg_dir / 'manifest.json',
        'checksums.sha256':     pkg_dir / 'checksums.sha256',
        'run_provenance.json':  pkg_dir / 'run_provenance.json',
        'cleaning_report.csv':  pkg_dir / 'cleaning_report.csv',
    }
    for name, path in required_files.items():
        if path.exists():
            print(f'  {name}: OK')
        else:
            errors.append(f'Missing required file: {name}')
            print(f'  {name}: MISSING')

    # Load manifest
    manifest: dict = {}
    if required_files['manifest.json'].exists():
        with open(required_files['manifest.json'], encoding='utf-8') as fh:
            manifest = json.load(fh)

    # Load cleaning report
    cleaning_map: dict[str, dict] = {}
    if required_files['cleaning_report.csv'].exists():
        cr = pd.read_csv(required_files['cleaning_report.csv'], dtype=str)
        for _, row in cr.iterrows():
            cleaning_map[_norm_staid(row['staid'])] = row.to_dict()

    # [2] Discover NC files
    print('\n[2] Discovering target NC files ...')
    nc_files = sorted(ts_dir.glob('*.nc'))
    n_nc = len(nc_files)
    print(f'  Found: {n_nc} NC files in {ts_dir}')
    if n_nc == 0:
        errors.append('No NC files found in time_series/')

    if args.expected_basins is not None:
        if n_nc != args.expected_basins:
            errors.append(f'Expected {args.expected_basins} NC files, found {n_nc}')
            print(f'  !! Count mismatch: expected {args.expected_basins}, got {n_nc}')
        else:
            print(f'  Basin count: OK ({n_nc} == {args.expected_basins})')

    # [3] Checksum verification
    print('\n[3] Verifying checksums ...')
    checksum_ok = 0
    checksum_fail = 0
    recorded: dict[str, str] = {}
    if required_files['checksums.sha256'].exists():
        with open(required_files['checksums.sha256'], encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('  ', 1)
                if len(parts) == 2:
                    h, rel_path = parts
                    fname = pathlib.Path(rel_path).name
                    recorded[fname] = h

        for nc in nc_files:
            expected_h = recorded.get(nc.name)
            if expected_h is None:
                warnings.append(f'Checksum not recorded for {nc.name}')
                continue
            actual_h = _sha256_file(nc)
            if actual_h == expected_h:
                checksum_ok += 1
            else:
                errors.append(f'Checksum mismatch for {nc.name}')
                checksum_fail += 1
        print(f'  OK: {checksum_ok}  FAIL: {checksum_fail}  '
              f'(unrecorded: {n_nc - checksum_ok - checksum_fail})')
    else:
        warnings.append('checksums.sha256 missing — checksum verification skipped')

    # [4] Per-basin NC audits
    print(f'\n[4] Auditing {n_nc} basin NCs ...')
    n_pass = 0
    n_fail_nc = 0
    all_nc_issues: list[str] = []

    for nc in nc_files:
        staid = nc.stem
        cr_row = cleaning_map.get(_norm_staid(staid))
        ok, nc_issues = audit_one_basin(nc, cr_row)
        if ok:
            n_pass += 1
        else:
            n_fail_nc += 1
            all_nc_issues.extend(nc_issues)
            errors.extend(nc_issues)

    print(f'  PASS: {n_pass}  FAIL: {n_fail_nc}')
    if all_nc_issues:
        for issue in all_nc_issues[:20]:
            print(f'  !! {issue}')
        if len(all_nc_issues) > 20:
            print(f'  ... ({len(all_nc_issues) - 20} more issues)')

    # [5] Cleaning summary
    print('\n[5] Cleaning summary ...')
    if cleaning_map:
        neg_total = sum(int(float(v.get('n_neg_cleaned', 0))) for v in cleaning_map.values())
        nan_before = sum(int(float(v.get('n_nan_before', 0))) for v in cleaning_map.values())
        nan_after  = sum(int(float(v.get('n_nan_after', 0))) for v in cleaning_map.values())
        valid_after = sum(int(float(v.get('n_valid_after', 0))) for v in cleaning_map.values())
        print(f'  Negative values cleaned (-> NaN): {neg_total}')
        print(f'  NaN before cleaning:              {nan_before}')
        print(f'  NaN after cleaning:               {nan_after}')
        print(f'  Valid hours after cleaning:        {valid_after}')
        if neg_total > 0:
            basins_with_neg = [
                s for s, v in cleaning_map.items()
                if int(float(v.get('n_neg_cleaned', 0))) > 0
            ]
            print(f'  Basins with negative values cleaned: {sorted(basins_with_neg)}')

    # [6] Policy: check for held-out operational-review basins
    print('\n[6] Policy compliance checks ...')
    policy = _load_policy(args.policy)
    excl_statuses = set(policy.get('inclusion', {}).get('exclude_status', []))
    nc_staids = {nc.stem for nc in nc_files}

    status_df = _load_status_df(args.status_csv)
    if status_df is not None and 'target_status' in status_df.columns:
        oper_review = set(
            status_df.loc[
                status_df['target_status'].isin(excl_statuses), 'STAID'
            ]
        )
        in_pkg_held = nc_staids & oper_review
        if in_pkg_held:
            errors.append(
                f'{len(in_pkg_held)} held-out basins (TARGET_OPERATIONAL_REVIEW) '
                f'found in package: {sorted(in_pkg_held)}'
            )
            print(f'  !! HELD-OUT BASINS IN PACKAGE: {sorted(in_pkg_held)}')
        else:
            n_held = len(oper_review)
            print(f'  Held-out basins absent from package: OK ({n_held} held out, 0 in package)')
    else:
        print('  [skip] No --status-csv provided — held-out basin check skipped.')

    # [7] Special-review surface report
    print('\n[7] Special-review basins ...')
    sr_basins = policy.get('special_review', {}).get('dominant_negative_qobs_basins', {})
    sr_in_pkg = [s for s in sr_basins if s in nc_staids]
    sr_absent  = [s for s in sr_basins if s not in nc_staids]
    if sr_in_pkg:
        print(f'  Special-review basins IN package (included after override):')
        for s in sr_in_pkg:
            n_neg = sr_basins[s].get('n_negative_values', '?')
            cr_row = cleaning_map.get(_norm_staid(s), {})
            cleaned = cr_row.get('n_neg_cleaned', '?')
            print(f'    {s}: original n_neg={n_neg}, cleaned={cleaned}')
        warnings.append(
            f'Special-review basins in package: {sr_in_pkg} — '
            'confirm review was performed before using for training.'
        )
    if sr_absent:
        print(f'  Special-review basins absent (excluded or not in smoke set): {sr_absent}')

    # [8] TARGET_QUALITY_REVIEW eligibility confirmation
    if status_df is not None and 'target_status' in status_df.columns:
        qr_in_pkg = set(
            status_df.loc[status_df['target_status'] == 'TARGET_QUALITY_REVIEW', 'STAID']
        ) & nc_staids
        if qr_in_pkg:
            print(f'\n[8] TARGET_QUALITY_REVIEW basins in package: {len(qr_in_pkg)} '
                  '(spike flag is advisory — these are eligible per policy)')
        else:
            print('\n[8] No TARGET_QUALITY_REVIEW basins in this package batch (OK for smoke).')
    else:
        print('\n[8] [skip] No --status-csv — TARGET_QUALITY_REVIEW eligibility check skipped.')

    # Summary
    elapsed = _time.time() - t0
    print('\n' + '=' * 70)
    overall = 'PASS' if not errors else 'FAIL'
    print(f'Audit {overall} in {elapsed:.1f}s')
    print(f'  Errors:   {len(errors)}')
    print(f'  Warnings: {len(warnings)}')
    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'  !! {e}')
    if warnings:
        print('\nWarnings:')
        for w in warnings:
            print(f'  ~~ {w}')
    print('=' * 70)

    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
