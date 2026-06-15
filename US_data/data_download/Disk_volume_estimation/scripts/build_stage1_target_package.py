"""
Flash-NH Stage 1 Milestone 2J-B — Target-Cleaned Streamflow Package Builder
===========================================================================

Reads canonical hourly NC files (USGS IV streamflow, written by
recover_usgs_iv_full_period_hourly.py) and applies the target policy from
config/stage1_target_policy.yaml to produce a NeuralHydrology-compatible
streamflow-target package.

Policy behavior (all driven by the YAML config, not hard-coded here):
  - Include basins with historical_training_utility_flag=True (requires --status-csv)
  - Exclude TARGET_OPERATIONAL_REVIEW basins from first package (requires --status-csv)
  - TARGET_QUALITY_REVIEW basins remain eligible (spike flag is advisory)
  - Set all negative qobs values to NaN; preserve existing NaN
  - No interpolation, gap filling, or imputation
  - Special-review basins (default: 02299472, 04073468) HALT the builder
    unless --allow-review-required is passed explicitly

Without --status-csv the builder operates in permissive smoke mode: it processes
all basins found in --canonical-dir (subject to --max-basins / --staids limits)
and still enforces the negative-cleaning and special-review rules.

Output per basin (NeuralHydrology GenericDataset-compatible):
  <out-dir>/time_series/<STAID>.nc
    Coordinate: date   (hourly UTC, datetime64[ns])
    Variable:   qobs_m3s  (float32, units='m3 s-1', _FillValue=-9999.0)

Companion outputs:
  <out-dir>/manifest.json
  <out-dir>/checksums.sha256
  <out-dir>/run_provenance.json
  <out-dir>/cleaning_report.csv

Usage — local smoke (5 basins from full-period pilot, no audit CSV needed):

  python scripts/build_stage1_target_package.py \\
      --canonical-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot/canonical \\
      --policy config/stage1_target_policy.yaml \\
      --out-dir tmp/stage1_target_package_smoke \\
      --max-basins 5 \\
      --force

Usage — full h2o build (requires audit CSV from the 2,843-basin acquisition):

  python scripts/build_stage1_target_package.py \\
      --canonical-dir /data42/omrip/Flash-NH/tmp/stage1_full_2843 \\
      --policy config/stage1_target_policy.yaml \\
      --status-csv /data42/omrip/Flash-NH/tmp/stage1_full_2843/audit/target_status.csv \\
      --out-dir /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \\
      --force

Hard guardrails:
  - Does NOT run model training.
  - Does NOT download data.
  - Does NOT alter positive observations.
  - Does NOT fill gaps or interpolate.
  - Special-review basins halt the build unless explicitly overridden.
"""
from __future__ import annotations

import argparse
import csv
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


def _find_canonical_ncs(canonical_dir: pathlib.Path) -> list[pathlib.Path]:
    """Locate *_hourly.nc files in a flat directory or recursively under shard subdirs."""
    direct = sorted(canonical_dir.glob('*_hourly.nc'))
    if direct:
        return direct
    return sorted(canonical_dir.rglob('*_hourly.nc'))


def _load_policy(policy_path: pathlib.Path) -> dict:
    with open(policy_path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _load_status_df(csv_path: pathlib.Path | None) -> pd.DataFrame | None:
    """Return DataFrame with STAID/target_status/historical_training_utility_flag, or None."""
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df['STAID'] = df['STAID'].apply(_norm_staid)
    return df


def _filter_basins_by_policy(
    staids: list[str],
    status_df: pd.DataFrame | None,
    policy: dict,
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """
    Apply inclusion policy to candidate STAID list.
    Returns (included_staids, exclusion_records).
    If status_df is None, returns all staids (permissive smoke mode).
    """
    if status_df is None:
        if verbose:
            print('  [policy] No --status-csv provided — permissive smoke mode: '
                  'all candidate basins included (no historical_utility_flag filtering).')
        return list(staids), []

    excl_statuses = set(policy.get('inclusion', {}).get('exclude_status', []))
    status_map = dict(zip(status_df['STAID'], status_df.get('target_status', pd.Series(dtype=str))))
    util_map: dict[str, bool] = {}
    if 'historical_training_utility_flag' in status_df.columns:
        util_map = {
            row['STAID']: str(row['historical_training_utility_flag']).strip().lower() == 'true'
            for _, row in status_df.iterrows()
        }

    included, excluded = [], []
    for staid in staids:
        ts = status_map.get(staid, 'UNKNOWN')
        util = util_map.get(staid, True)  # unknown → default include
        if not util:
            excluded.append({'staid': staid, 'reason': 'historical_training_utility_flag=False', 'target_status': ts})
        elif ts in excl_statuses:
            excluded.append({'staid': staid, 'reason': f'target_status={ts} is in exclude_status list', 'target_status': ts})
        else:
            included.append(staid)

    if verbose and excluded:
        print(f'  [policy] Excluded {len(excluded)} basins by status/utility policy.')
    return included, excluded


def _check_special_review(
    staid: str,
    policy: dict,
    allow_review_required: set[str],
) -> str:
    """
    Return 'halt', 'exclude', 'include', or 'pass' for this STAID.
    'halt' → builder must stop unless STAID in allow_review_required.
    """
    sr = policy.get('special_review', {}).get('dominant_negative_qobs_basins', {})
    if staid not in sr:
        return 'pass'
    action = sr[staid].get('first_package_action', 'review_required')
    n_neg  = sr[staid].get('n_negative_values', '?')
    if action == 'review_required':
        if staid in allow_review_required:
            print(f'  [special-review] {staid}: review_required OVERRIDDEN via --allow-review-required '
                  f'(n_neg={n_neg}). Including with negative→NaN cleaning.')
            return 'include'
        print(f'\n  !! SPECIAL-REVIEW HALT: {staid} has first_package_action=review_required '
              f'and {n_neg} negative qobs values.')
        print(f'     Include with --allow-review-required {staid} to override.')
        print(f'     Exclude with --exclude-staids {staid} to skip it.\n')
        return 'halt'
    if action == 'exclude':
        print(f'  [special-review] {staid}: action=exclude — skipping.')
        return 'exclude'
    # include_after_nan_clamp or any other explicit include action
    print(f'  [special-review] {staid}: action={action!r} — including with negative→NaN cleaning.')
    return 'include'


def process_one_basin(
    nc_path: pathlib.Path,
    staid: str,
    policy: dict,
    out_ts_dir: pathlib.Path,
    force: bool,
) -> dict:
    """Load canonical NC, clean, write target NC. Return per-basin result dict."""
    out_nc = out_ts_dir / f'{staid}.nc'
    if out_nc.exists() and not force:
        return {
            'staid': staid, 'status': 'SKIP_EXISTS',
            'n_neg_cleaned': 0, 'n_nan_before': 0,
            'n_nan_after': 0, 'n_valid_after': 0,
            'n_hours': 0, 'period_start_utc': '', 'period_end_utc': '',
        }

    with xr.open_dataset(nc_path) as ds_in:
        # Accept either 'streamflow' (from acquisition) or 'qobs_m3s' (already packaged)
        if 'streamflow' in ds_in:
            sf_var = ds_in['streamflow']
        elif 'qobs_m3s' in ds_in:
            sf_var = ds_in['qobs_m3s']
        else:
            return {
                'staid': staid, 'status': 'ERROR_NO_SF_VAR',
                'n_neg_cleaned': 0, 'n_nan_before': 0,
                'n_nan_after': 0, 'n_valid_after': 0,
                'n_hours': 0, 'period_start_utc': '', 'period_end_utc': '',
            }

        # Accept 'time' or 'date' as the time coordinate name
        tc_name = 'time' if 'time' in ds_in.coords else 'date'
        time_coord = pd.DatetimeIndex(ds_in.coords[tc_name].values)
        if time_coord.tz is not None:
            time_coord = time_coord.tz_convert('UTC').tz_localize(None)

        values = sf_var.values.astype(np.float64)
        src_attrs = dict(sf_var.attrs)
        period_start = src_attrs.get('period_start_utc', str(time_coord[0]))[:19] + 'Z'
        period_end   = src_attrs.get('period_end_utc',   str(time_coord[-1]))[:19] + 'Z'

    n_hours      = len(values)
    n_nan_before = int(np.isnan(values).sum())

    # Apply cleaning: negative → NaN
    cleaning = policy.get('target_cleaning', {})
    n_neg_cleaned = 0
    if cleaning.get('set_negative_to_nan', True):
        neg_mask = (values < 0) & ~np.isnan(values)
        n_neg_cleaned = int(neg_mask.sum())
        values[neg_mask] = np.nan

    n_nan_after   = int(np.isnan(values).sum())
    n_valid_after = n_hours - n_nan_after

    # Build output dataset (NeuralHydrology GenericDataset convention: 'date' coord)
    ref_time = pd.Timestamp(time_coord[0]).strftime('%Y-%m-%d %H:%M:%S')
    ds_out = xr.Dataset(
        {'qobs_m3s': (['date'], values.astype(np.float32))},
        coords={'date': time_coord.values},
        attrs={
            'Conventions':              'CF-1.8',
            'staid':                    staid,
            'STAID':                    staid,
            'gauge_id':                 staid,
            'milestone':                'Flash-NH Stage 1 2J-B target package',
            'policy_name':              policy.get('policy_name', 'unknown'),
            'policy_version':           str(policy.get('policy_version', 'unknown')),
            'n_neg_cleaned':            n_neg_cleaned,
            'n_nan_before_cleaning':    n_nan_before,
            'n_nan_after_cleaning':     n_nan_after,
            'n_valid_after_cleaning':   n_valid_after,
            'source_nc':                nc_path.name,
            'created_utc':              CREATED_UTC,
            'history':                  f'Created {CREATED_UTC} by {SCRIPT_NAME} (2J-B)',
        },
    )
    ds_out['qobs_m3s'].attrs = {
        'units':             'm3 s-1',
        'long_name':         'Observed streamflow (USGS NWIS IV, cleaned per target policy)',
        'source_product':    'USGS NWIS Instantaneous Values',
        'cleaning_applied':  'negative_to_nan' if n_neg_cleaned > 0 else 'none',
        'n_neg_cleaned':     n_neg_cleaned,
        'nan_preserved':     'True',
        'no_interpolation':  'True',
        'no_gap_filling':    'True',
        'no_imputation':     'True',
        'STAID':             staid,
        'period_start_utc':  period_start,
        'period_end_utc':    period_end,
    }
    ds_out['date'].attrs = {
        'timezone':    'UTC (naive datetime64; no tz offset stored)',
        'description': 'UTC hourly, proleptic_gregorian calendar',
    }

    enc = {
        'qobs_m3s': {'dtype': 'float32', '_FillValue': -9999.0},
        'date': {
            'dtype':    'float64',
            'units':    f'hours since {ref_time}',
            'calendar': 'proleptic_gregorian',
        },
    }

    tmp_path = out_nc.with_suffix('.nc.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    ds_out.to_netcdf(str(tmp_path), encoding=enc)
    if out_nc.exists():
        out_nc.unlink()
    tmp_path.rename(out_nc)

    return {
        'staid':            staid,
        'status':           'PASS',
        'n_neg_cleaned':    n_neg_cleaned,
        'n_nan_before':     n_nan_before,
        'n_nan_after':      n_nan_after,
        'n_valid_after':    n_valid_after,
        'n_hours':          n_hours,
        'period_start_utc': period_start,
        'period_end_utc':   period_end,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description='Flash-NH Stage 1 2J-B: build target-cleaned streamflow package.'
    )
    p.add_argument('--canonical-dir', required=True, type=pathlib.Path,
                   help='Directory containing *_hourly.nc canonical acquisition files.')
    p.add_argument('--policy', required=True, type=pathlib.Path,
                   help='Path to stage1_target_policy.yaml.')
    p.add_argument('--out-dir', required=True, type=pathlib.Path,
                   help='Output package directory.')
    p.add_argument('--status-csv', type=pathlib.Path, default=None,
                   help='Path to audit/target_status.csv. If omitted: permissive smoke mode.')
    p.add_argument('--max-basins', type=int, default=None,
                   help='Limit to first N basins (for smoke testing).')
    p.add_argument('--staids', type=str, default=None,
                   help='Comma-separated STAID list. Selects specific basins.')
    p.add_argument('--exclude-staids', type=str, default=None,
                   help='Comma-separated STAIDs to unconditionally exclude.')
    p.add_argument('--allow-review-required', type=str, default=None,
                   help='Comma-separated STAIDs to override review_required halt.')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing output NCs.')
    args = p.parse_args()

    t0 = _time.time()
    print('=' * 70)
    print('Flash-NH Stage 1 2J-B — Target Package Builder')
    print(f'Started: {CREATED_UTC}')
    print('=' * 70)

    # Load policy
    policy = _load_policy(args.policy)
    pname  = policy.get('policy_name', 'unknown')
    pver   = policy.get('policy_version', 'unknown')
    print(f'\n[policy] {pname} v{pver}')

    if not policy.get('package_build_status', {}).get('local_smoke_permitted', False):
        if args.max_basins is None and args.staids is None:
            print('\n!! ABORT: policy.package_build_status.local_smoke_permitted=false '
                  'and this is not a smoke run (no --max-basins or --staids).\n'
                  '   Use --max-basins N for a smoke run, or update the policy YAML.')
            return 1

    # Parse CLI overrides
    allow_review: set[str] = set()
    if args.allow_review_required:
        allow_review = {_norm_staid(s) for s in args.allow_review_required.split(',')}

    exclude_explicit: set[str] = set()
    if args.exclude_staids:
        exclude_explicit = {_norm_staid(s) for s in args.exclude_staids.split(',')}

    explicit_staids: list[str] | None = None
    if args.staids:
        explicit_staids = [_norm_staid(s) for s in args.staids.split(',')]

    # Discover canonical NCs
    canonical_dir = args.canonical_dir
    if not canonical_dir.exists():
        print(f'\n!! ERROR: --canonical-dir not found: {canonical_dir}')
        return 1

    all_ncs = _find_canonical_ncs(canonical_dir)
    print(f'\n[scan] Found {len(all_ncs)} canonical NC files in {canonical_dir}')
    if not all_ncs:
        print('!! ERROR: No *_hourly.nc files found.')
        return 1

    # Build STAID → nc_path map
    nc_map: dict[str, pathlib.Path] = {}
    for nc in all_ncs:
        staid = _norm_staid(nc.name.replace('_hourly.nc', ''))
        nc_map[staid] = nc

    # Select candidate basin list
    if explicit_staids is not None:
        candidates = [s for s in explicit_staids if s in nc_map]
        missing = [s for s in explicit_staids if s not in nc_map]
        if missing:
            print(f'  [warn] STAIDs not found in canonical-dir: {missing}')
    else:
        candidates = sorted(nc_map.keys())

    # Apply --max-basins
    if args.max_basins is not None and args.max_basins < len(candidates):
        candidates = candidates[:args.max_basins]
        print(f'  [smoke] Limited to first {args.max_basins} basins.')

    # Remove explicitly excluded
    if exclude_explicit:
        candidates = [s for s in candidates if s not in exclude_explicit]
        print(f'  [exclude] {len(exclude_explicit)} basins unconditionally excluded.')

    print(f'  Candidates after selection: {len(candidates)}')

    # Load status CSV and apply policy filter
    status_df = _load_status_df(args.status_csv)
    included, policy_excluded = _filter_basins_by_policy(candidates, status_df, policy)
    print(f'  After policy filter: {len(included)} included, {len(policy_excluded)} excluded.')

    # Check for special-review basins — must do before any processing
    halt_required = []
    for staid in included:
        decision = _check_special_review(staid, policy, allow_review)
        if decision == 'halt':
            halt_required.append(staid)

    if halt_required:
        print(f'\n!! BUILD HALTED by special-review policy for: {halt_required}')
        print('   Pass --allow-review-required ' + ','.join(halt_required) + ' to override.')
        print('   Pass --exclude-staids ' + ','.join(halt_required) + ' to exclude them.')
        return 1

    # Build per-basin special-review disposition map
    sr_disposition: dict[str, str] = {}
    final_included: list[str] = []
    sr_excluded: list[str] = []
    for staid in included:
        decision = _check_special_review(staid, policy, allow_review)
        if decision == 'exclude':
            sr_excluded.append(staid)
            sr_disposition[staid] = 'excluded_by_special_review'
        else:
            final_included.append(staid)
            if decision == 'include':
                sr_disposition[staid] = 'included_override'

    print(f'  After special-review: {len(final_included)} to process, '
          f'{len(sr_excluded)} excluded by review policy.')

    if not final_included:
        print('!! No basins to process after all filtering.')
        return 1

    # Create output directories
    out_dir = args.out_dir
    ts_dir  = out_dir / 'time_series'
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Process basins
    print(f'\n[build] Processing {len(final_included)} basins ...')
    results: list[dict] = []
    n_pass = 0
    n_fail = 0

    for i, staid in enumerate(final_included, 1):
        nc_path = nc_map[staid]
        res = process_one_basin(nc_path, staid, policy, ts_dir, args.force)
        results.append(res)
        if res['status'] == 'PASS':
            n_pass += 1
            tag = ''
            if res['n_neg_cleaned'] > 0:
                tag = f' [cleaned {res["n_neg_cleaned"]} neg→NaN]'
            print(f'  [{i:4d}/{len(final_included)}] {staid}  '
                  f'valid={res["n_valid_after"]}  nan={res["n_nan_after"]}{tag}')
        elif res['status'] == 'SKIP_EXISTS':
            n_pass += 1
            print(f'  [{i:4d}/{len(final_included)}] {staid}  SKIP (exists)')
        else:
            n_fail += 1
            print(f'  [{i:4d}/{len(final_included)}] {staid}  !! {res["status"]}')

    # Write cleaning report
    cleaning_csv = out_dir / 'cleaning_report.csv'
    pd.DataFrame(results).to_csv(cleaning_csv, index=False)
    print(f'\n[report] Written cleaning_report.csv ({len(results)} rows)')

    # Write checksums
    checksum_path = out_dir / 'checksums.sha256'
    nc_files = sorted(ts_dir.glob('*.nc'))
    checksums: dict[str, str] = {}
    with open(checksum_path, 'w', encoding='utf-8') as fh:
        for nc in nc_files:
            h = _sha256_file(nc)
            checksums[nc.name] = h
            fh.write(f'{h}  time_series/{nc.name}\n')
    print(f'[checksums] Written checksums.sha256 ({len(checksums)} files)')

    # Write manifest
    n_neg_total   = sum(r.get('n_neg_cleaned', 0) for r in results)
    n_nan_before  = sum(r.get('n_nan_before', 0) for r in results)
    n_nan_after   = sum(r.get('n_nan_after', 0) for r in results)
    n_valid_total = sum(r.get('n_valid_after', 0) for r in results)

    manifest = {
        'created_utc':         CREATED_UTC,
        'milestone':           'Flash-NH Stage 1 2J-B',
        'policy_name':         pname,
        'policy_version':      str(pver),
        'policy_file':         str(args.policy),
        'canonical_dir':       str(canonical_dir),
        'n_basins_processed':  n_pass,
        'n_basins_failed':     n_fail,
        'n_basins_policy_excluded': len(policy_excluded),
        'n_basins_sr_excluded': len(sr_excluded),
        'smoke_mode':          args.max_basins is not None or args.staids is not None,
        'status_csv_used':     args.status_csv is not None,
        'basins':              [r['staid'] for r in results if r['status'] == 'PASS'],
        'cleaning_summary': {
            'n_neg_cleaned':   n_neg_total,
            'n_nan_before':    n_nan_before,
            'n_nan_after':     n_nan_after,
            'n_valid_after':   n_valid_total,
        },
        'policy_excluded':     policy_excluded,
        'sr_excluded_staids':  sr_excluded,
        'nc_file_pattern':     'time_series/<STAID>.nc',
    }
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f'[manifest] Written manifest.json')

    # Write provenance
    provenance = {
        'script':              SCRIPT_NAME,
        'created_utc':         CREATED_UTC,
        'args': {
            'canonical_dir':   str(args.canonical_dir),
            'policy':          str(args.policy),
            'out_dir':         str(args.out_dir),
            'status_csv':      str(args.status_csv) if args.status_csv else None,
            'max_basins':      args.max_basins,
            'staids':          args.staids,
            'exclude_staids':  args.exclude_staids,
            'allow_review_required': args.allow_review_required,
            'force':           args.force,
        },
        'policy_sha256':       _sha256_file(args.policy),
        'n_input_nc_files':    len(all_ncs),
    }
    prov_path = out_dir / 'run_provenance.json'
    with open(prov_path, 'w', encoding='utf-8') as fh:
        json.dump(provenance, fh, indent=2, default=str)
    print(f'[provenance] Written run_provenance.json')

    elapsed = _time.time() - t0
    print('\n' + '=' * 70)
    overall = 'PASS' if n_fail == 0 else 'FAIL'
    print(f'Build {overall} in {elapsed:.1f}s')
    print(f'  Basins processed:  {n_pass}')
    print(f'  Basins failed:     {n_fail}')
    print(f'  Negative->NaN:     {n_neg_total} values across all basins')
    print(f'  NaN before/after:  {n_nan_before} / {n_nan_after}')
    print(f'  Valid hours total: {n_valid_total}')
    print(f'  Output:            {out_dir}')
    print('=' * 70)

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
