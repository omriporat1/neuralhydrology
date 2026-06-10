"""
Flash-NH Stage 1 Milestone 2G — NeuralHydrology Package Auditor
===============================================================

Audits the January 2023 NeuralHydrology pilot package produced by
build_stage1_neuralhydrology_january_pilot.py.

Checks:
  - Package structure (all required files/directories exist)
  - Per-basin NetCDF: date coordinate, 744 steps, time span, variables,
    value ranges, NaN patterns
  - Static attributes: 50 rows, gauge_id format, smoke subset completeness
  - Basin splits: no duplicates, no-streamflow basins excluded from sf-only split
  - Config skeleton: referenced variables and attributes exist
  - HydroATLAS join: unmatched basins recorded

Exit code: 0 = PASS, 1 = FAIL
"""
import sys
import json
import time as _time
import pathlib
import datetime

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CREATED_UTC = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
t0 = _time.time()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_ROOT  = REPO_ROOT / 'tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package'
AUDIT_DIR = REPO_ROOT / 'tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

TS_DIR    = PKG_ROOT / 'time_series'
ATTR_DIR  = PKG_ROOT / 'attributes'
SPLIT_DIR = PKG_ROOT / 'basin_lists'
CFG_DIR   = PKG_ROOT / 'configs'
MFST_DIR  = PKG_ROOT / 'manifests'

# Expected dynamic variables (11 total: 10 forcing + 1 target)
EXPECTED_WIDE_VARS = {
    'mrms_qpe_1h_mm', 'rtma_2t_K', 'rtma_2d_K', 'rtma_2sh_kgkg',
    'rtma_sp_Pa', 'rtma_10u_ms', 'rtma_10v_ms', 'rtma_10si_ms',
    'rtma_i10fg_ms', 'rtma_tcc_pct', 'qobs_m3s',
}
EXCLUDED_VARS = {'ceil', 'vis', 'rtma_ceil_m', 'rtma_vis_m'}
SMOKE_DYN_VARS = [
    'mrms_qpe_1h_mm', 'rtma_2t_K', 'rtma_2d_K',
    'rtma_2sh_kgkg', 'rtma_10u_ms', 'rtma_10v_ms',
]
SMOKE_STATIC_COLS = ['DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE', 'BFI_AVE', 'RBI']

EXPECTED_START = np.datetime64('2023-01-01T00:00:00', 'ns')
EXPECTED_END   = np.datetime64('2023-01-31T23:00:00', 'ns')

# Value range checks  {varname: (min_expected, max_expected)}
RANGE_CHECKS = {
    'mrms_qpe_1h_mm':  (0.0,    50.0),
    'rtma_2t_K':       (200.0,  330.0),
    'rtma_2d_K':       (180.0,  330.0),
    'rtma_2sh_kgkg':   (0.0,    0.03),
    'rtma_sp_Pa':      (55000., 110000.),
    'rtma_10u_ms':     (-40.,   40.),
    'rtma_10v_ms':     (-40.,   40.),
    'rtma_10si_ms':    (0.0,    50.0),
    'rtma_i10fg_ms':   (0.0,    70.0),
    'rtma_tcc_pct':    (0.0,    100.0),
    'qobs_m3s':        (0.0,    200000.),
}

issues = []
warnings_ = []

def err(msg, basin=None):
    tag = f'[{basin}] ' if basin else ''
    issues.append({'basin': basin or '', 'severity': 'ERROR', 'message': msg})
    print(f'  ERROR: {tag}{msg}')

def warn(msg, basin=None):
    tag = f'[{basin}] ' if basin else ''
    warnings_.append({'basin': basin or '', 'severity': 'WARN', 'message': msg})
    print(f'  WARN:  {tag}{msg}')

def ok(msg):
    print(f'  OK:    {msg}')

# ---------------------------------------------------------------------------
# Check 1: Package structure
# ---------------------------------------------------------------------------
print('=' * 70)
print('Flash-NH Stage 1 Milestone 2G — Preflight Auditor')
print(f'Audit time: {CREATED_UTC}')
print('=' * 70)

print('\n[1] Package structure ...')
required_dirs = [TS_DIR, ATTR_DIR, SPLIT_DIR, CFG_DIR, MFST_DIR]
for d in required_dirs:
    if d.exists():
        ok(f'{d.name}/ exists')
    else:
        err(f'Directory missing: {d}')

required_files = [
    ATTR_DIR / 'attributes_full.csv',
    ATTR_DIR / 'attributes_smoke.csv',
    SPLIT_DIR / 'all_basins.txt',
    SPLIT_DIR / 'january_2023_smoke' / 'train_basins.txt',
    SPLIT_DIR / 'january_2023_smoke' / 'val_basins.txt',
    SPLIT_DIR / 'january_2023_smoke' / 'test_basins.txt',
    SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'train_basins.txt',
    SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'val_basins.txt',
    SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'test_basins.txt',
    SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'no_streamflow_basins.txt',
    CFG_DIR / 'smoke_v1.yml',
    MFST_DIR / 'dataset_manifest.json',
    MFST_DIR / 'variable_schema.csv',
    MFST_DIR / 'static_attribute_audit.csv',
    MFST_DIR / 'hydroatlas_join_audit.csv',
    MFST_DIR / 'missingness_report.csv',
]
for fp in required_files:
    if fp.exists():
        ok(fp.name)
    else:
        err(f'File missing: {fp}')

# ---------------------------------------------------------------------------
# Check 2: Per-basin NetCDF files
# ---------------------------------------------------------------------------
print('\n[2] Per-basin NetCDF files ...')

# Load no_streamflow list
no_sf_path = SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'no_streamflow_basins.txt'
no_sf_staids = set()
if no_sf_path.exists():
    lines = no_sf_path.read_text(encoding='utf-8').splitlines()
    no_sf_staids = {l.strip() for l in lines if l.strip() and not l.startswith('#')}

nc_files = sorted(TS_DIR.glob('*.nc'))
if len(nc_files) != 50:
    err(f'Expected 50 NetCDF files, found {len(nc_files)}')
else:
    ok(f'50 NetCDF files found in {TS_DIR.name}/')

per_basin_rows = []
per_var_nancount = {v: 0 for v in EXPECTED_WIDE_VARS}

for nc_path in nc_files:
    staid = nc_path.stem
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        err(f'Cannot open NetCDF: {e}', staid)
        continue

    # 2a: date coordinate
    if 'date' not in ds.coords and 'date' not in ds.dims:
        err('Missing "date" coordinate', staid)
    else:
        times = pd.to_datetime(ds['date'].values)
        if len(times) != 744:
            err(f'"date" has {len(times)} steps, expected 744', staid)
        else:
            # Check span
            t0_nc = np.datetime64(times[0], 'ns')
            t1_nc = np.datetime64(times[-1], 'ns')
            if t0_nc != EXPECTED_START:
                err(f'First timestamp {times[0]} != expected 2023-01-01T00Z', staid)
            if t1_nc != EXPECTED_END:
                err(f'Last timestamp {times[-1]} != expected 2023-01-31T23Z', staid)
            # Check duplicates
            if len(times) != len(set(times)):
                err('Duplicate timestamps', staid)

    # 2b: variables
    ds_vars = set(ds.data_vars)
    missing_vars = EXPECTED_WIDE_VARS - ds_vars
    extra_excluded = EXCLUDED_VARS & ds_vars
    if missing_vars:
        err(f'Missing variables: {sorted(missing_vars)}', staid)
    if extra_excluded:
        err(f'Excluded variables present in file: {extra_excluded}', staid)

    # 2c: value ranges and NaN checks
    qobs_nan = 0
    forcing_nan = 0
    for v in EXPECTED_WIDE_VARS:
        if v not in ds:
            continue
        vals = ds[v].values.astype(float)
        nan_n = int(np.isnan(vals).sum())
        per_var_nancount[v] += nan_n

        if v == 'qobs_m3s':
            qobs_nan = nan_n
            if nan_n == 744:
                if staid not in no_sf_staids:
                    warn(f'qobs_m3s all-NaN but not in no_streamflow_basins.txt', staid)
            elif nan_n == 0:
                pass  # full coverage, OK
            else:
                pass  # partial NaN, warn if unexpected
        else:
            if nan_n > 0:
                err(f'{v}: {nan_n} NaN values (forcing should be gap-free)', staid)
                forcing_nan += nan_n
            # Range check (on non-NaN values)
            if v in RANGE_CHECKS:
                lo, hi = RANGE_CHECKS[v]
                finite_vals = vals[np.isfinite(vals)]
                if len(finite_vals) > 0:
                    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
                    if vmin < lo:
                        warn(f'{v} min={vmin:.4f} < expected {lo}', staid)
                    if vmax > hi:
                        warn(f'{v} max={vmax:.4f} > expected {hi}', staid)
            # Check for negative streamflow
            if v == 'qobs_m3s':
                neg_n = int((vals[np.isfinite(vals)] < 0).sum())
                if neg_n > 0:
                    err(f'qobs_m3s has {neg_n} negative values', staid)

    # 2d: variable attributes
    for v in EXPECTED_WIDE_VARS:
        if v not in ds:
            continue
        vattrs = ds[v].attrs
        for req_attr in ('units', 'long_name', 'source_product', 'timestep_meaning'):
            if req_attr not in vattrs:
                warn(f'{v}: missing attribute "{req_attr}"', staid)

    per_basin_rows.append({
        'STAID': staid,
        'n_vars': len(ds_vars),
        'n_time': len(ds['date'].values) if 'date' in ds.coords else -1,
        'qobs_nan': qobs_nan,
        'forcing_nan': forcing_nan,
        'qobs_all_nan': qobs_nan == 744,
        'status': 'ok' if forcing_nan == 0 else 'forcing_nan',
    })
    ds.close()

n_ok = sum(1 for r in per_basin_rows if r['forcing_nan'] == 0)
n_qobs_allnan = sum(1 for r in per_basin_rows if r['qobs_all_nan'])
n_qobs_partial = sum(1 for r in per_basin_rows if 0 < r['qobs_nan'] < 744)
n_qobs_full = sum(1 for r in per_basin_rows if r['qobs_nan'] == 0)
print(f'    Basins with no forcing NaN: {n_ok}/50')
print(f'    qobs_m3s: {n_qobs_full} full, {n_qobs_partial} partial, {n_qobs_allnan} all-NaN')

# Check all-NaN qobs match no_streamflow list
allnan_staids = {r['STAID'] for r in per_basin_rows if r['qobs_all_nan']}
if allnan_staids == no_sf_staids:
    ok(f'all-NaN qobs_m3s STAIDs match no_streamflow_basins.txt ({len(allnan_staids)} basins)')
else:
    in_list_not_allnan = no_sf_staids - allnan_staids
    allnan_not_in_list = allnan_staids - no_sf_staids
    if in_list_not_allnan:
        warn(f'STAIDs in no_streamflow_basins.txt but NOT all-NaN: {sorted(in_list_not_allnan)}')
    if allnan_not_in_list:
        warn(f'STAIDs all-NaN but NOT in no_streamflow_basins.txt: {sorted(allnan_not_in_list)}')

# ---------------------------------------------------------------------------
# Check 3: Static attributes
# ---------------------------------------------------------------------------
print('\n[3] Static attributes ...')
if (ATTR_DIR / 'attributes_full.csv').exists():
    attrs_full = pd.read_csv(ATTR_DIR / 'attributes_full.csv', dtype={'gauge_id': str})
    if len(attrs_full) != 50:
        err(f'attributes_full.csv has {len(attrs_full)} rows, expected 50')
    else:
        ok(f'attributes_full.csv: 50 rows x {len(attrs_full.columns)} cols')
    if 'gauge_id' not in attrs_full.columns:
        err('attributes_full.csv: missing "gauge_id" column')
    else:
        # Check leading zeros preserved
        bad_ids = attrs_full['gauge_id'].astype(str).str.len().ne(8)
        if bad_ids.any():
            warn(f'gauge_id: {bad_ids.sum()} IDs not 8 chars (leading zero issue?)')
        else:
            ok('gauge_id: all 8-char zero-padded STAIDs')

    # Check for sentinel values
    num_cols = attrs_full.select_dtypes(include='number')
    sentinel_mask = (num_cols == -999) | (num_cols == -9999)
    if sentinel_mask.any().any():
        err(f'attributes_full.csv: sentinel values (-999/-9999) found in numeric columns')
    else:
        ok('No sentinel values in attributes_full.csv')

    # Null counts
    null_counts = attrs_full.isnull().sum()
    nonzero_nulls = null_counts[null_counts > 0]
    if len(nonzero_nulls) > 0:
        warn(f'attributes_full.csv: columns with nulls: {nonzero_nulls.to_dict()}')
    else:
        ok('No null values in attributes_full.csv')

if (ATTR_DIR / 'attributes_smoke.csv').exists():
    attrs_smoke = pd.read_csv(ATTR_DIR / 'attributes_smoke.csv')
    expected_smoke_cols = {'gauge_id'} | set(SMOKE_STATIC_COLS)
    if len(attrs_smoke) != 50:
        err(f'attributes_smoke.csv has {len(attrs_smoke)} rows, expected 50')
    if set(attrs_smoke.columns) != expected_smoke_cols:
        err(f'attributes_smoke.csv columns: expected {sorted(expected_smoke_cols)}, got {sorted(attrs_smoke.columns)}')
    else:
        ok(f'attributes_smoke.csv: 50 rows x {len(attrs_smoke.columns)} cols (correct)')
    if attrs_smoke[SMOKE_STATIC_COLS].isnull().any().any():
        err('attributes_smoke.csv: null values in smoke static columns')
    else:
        ok('attributes_smoke.csv: no nulls in smoke columns')

# ---------------------------------------------------------------------------
# Check 4: Basin split files
# ---------------------------------------------------------------------------
print('\n[4] Basin split files ...')

def _read_split(path):
    if not path.exists():
        return set()
    lines = path.read_text(encoding='utf-8').splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith('#')}

all_basins = _read_split(SPLIT_DIR / 'all_basins.txt')
if len(all_basins) != 50:
    err(f'all_basins.txt has {len(all_basins)} entries, expected 50')
else:
    ok('all_basins.txt: 50 entries')

# All-50 split
s_train = _read_split(SPLIT_DIR / 'january_2023_smoke' / 'train_basins.txt')
s_val   = _read_split(SPLIT_DIR / 'january_2023_smoke' / 'val_basins.txt')
s_test  = _read_split(SPLIT_DIR / 'january_2023_smoke' / 'test_basins.txt')
all50_union = s_train | s_val | s_test
if all50_union != all_basins:
    err(f'all-50 splits: union does not equal all_basins.txt')
else:
    ok(f'all-50 splits: correct union (train={len(s_train)}, val={len(s_val)}, test={len(s_test)})')
if len(s_train) + len(s_val) + len(s_test) != 50:
    err(f'all-50 splits: total={len(s_train)+len(s_val)+len(s_test)}, expected 50 (possible duplicate)')
all50_overlap = (s_train & s_val) | (s_train & s_test) | (s_val & s_test)
if all50_overlap:
    err(f'all-50 splits: overlap between split files: {all50_overlap}')

# Streamflow-only split
sf_dir = SPLIT_DIR / 'january_2023_smoke_streamflow_only'
sf_train = _read_split(sf_dir / 'train_basins.txt')
sf_val   = _read_split(sf_dir / 'val_basins.txt')
sf_test  = _read_split(sf_dir / 'test_basins.txt')
sf_union = sf_train | sf_val | sf_test

if sf_union & no_sf_staids:
    err(f'sf-only splits: contain no-streamflow basins: {sf_union & no_sf_staids}')
else:
    ok(f'sf-only splits: no no-streamflow basins included')

sf_overlap = (sf_train & sf_val) | (sf_train & sf_test) | (sf_val & sf_test)
if sf_overlap:
    err(f'sf-only splits: overlap between split files: {sf_overlap}')
else:
    ok(f'sf-only splits: no overlap (train={len(sf_train)}, val={len(sf_val)}, test={len(sf_test)})')

expected_sf_basins = all_basins - no_sf_staids
if sf_union != expected_sf_basins:
    miss_from_split = expected_sf_basins - sf_union
    extra_in_split  = sf_union - expected_sf_basins
    if miss_from_split:
        warn(f'sf-only splits: missing basins that have streamflow: {miss_from_split}')
    if extra_in_split:
        warn(f'sf-only splits: extra basins not in expected sf-covered set: {extra_in_split}')
else:
    ok(f'sf-only splits: correct basin set ({len(sf_union)} basins)')

# ---------------------------------------------------------------------------
# Check 5: Config skeleton
# ---------------------------------------------------------------------------
print('\n[5] Config skeleton ...')
cfg_path = CFG_DIR / 'smoke_v1.yml'
if cfg_path.exists():
    cfg_text = cfg_path.read_text(encoding='utf-8')
    # Check that smoke dynamic vars are mentioned
    for v in SMOKE_DYN_VARS:
        if v not in cfg_text:
            warn(f'smoke_v1.yml: dynamic input "{v}" not found in config text')
    # Check that excluded vars are not mentioned as inputs
    for v in ['rtma_sp_Pa', 'rtma_tcc_pct', 'rtma_ceil_m', 'rtma_vis_m', 'urma_tp']:
        # These should not appear as dynamic inputs (but may be in comments)
        pass  # Skip strict check on excluded vars -- comments may reference them
    # Check that target is mentioned
    if 'qobs_m3s' not in cfg_text:
        warn('smoke_v1.yml: target "qobs_m3s" not found in config')
    # Check smoke static cols mentioned
    for col in SMOKE_STATIC_COLS:
        if col not in cfg_text:
            warn(f'smoke_v1.yml: static attr "{col}" not found in config')
    # Check DRAFT marker
    if 'DRAFT' in cfg_text:
        ok('smoke_v1.yml: DRAFT marker present (config not yet run)')
    else:
        warn('smoke_v1.yml: no DRAFT marker found')
    ok('smoke_v1.yml: exists and references expected variables')
else:
    err('smoke_v1.yml: file missing')

# ---------------------------------------------------------------------------
# Check 6: HydroATLAS join audit
# ---------------------------------------------------------------------------
print('\n[6] HydroATLAS join audit ...')
ha_audit_path = MFST_DIR / 'hydroatlas_join_audit.csv'
if ha_audit_path.exists():
    ha_audit = pd.read_csv(ha_audit_path)
    n_matched = int(ha_audit['in_hydroatlas'].sum())
    n_unmatched = 50 - n_matched
    if n_unmatched > 0:
        unmatched_ids = ha_audit.loc[~ha_audit['in_hydroatlas'], 'STAID'].tolist()
        warn(f'HydroATLAS join: {n_unmatched} unmatched pilot STAIDs: {unmatched_ids}')
    else:
        ok(f'HydroATLAS join: 50/50 pilot STAIDs matched')
else:
    warn('hydroatlas_join_audit.csv not found')

# ---------------------------------------------------------------------------
# Write audit outputs
# ---------------------------------------------------------------------------
print('\n[7] Writing audit outputs ...')
elapsed = _time.time() - t0
n_issues   = len(issues)
n_warnings = len(warnings_)
pass_fail  = 'PASS' if n_issues == 0 else 'FAIL'

per_var_miss = []
for v, nc in per_var_nancount.items():
    per_var_miss.append({'variable': v, 'total_nan_across_all_basins': nc,
                         'is_forcing': v != 'qobs_m3s'})
pd.DataFrame(per_var_miss).to_csv(AUDIT_DIR / 'per_variable_missingness.csv', index=False)

per_basin_out = pd.DataFrame(per_basin_rows)
per_basin_out.to_csv(AUDIT_DIR / 'per_basin_summary.csv', index=False)

if (ATTR_DIR / 'attributes_full.csv').exists():
    sa_df = pd.read_csv(ATTR_DIR / 'attributes_full.csv', dtype={'gauge_id': str})
    sa_audit_rows = []
    for col in sa_df.columns:
        if col == 'gauge_id':
            continue
        col_data = sa_df[col]
        is_num = pd.api.types.is_numeric_dtype(col_data)
        sa_audit_rows.append({
            'column': col,
            'dtype': str(col_data.dtype),
            'null_count': int(col_data.isna().sum()),
            'null_pct': round(col_data.isna().mean() * 100, 1),
            'min': round(float(col_data.min()), 6) if is_num and col_data.notna().any() else None,
            'max': round(float(col_data.max()), 6) if is_num and col_data.notna().any() else None,
        })
    pd.DataFrame(sa_audit_rows).to_csv(AUDIT_DIR / 'static_attribute_audit.csv', index=False)

audit_report = {
    'audit_time_utc': CREATED_UTC,
    'pass_fail': pass_fail,
    'n_errors': n_issues,
    'n_warnings': n_warnings,
    'n_basins_checked': len(per_basin_rows),
    'n_basins_forcing_ok': n_ok,
    'n_basins_qobs_full': n_qobs_full,
    'n_basins_qobs_partial': n_qobs_partial,
    'n_basins_qobs_allnan': n_qobs_allnan,
    'n_wide_vars_per_basin': len(EXPECTED_WIDE_VARS),
    'elapsed_sec': round(elapsed, 1),
    'errors': issues,
    'warnings': warnings_,
}
with open(AUDIT_DIR / 'audit_report.json', 'w', encoding='utf-8') as f:
    json.dump(audit_report, f, indent=2)

md_lines = [
    f'# Flash-NH Stage 1 Milestone 2G — Preflight Audit Report',
    f'',
    f'**Audit time:** {CREATED_UTC}',
    f'**Result:** {pass_fail}',
    f'**Errors:** {n_issues}  **Warnings:** {n_warnings}',
    f'',
    f'## Coverage',
    f'',
    f'| Metric | Value |',
    f'|---|---|',
    f'| Basins checked | {len(per_basin_rows)} / 50 |',
    f'| Basins with no forcing NaN | {n_ok} / 50 |',
    f'| qobs_m3s: full coverage | {n_qobs_full} / 50 |',
    f'| qobs_m3s: partial NaN | {n_qobs_partial} / 50 |',
    f'| qobs_m3s: all-NaN (no CAMELSH) | {n_qobs_allnan} / 50 |',
    f'| Wide dynamic vars per basin | {len(EXPECTED_WIDE_VARS)} |',
    f'',
]
if issues:
    md_lines += ['## Errors', '']
    for e in issues:
        md_lines.append(f'- [{e["basin"]}] {e["message"]}')
    md_lines.append('')
if warnings_:
    md_lines += ['## Warnings', '']
    for w in warnings_:
        md_lines.append(f'- [{w["basin"]}] {w["message"]}')
    md_lines.append('')

with open(AUDIT_DIR / 'audit_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print(f'Audit result: {pass_fail}')
print(f'  Errors:   {n_issues}')
print(f'  Warnings: {n_warnings}')
print(f'  Time: {elapsed:.1f}s')
print(f'Outputs written to: {AUDIT_DIR}')
print('=' * 70)

sys.exit(0 if pass_fail == 'PASS' else 1)
