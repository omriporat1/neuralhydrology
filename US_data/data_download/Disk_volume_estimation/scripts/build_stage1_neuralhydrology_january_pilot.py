"""
Flash-NH Stage 1 Milestone 2G — NeuralHydrology NetCDF Builder
===============================================================

Builds the January 2023 NeuralHydrology GenericDataset-compatible pilot package
for 50 basins:
  - One NetCDF per basin  (time_series/<STAID>.nc)
  - Static attributes CSV (attributes_full.csv, attributes_smoke.csv)
  - Basin split files
  - Smoke config skeleton (configs/smoke_v1.yml)
  - Package manifests

Design constraints:
  - Do NOT train any model.
  - Do NOT add URMA QPE to model inputs.
  - MRMS variable 'unknown' renamed to 'mrms_qpe_1h_mm'.
  - RTMA ceil and vis excluded from output.
  - Missing CAMELSH targets written as all-NaN (22/50 basins).
  - NaN preserved; no imputation.
"""
import sys
import json
import time as _time
import pathlib
import warnings
import datetime

import numpy as np
import pandas as pd
import xarray as xr

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

CREATED_UTC = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent
PARQUET    = REPO_ROOT / 'tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet'
MANIFEST   = REPO_ROOT / 'tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv'
GAGES2_PQ  = REPO_ROOT / 'reports/flashnh_basin_screening_v001/all_basins_merged.parquet'
CAMELSH_DIR = pathlib.Path('C:/PhD/Python/neuralhydrology/US_data/data_download/CAMELSH_resolution_test/data/raw/camelsh')
HYDROATLAS_CSV = pathlib.Path('C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_hydroATLAS.csv')

PKG_ROOT   = REPO_ROOT / 'tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package'
AUDIT_DIR  = REPO_ROOT / 'tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/audit'

TS_DIR     = PKG_ROOT / 'time_series'
ATTR_DIR   = PKG_ROOT / 'attributes'
SPLIT_DIR  = PKG_ROOT / 'basin_lists'
CFG_DIR    = PKG_ROOT / 'configs'
MFST_DIR   = PKG_ROOT / 'manifests'

for d in [TS_DIR, ATTR_DIR, SPLIT_DIR, CFG_DIR, MFST_DIR, AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Variable rename map  (product, parquet_var) -> (nc_name, units, long_name, timestep)
# ---------------------------------------------------------------------------
VAR_RENAME = {
    ('mrms_qpe_1h_pass1',     'unknown'): ('mrms_qpe_1h_mm',  'mm',        'MRMS QPE 1-hour accumulation (Pass1)',   'accumulation ending at valid_time_utc'),
    ('rtma_conus_aws_2p5km',  '2t'):      ('rtma_2t_K',        'K',         '2-metre temperature',                   'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  '2d'):      ('rtma_2d_K',        'K',         '2-metre dewpoint temperature',          'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  '2sh'):     ('rtma_2sh_kgkg',    'kg kg**-1', '2-metre specific humidity',             'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  'sp'):      ('rtma_sp_Pa',       'Pa',        'Surface pressure',                      'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  '10u'):     ('rtma_10u_ms',      'm s**-1',   '10-metre U wind component',             'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  '10v'):     ('rtma_10v_ms',      'm s**-1',   '10-metre V wind component',             'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  '10si'):    ('rtma_10si_ms',     'm s**-1',   '10-metre wind speed',                   'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  'i10fg'):   ('rtma_i10fg_ms',    'm s**-1',   '10-metre wind gust speed',              'instantaneous analysis valid time'),
    ('rtma_conus_aws_2p5km',  'tcc'):     ('rtma_tcc_pct',     '%',         'Total cloud cover',                     'instantaneous analysis valid time'),
}
EXCLUDE_PAIRS = {
    ('rtma_conus_aws_2p5km', 'ceil'),
    ('rtma_conus_aws_2p5km', 'vis'),
}
WIDE_DYN_VARS = [v[0] for v in VAR_RENAME.values()]
SMOKE_DYN_VARS = [
    'mrms_qpe_1h_mm', 'rtma_2t_K', 'rtma_2d_K',
    'rtma_2sh_kgkg', 'rtma_10u_ms', 'rtma_10v_ms',
]
SMOKE_STATIC_COLS = ['DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE', 'BFI_AVE', 'RBI']

# Manifest columns to keep as model attributes
MANIFEST_ATTR_KEEP = [
    'BFI_AVE', 'CANALS_PCT', 'DRAIN_SQKM', 'HYDRO_DISTURB_INDX',
    'LAT_GAGE', 'LNG_GAGE', 'RBI', 'WATERNLCD06',
    'lka_pc_use', 'dor_pc_pva',
    'max_abs_hourly_jump_over_Q50', 'max_hourly_rise_per_km2',
    'q95_q50_ratio', 'zero_flow_fraction',
]
# Extra GAGES-II columns not in manifest
GAGES2_EXTRA = [
    'STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY',
    'ARTIFPATH_PCT', 'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT',
    'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT', 'RUNAVE7100',
    'WB5100_JAN_MM','WB5100_FEB_MM','WB5100_MAR_MM','WB5100_APR_MM',
    'WB5100_MAY_MM','WB5100_JUN_MM','WB5100_JUL_MM','WB5100_AUG_MM',
    'WB5100_SEP_MM','WB5100_OCT_MM','WB5100_NOV_MM','WB5100_DEC_MM',
    'WB5100_ANN_MM',
    'PCT_1ST_ORDER','PCT_2ND_ORDER','PCT_3RD_ORDER','PCT_4TH_ORDER',
    'PCT_5TH_ORDER','PCT_NO_ORDER',
]
# HydroATLAS columns to exclude from static table
HYDROATLAS_EXCLUDE = {'STAID', 'area_fraction_used_for_aggregation'}

def _norm_staid(s):
    """Normalize a STAID value to 8-char zero-padded string."""
    try:
        return f'{int(float(str(s).strip())):08d}'
    except (ValueError, TypeError):
        return str(s).strip().zfill(8)

# ---------------------------------------------------------------------------
# Section 1: Load and validate parquet
# ---------------------------------------------------------------------------
print('=' * 70)
print('Flash-NH Stage 1 Milestone 2G — NetCDF Builder')
print(f'Started: {CREATED_UTC}')
print('=' * 70)
t0 = _time.time()

print('\n[1] Loading combined extraction parquet ...')
df_raw = pd.read_parquet(PARQUET)
df_raw['STAID'] = df_raw['STAID'].apply(_norm_staid)
print(f'    Shape: {df_raw.shape}')
print(f'    STAIDs: {df_raw["STAID"].nunique()}')
print(f'    Time steps: {df_raw["valid_time_utc"].nunique()}')
print(f'    Products: {sorted(df_raw["product"].unique())}')
print(f'    NaN in weighted_mean: {df_raw["weighted_mean"].isna().sum()}')

assert df_raw['STAID'].nunique() == 50, 'Expected 50 basins'
assert df_raw['valid_time_utc'].nunique() == 744, 'Expected 744 time steps'
assert df_raw['weighted_mean'].isna().sum() == 0, 'Unexpected NaN in forcing'
print('    Assertions: PASS (50 basins, 744 steps, 0 NaN)')

# Build a composite varname column for pivot
def _get_nc_name(row):
    key = (row['product'], row['variable'])
    if key in EXCLUDE_PAIRS:
        return None
    return VAR_RENAME.get(key, (None,))[0]

df_raw['nc_varname'] = df_raw.apply(_get_nc_name, axis=1)
n_exclude = df_raw['nc_varname'].isna().sum()
df_use = df_raw.dropna(subset=['nc_varname']).copy()
print(f'    Excluded (ceil/vis): {n_exclude} rows; using {len(df_use)} rows')

# Build reference time index (UTC, tz-naive for xarray)
all_times_utc = pd.to_datetime(df_use['valid_time_utc'].unique())
if all_times_utc.tz is not None:
    ref_index = all_times_utc.tz_convert('UTC').tz_localize(None).sort_values()
else:
    ref_index = all_times_utc.sort_values()
assert len(ref_index) == 744
print(f'    Reference time: {ref_index[0]} to {ref_index[-1]}')

# ---------------------------------------------------------------------------
# Section 2: Per-basin NetCDF files
# ---------------------------------------------------------------------------
print('\n[2] Building per-basin NetCDF files ...')
pilot_staids = sorted(df_use['STAID'].unique())
no_streamflow_staids = []
nc_written = 0
per_basin_rows = []

for staid in pilot_staids:
    sub = df_use[df_use['STAID'] == staid].copy()

    # Normalize time for this basin
    if pd.api.types.is_extension_array_dtype(sub['valid_time_utc']) or \
       hasattr(sub['valid_time_utc'].dtype, 'tz'):
        times = pd.to_datetime(sub['valid_time_utc']).dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        times = pd.to_datetime(sub['valid_time_utc'])
    sub = sub.copy()
    sub['_t'] = times

    # Pivot: index=time, columns=nc_varname, values=weighted_mean
    wide = sub.pivot(index='_t', columns='nc_varname', values='weighted_mean')
    wide = wide.reindex(ref_index)   # align to reference time index
    wide.index.name = 'date'

    assert set(wide.columns) == set(WIDE_DYN_VARS), \
        f'{staid}: expected {set(WIDE_DYN_VARS)}, got {set(wide.columns)}'
    assert wide.isnull().sum().sum() == 0, f'{staid}: unexpected NaN in forcing'

    # Load CAMELSH streamflow
    camelsh_path = CAMELSH_DIR / f'{staid}_hourly.nc'
    nan_count = None
    if camelsh_path.exists():
        with xr.open_dataset(camelsh_path) as ds_c:
            if 'streamflow' not in ds_c:
                qobs = np.full(744, np.nan, dtype=np.float32)
                has_camelsh = False
            else:
                sf_raw = ds_c['streamflow']
                t_raw = pd.to_datetime(sf_raw.coords[sf_raw.dims[0]].values)
                if t_raw.tz is not None:
                    t_raw = t_raw.tz_convert('UTC').tz_localize(None)
                sf_series = pd.Series(sf_raw.values.astype(float), index=t_raw)
                sf_reindexed = sf_series.reindex(ref_index)
                qobs = sf_reindexed.values.astype(np.float32)
                has_camelsh = True
                nan_count = int(np.isnan(qobs).sum())
    else:
        qobs = np.full(744, np.nan, dtype=np.float32)
        has_camelsh = False
        no_streamflow_staids.append(staid)

    # Build xarray Dataset
    data_vars = {}
    for nc_name, (_, units, long_name, ts_meaning) in VAR_RENAME.items():
        col = VAR_RENAME[nc_name][0] if isinstance(nc_name, tuple) else nc_name
        # nc_name here is the key tuple; col is the mapped name
    # Rebuild cleanly
    data_vars = {}
    for key, (nc_col, units, long_name, ts_meaning) in VAR_RENAME.items():
        arr = wide[nc_col].values.astype(np.float32)
        data_vars[nc_col] = xr.DataArray(
            arr, dims=['date'],
            attrs={
                'units': units,
                'long_name': long_name,
                'source_product': key[0],
                'source_variable': key[1],
                'timestep_meaning': ts_meaning,
            }
        )
    data_vars['qobs_m3s'] = xr.DataArray(
        qobs, dims=['date'],
        attrs={
            'units': 'm3 s**-1',
            'long_name': 'Observed streamflow (CAMELSH hourly)',
            'source_product': 'CAMELSH hourly NetCDF',
            'source_variable': 'streamflow',
            'timestep_meaning': 'instantaneous hourly discharge at gauge',
        }
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'date': ref_index.values},
        attrs={
            'STAID': staid,
            'gauge_id': staid,
            'created_utc': CREATED_UTC,
            'milestone': 'Flash-NH Stage 1 Milestone 2G',
            'time_zone': 'UTC',
            'camelsh_available': str(has_camelsh),
            'source_parquet': str(PARQUET.name),
            'description': (
                'Flash-NH Stage 1 January 2023 pilot. NeuralHydrology GenericDataset format. '
                'Wide format: 10 dynamic forcing variables + qobs_m3s target. '
                'MRMS variable renamed from unknown to mrms_qpe_1h_mm. '
                'Missing CAMELSH basins have all-NaN qobs_m3s.'
            ),
        }
    )

    # Encoding
    enc = {v: {'dtype': 'float32', '_FillValue': -9999.0} for v in ds.data_vars}
    enc['date'] = {'dtype': 'float64', 'units': 'hours since 2023-01-01 00:00:00', 'calendar': 'proleptic_gregorian'}

    out_nc = TS_DIR / f'{staid}.nc'
    ds.to_netcdf(out_nc, encoding=enc)
    nc_written += 1

    basin_nan_q = int(np.isnan(qobs).sum())
    per_basin_rows.append({
        'STAID': staid,
        'nc_file': out_nc.name,
        'camelsh_available': has_camelsh,
        'qobs_nan_hours': basin_nan_q,
        'qobs_all_nan': basin_nan_q == 744,
        'forcing_nan_count': 0,
    })
    if nc_written % 10 == 0:
        print(f'    Written {nc_written}/50 ...')

print(f'    Written {nc_written}/50 basin NetCDF files')
print(f'    No-streamflow basins: {len(no_streamflow_staids)}')

# Write no_streamflow_basins.txt
no_sf_path = SPLIT_DIR / 'january_2023_smoke_streamflow_only' / 'no_streamflow_basins.txt'
no_sf_path.parent.mkdir(parents=True, exist_ok=True)
with open(no_sf_path, 'w') as f:
    f.write('# Flash-NH Stage 1 Milestone 2G\n')
    f.write('# Basins without local CAMELSH hourly NetCDF file.\n')
    f.write('# qobs_m3s written as all-NaN in the wide NetCDF package.\n')
    f.write('# Excluded from streamflow-only smoke split files.\n')
    f.write('# See streamflow_recovery_plan.md for Milestone 2H plan.\n')
    for s in sorted(no_streamflow_staids):
        f.write(s + '\n')
print(f'    Written no_streamflow_basins.txt ({len(no_streamflow_staids)} STAIDs)')

# ---------------------------------------------------------------------------
# Section 3: Static attributes — manifest base
# ---------------------------------------------------------------------------
print('\n[3] Building static attributes ...')
manifest = pd.read_csv(MANIFEST)
manifest['STAID'] = manifest['STAID'].apply(_norm_staid)
manifest = manifest.set_index('STAID')

keep_from_manifest = [c for c in MANIFEST_ATTR_KEEP if c in manifest.columns]
attrs_base = manifest[keep_from_manifest].copy()
print(f'    Manifest attrs: {len(keep_from_manifest)} columns from {len(attrs_base)} basins')

# Section 3a: Merge GAGES-II extras
gii = pd.read_parquet(GAGES2_PQ)
gii['STAID'] = gii['STAID'].apply(_norm_staid)
gii = gii.set_index('STAID')
extra_in_gii = [c for c in GAGES2_EXTRA if c in gii.columns]
attrs_base = attrs_base.join(gii[extra_in_gii], how='left')
print(f'    GAGES-II extra attrs added: {len(extra_in_gii)} columns')

# Section 3b: HydroATLAS join audit
print('\n[3b] HydroATLAS join audit ...')
ha = pd.read_csv(HYDROATLAS_CSV)
ha['STAID_norm'] = ha['STAID'].apply(_norm_staid)
ha = ha.set_index('STAID_norm')

pilot_set = set(pilot_staids)
ha_ids = set(ha.index)
ha_matched = sorted(pilot_set & ha_ids)
ha_unmatched = sorted(pilot_set - ha_ids)
print(f'    HydroATLAS shape: {ha.shape}')
print(f'    Pilot matched: {len(ha_matched)}/50')
print(f'    Pilot unmatched: {ha_unmatched}')

# Join HydroATLAS (exclude STAID col and area_fraction col, deduplicate with existing)
ha_cols_raw = [c for c in ha.columns if c not in HYDROATLAS_EXCLUDE]
# Drop columns already in attrs_base (lka_pc_use, dor_pc_pva already in manifest)
ha_new_cols = [c for c in ha_cols_raw if c not in attrs_base.columns]
ha_sub = ha[ha_new_cols]
attrs_base = attrs_base.join(ha_sub, how='left')
# Replace HydroATLAS nodata sentinels (-999, -9999) with NaN
ha_num_cols = [c for c in ha_new_cols if c in attrs_base.columns and
               pd.api.types.is_numeric_dtype(attrs_base[c])]
attrs_base[ha_num_cols] = attrs_base[ha_num_cols].replace(-999, np.nan).replace(-9999, np.nan)
print(f'    HydroATLAS new columns added: {len(ha_new_cols)}')
print(f'    attributes_full total columns: {len(attrs_base.columns)}')

# Build hydroatlas_join_audit.csv
ha_audit_rows = []
for s in pilot_staids:
    ha_audit_rows.append({
        'STAID': s,
        'in_hydroatlas': s in ha_ids,
        'ha_null_count': int(attrs_base.loc[s, ha_new_cols].isna().sum()) if s in ha_ids else len(ha_new_cols),
    })
ha_audit_df = pd.DataFrame(ha_audit_rows)

# Write no_hydroatlas_basins.txt if needed
if ha_unmatched:
    with open(ATTR_DIR / 'no_hydroatlas_basins.txt', 'w') as f:
        f.write('# Basins not found in HydroATLAS table\n')
        for s in ha_unmatched:
            f.write(s + '\n')
    print(f'    Written no_hydroatlas_basins.txt ({len(ha_unmatched)} STAIDs)')
else:
    print('    HydroATLAS join is clean (50/50). No no_hydroatlas_basins.txt needed.')

# Section 3c: Write attributes_full.csv
# Reset index so gauge_id is the first column
attrs_full = attrs_base.copy()
attrs_full.index.name = 'gauge_id'
attrs_full = attrs_full.reset_index()
# Ensure leading zeros on gauge_id
attrs_full['gauge_id'] = attrs_full['gauge_id'].apply(_norm_staid)
attrs_full_path = ATTR_DIR / 'attributes_full.csv'
attrs_full.to_csv(attrs_full_path, index=False)
print(f'\n    Written attributes_full.csv: {attrs_full.shape[0]} rows x {attrs_full.shape[1]} cols')

# Section 3d: Write attributes_smoke.csv
smoke_cols = ['gauge_id'] + SMOKE_STATIC_COLS
attrs_smoke = attrs_full[smoke_cols].copy()
attrs_smoke_path = ATTR_DIR / 'attributes_smoke.csv'
attrs_smoke.to_csv(attrs_smoke_path, index=False)
print(f'    Written attributes_smoke.csv: {attrs_smoke.shape[0]} rows x {attrs_smoke.shape[1]} cols')

# Section 3e: Static attribute audit table
static_audit_rows = []
for col in attrs_full.columns:
    if col == 'gauge_id':
        continue
    col_data = attrs_full[col]
    null_n = int(col_data.isna().sum())
    if pd.api.types.is_numeric_dtype(col_data):
        static_audit_rows.append({
            'column': col,
            'dtype': str(col_data.dtype),
            'null_count': null_n,
            'null_pct': round(null_n / 50 * 100, 1),
            'min': round(float(col_data.min()), 6) if null_n < 50 else None,
            'max': round(float(col_data.max()), 6) if null_n < 50 else None,
            'mean': round(float(col_data.mean()), 6) if null_n < 50 else None,
            'in_smoke_subset': col in SMOKE_STATIC_COLS,
        })
static_audit_df = pd.DataFrame(static_audit_rows)

# ---------------------------------------------------------------------------
# Section 4: Basin lists and splits
# ---------------------------------------------------------------------------
print('\n[4] Writing basin lists and splits ...')
import random

# all_basins.txt
all_basins_path = SPLIT_DIR / 'all_basins.txt'
with open(all_basins_path, 'w') as f:
    for s in pilot_staids:
        f.write(s + '\n')
print(f'    Written all_basins.txt ({len(pilot_staids)} STAIDs)')

def _write_splits(staids, out_dir, split_ratios=(0.70, 0.15, 0.15), seed=42):
    rng = random.Random(seed)
    shuffled = staids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * split_ratios[0])
    n_val   = round(n * split_ratios[1])
    n_test  = n - n_train - n_val
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train+n_val]
    test  = shuffled[n_train+n_val:]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lst in [('train_basins.txt', train), ('val_basins.txt', val), ('test_basins.txt', test)]:
        with open(out_dir / name, 'w') as f:
            for s in sorted(lst):
                f.write(s + '\n')
    return train, val, test

# All-50 smoke split
all50_dir = SPLIT_DIR / 'january_2023_smoke'
tr50, va50, te50 = _write_splits(list(pilot_staids), all50_dir, seed=42)
print(f'    all-50 split: train={len(tr50)}, val={len(va50)}, test={len(te50)}')

# Streamflow-covered basins only
sf_covered = sorted(s for s in pilot_staids if s not in set(no_streamflow_staids))
sfonly_dir = SPLIT_DIR / 'january_2023_smoke_streamflow_only'
tr_sf, va_sf, te_sf = _write_splits(list(sf_covered), sfonly_dir, seed=42)
print(f'    streamflow-only split: n={len(sf_covered)}, train={len(tr_sf)}, val={len(va_sf)}, test={len(te_sf)}')
# no_streamflow_basins.txt is already written above in the right location

# ---------------------------------------------------------------------------
# Section 5: Smoke config skeleton
# ---------------------------------------------------------------------------
print('\n[5] Writing smoke config skeleton ...')
smoke_cfg_content = f"""\
# DRAFT CONFIG SKELETON -- not yet run
# Flash-NH Stage 1 Milestone 2G
# Purpose: first technical smoke run on January 2023 pilot (streamflow-covered basins)
# Seed: 42
#
# IMPORTANT:
#   - This is a technical smoke config, NOT a scientific training config.
#   - Do not use for performance claims or hyperparameter tuning.
#   - Use january_2023_smoke_streamflow_only splits only.
#   - Both rtma_2d_K and rtma_2sh_kgkg are included (V1 decision; ablation to follow).
#   - rtma_sp_Pa, rtma_tcc_pct, rtma_10si_ms, rtma_i10fg_ms are in wide NetCDF but excluded here.
#
# Data directory structure (relative to this config file's location):
#   data_dir/
#     time_series/<STAID>.nc       -- per-basin NetCDF (GenericDataset)
#     attributes/attributes_smoke.csv
#     basin_lists/...

experiment_name: flashnh_stage1_smoke_v1
run_dir: /path/to/runs/flashnh_stage1   # UPDATE before running

# Data
dataset: generic
train_dir: {str(TS_DIR).replace(chr(92), '/')}
dynamic_inputs:
  - mrms_qpe_1h_mm
  - rtma_2t_K
  - rtma_2d_K
  - rtma_2sh_kgkg
  - rtma_10u_ms
  - rtma_10v_ms

target_variables:
  - qobs_m3s

static_attributes_path: {str(attrs_smoke_path).replace(chr(92), '/')}
static_attributes:
  - DRAIN_SQKM
  - LAT_GAGE
  - LNG_GAGE
  - BFI_AVE
  - RBI

# Basins (streamflow-covered only)
train_basin_file: {str(sfonly_dir / 'train_basins.txt').replace(chr(92), '/')}
validation_basin_file: {str(sfonly_dir / 'val_basins.txt').replace(chr(92), '/')}
test_basin_file: {str(sfonly_dir / 'test_basins.txt').replace(chr(92), '/')}

# Period (January 2023 only -- technical smoke)
train_start_date: "2023-01-01"
train_end_date: "2023-01-31"
validation_start_date: "2023-01-01"
validation_end_date: "2023-01-31"
test_start_date: "2023-01-01"
test_end_date: "2023-01-31"

# Model (LSTM placeholder -- update before running)
model: cudalstm
hidden_size: 64
initial_forget_bias: 3
dropout: 0.4
batch_size: 256
epochs: 10
learning_rate: 0.001

# Sequence
seq_length: 168     # 7 days hourly lookback
predict_last_n: 24  # predict last 24 hours of sequence

seed: 42
"""
cfg_path = CFG_DIR / 'smoke_v1.yml'
with open(cfg_path, 'w', encoding='utf-8') as f:
    f.write(smoke_cfg_content)
print(f'    Written smoke_v1.yml')

# ---------------------------------------------------------------------------
# Section 6: Manifests
# ---------------------------------------------------------------------------
print('\n[6] Writing manifests ...')

# dataset_manifest.json
per_basin_df = pd.DataFrame(per_basin_rows)
n_full_sf = int((per_basin_df['qobs_nan_hours'] == 0).sum())
n_partial  = int(((per_basin_df['qobs_nan_hours'] > 0) & (per_basin_df['qobs_nan_hours'] < 744)).sum())
n_all_nan  = int((per_basin_df['qobs_nan_hours'] == 744).sum())

manifest_dict = {
    'created_utc': CREATED_UTC,
    'milestone': 'Flash-NH Stage 1 Milestone 2G',
    'description': 'NeuralHydrology GenericDataset-compatible January 2023 pilot package',
    'n_basins': nc_written,
    'n_time_steps': 744,
    'time_start': '2023-01-01T00:00:00Z',
    'time_end': '2023-01-31T23:00:00Z',
    'time_frequency': 'hourly UTC',
    'wide_dynamic_vars': WIDE_DYN_VARS + ['qobs_m3s'],
    'smoke_dynamic_inputs': SMOKE_DYN_VARS,
    'smoke_static_attrs': SMOKE_STATIC_COLS,
    'target_variable': 'qobs_m3s',
    'streamflow_coverage': {
        'total_basins': 50,
        'basins_with_full_target': n_full_sf,
        'basins_with_partial_target': n_partial,
        'basins_all_nan_target': n_all_nan,
        'no_streamflow_staids': sorted(no_streamflow_staids),
    },
    'hydroatlas_join': {
        'source': str(HYDROATLAS_CSV),
        'n_pilot_matched': len(ha_matched),
        'n_pilot_unmatched': len(ha_unmatched),
        'unmatched_staids': ha_unmatched,
        'new_cols_added': len(ha_new_cols),
    },
    'attributes_full_cols': len(attrs_full.columns) - 1,  # exclude gauge_id
    'package_dir': str(PKG_ROOT),
    'nc_file_pattern': 'time_series/<STAID>.nc',
}
with open(MFST_DIR / 'dataset_manifest.json', 'w', encoding='utf-8') as f:
    json.dump(manifest_dict, f, indent=2, default=str)
print('    Written dataset_manifest.json')

# variable_schema.csv
schema_rows = []
for key, (nc_col, units, long_name, ts_meaning) in VAR_RENAME.items():
    arr = per_basin_df['forcing_nan_count'].sum()  # always 0 here
    schema_rows.append({
        'nc_varname': nc_col,
        'source_product': key[0],
        'source_variable': key[1],
        'units': units,
        'long_name': long_name,
        'timestep_meaning': ts_meaning,
        'in_wide_nc': True,
        'in_smoke_config': nc_col in SMOKE_DYN_VARS,
        'nan_count_total': 0,
    })
schema_rows.append({
    'nc_varname': 'qobs_m3s',
    'source_product': 'CAMELSH',
    'source_variable': 'streamflow',
    'units': 'm3 s**-1',
    'long_name': 'Observed streamflow',
    'timestep_meaning': 'instantaneous hourly discharge at gauge',
    'in_wide_nc': True,
    'in_smoke_config': True,
    'nan_count_total': int(per_basin_df['qobs_nan_hours'].sum()),
})
schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(MFST_DIR / 'variable_schema.csv', index=False)
print('    Written variable_schema.csv')

# static_attribute_audit.csv
static_audit_df.to_csv(MFST_DIR / 'static_attribute_audit.csv', index=False)
per_basin_df.to_csv(MFST_DIR / 'missingness_report.csv', index=False)
ha_audit_df.to_csv(MFST_DIR / 'hydroatlas_join_audit.csv', index=False)
print('    Written static_attribute_audit.csv, missingness_report.csv, hydroatlas_join_audit.csv')

# per_basin_summary.csv to audit dir
per_basin_df.to_csv(AUDIT_DIR / 'per_basin_summary.csv', index=False)

# ---------------------------------------------------------------------------
# Section 7: README
# ---------------------------------------------------------------------------
readme = f"""\
# Flash-NH Stage 1 — NeuralHydrology January 2023 Pilot Package

**Created:** {CREATED_UTC}
**Milestone:** 2G

## Package structure

```
time_series/             # 50 per-basin NetCDF files (<STAID>.nc)
attributes/
  attributes_full.csv    # all wide-eligible static attributes ({len(attrs_full.columns)-1} cols)
  attributes_smoke.csv   # smoke-run subset (5 cols: DRAIN_SQKM, LAT/LNG_GAGE, BFI_AVE, RBI)
basin_lists/
  all_basins.txt         # all 50 pilot STAIDs
  january_2023_smoke/    # all-50 technical splits (seed=42)
  january_2023_smoke_streamflow_only/  # 28-basin streamflow-covered splits (seed=42)
    no_streamflow_basins.txt  # 22 STAIDs with all-NaN qobs_m3s
configs/
  smoke_v1.yml           # DRAFT config skeleton -- not yet run
manifests/
  dataset_manifest.json
  variable_schema.csv
  static_attribute_audit.csv
  hydroatlas_join_audit.csv
  missingness_report.csv
```

## Dynamic variables (per-basin NetCDF)

Wide format: 10 forcing variables + qobs_m3s target.

Smoke config inputs: mrms_qpe_1h_mm, rtma_2t_K, rtma_2d_K, rtma_2sh_kgkg, rtma_10u_ms, rtma_10v_ms.

MRMS note: parquet stores variable='unknown'; renamed to mrms_qpe_1h_mm (GRIB param-ID issue).

## Streamflow coverage

- {n_full_sf}/50 basins: full qobs_m3s (0 NaN hours)
- {n_partial}/50 basins: partial qobs_m3s (some NaN hours)
- {n_all_nan}/50 basins: all-NaN qobs_m3s (no CAMELSH file)

Milestone 2H streamflow recovery is required before scientific training.

## HydroATLAS

Full HydroATLAS table joined (50/50 match after STAID normalization).
{len(ha_new_cols)} HydroATLAS columns added to attributes_full.csv.

## Splits

These are technical smoke splits (seed=42). Not scientific train/val/test splits.
Do not use for performance claims.

For any actual smoke training run, use january_2023_smoke_streamflow_only.
"""
with open(PKG_ROOT / 'README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print('    Written README.md')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
elapsed = _time.time() - t0
print('\n' + '=' * 70)
print(f'Builder complete in {elapsed:.1f}s')
print(f'Package dir: {PKG_ROOT}')
print(f'  NetCDF files: {nc_written}')
print(f'  Wide dynamic vars: {len(WIDE_DYN_VARS)} + qobs_m3s = {len(WIDE_DYN_VARS)+1}')
print(f'  attributes_full cols: {len(attrs_full.columns)-1}')
print(f'  attributes_smoke cols: {len(SMOKE_STATIC_COLS)}')
print(f'  HydroATLAS join: {len(ha_matched)}/50 matched, {len(ha_new_cols)} new cols')
print(f'  Streamflow: {n_full_sf} full, {n_partial} partial, {n_all_nan} all-NaN')
print('=' * 70)
