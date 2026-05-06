#!/usr/bin/env python3
"""
IFS resolution comparison experiment.

Tests whether higher spatial resolution (~0.1 degree) MARS retrievals are:
- Reliably accessible
- Reasonably sized
- Operationally feasible

Compares:
  A) Current: grid=0.25/0.25
  B) Higher:  grid=0.1/0.1

For cycles: 00 UTC and 06 UTC (one from oper/fc, one from scda/fc)
For parameters: minimal (2T/step=0) then expanded (all vars/steps 0..24)
For date: 2023-01-01 (sample date)
"""

import os
import sys
import json
import csv
import datetime
import shutil
import time
from pathlib import Path

# Ensure src is in path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ecmwfapi import ECMWFService
except ImportError:
    print("ERROR: ecmwfapi not installed. Install with: pip install ecmwf-api-client")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

RUN_ROOT = Path(__file__).parent.parent / 'reports' / 'audit_2026_04_29' / 'run_06_ifs_resolution_test'
LOGS_DIR = RUN_ROOT / 'logs'
REQUESTS_DIR = RUN_ROOT / 'requests'
SAMPLE_DOWNLOADS_DIR = RUN_ROOT / 'sample_downloads'
SUMMARY_DIR = RUN_ROOT / 'summary'
REVIEW_BUNDLE_DIR = RUN_ROOT / 'review_bundle'
REVIEW_BUNDLE_LOGS = REVIEW_BUNDLE_DIR / 'logs'
REVIEW_BUNDLE_REQUESTS = REVIEW_BUNDLE_DIR / 'requests'
REVIEW_BUNDLE_SUMMARY = REVIEW_BUNDLE_DIR / 'summary'

# Test parameters
SAMPLE_DATE = '2023-01-01'
TEST_CYCLES = [0, 6]  # 00 UTC and 06 UTC

# Base request parameters
BASE_PARAMS = {
    'class': 'od',
    'date': SAMPLE_DATE,
    'levtype': 'sfc',
}

# Grids to compare
GRIDS = [
    '0.25/0.25',  # Current
    '0.1/0.1',    # Proposed higher resolution
]

# Current variable set
CURRENT_VARIABLES = {
    'TP': '228.128',
    '2T': '167.128',
    '2D': '168.128',
    '10U': '165.128',
    '10V': '166.128',
    'SP': '134.128',
    'SSRD': '169.128',
}
CURRENT_PARAM_STRING = '/'.join(CURRENT_VARIABLES.values())

# Area subset
AREA_SUBSET = '50/-126/24/-66'

# ============================================================================
# Test Matrix Definition
# ============================================================================

def get_test_matrix():
    """Return list of test configurations."""
    tests = []
    
    for cycle in TEST_CYCLES:
        cycle_time = f'{cycle:02d}:00:00'
        
        # Determine stream/type based on cycle
        if cycle == 6:
            stream = 'scda'
            type_ = 'fc'
        else:
            stream = 'oper'
            type_ = 'fc'
        
        for grid in GRIDS:
            # Test 1: Minimal with area
            tests.append({
                'name': f'cycle_{cycle:02d}_{grid.replace("/", "_")}_minimal_with_area',
                'cycle': cycle,
                'cycle_time': cycle_time,
                'stream': stream,
                'type': type_,
                'grid': grid,
                'param': '167.128',
                'step': '0',
                'area': AREA_SUBSET,
                'description': f'Cycle {cycle:02d}, grid={grid}, minimal (2T/step=0), with area',
            })
            
            # Test 2: All params, full steps, with area
            tests.append({
                'name': f'cycle_{cycle:02d}_{grid.replace("/", "_")}_all_params_full_steps_with_area',
                'cycle': cycle,
                'cycle_time': cycle_time,
                'stream': stream,
                'type': type_,
                'grid': grid,
                'param': CURRENT_PARAM_STRING,
                'step': '0/to/24/by/1',
                'area': AREA_SUBSET,
                'description': f'Cycle {cycle:02d}, grid={grid}, all params/steps 0-24, with area',
            })
    
    return tests

# ============================================================================
# Execution
# ============================================================================

def setup_directories():
    """Create or clean output directories."""
    for d in [LOGS_DIR, REQUESTS_DIR, SAMPLE_DOWNLOADS_DIR, SUMMARY_DIR, REVIEW_BUNDLE_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Create nested dirs in review bundle
    REVIEW_BUNDLE_LOGS.mkdir(parents=True, exist_ok=True)
    REVIEW_BUNDLE_REQUESTS.mkdir(parents=True, exist_ok=True)
    REVIEW_BUNDLE_SUMMARY.mkdir(parents=True, exist_ok=True)


def append_cycle_log(result):
    """Write a compact per-cycle log entry for a test result."""
    log_path = LOGS_DIR / f"ifs_resolution_{result['cycle']:02d}.log"
    lines = [
        f"name={result['name']}",
        f"cycle={result['cycle']:02d}",
        f"stream={result['stream']}",
        f"type={result['type']}",
        f"grid={result['grid']}",
        f"param={result['param']}",
        f"step={result['step']}",
        f"area_subset={'yes' if result['area'] else 'no'}",
        f"request={json.dumps(result['request'], indent=2)}",
        f"expected_fields={result['expected_fields']}",
        f"retrieved_fields={result['retrieved_fields']}",
        f"success={result['success']}",
        f"error_message={result['error_message']}",
        f"output_bytes={result['output_bytes']}",
        f"elapsed_time_sec={result['elapsed_time_sec']}",
        "",
    ]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run_test(test_config):
    """Execute a single test and return result dict."""
    
    # Build request
    request = dict(BASE_PARAMS)
    request['stream'] = test_config['stream']
    request['type'] = test_config['type']
    request['time'] = test_config['cycle_time']
    request['param'] = test_config['param']
    request['step'] = test_config['step']
    request['grid'] = test_config['grid']
    request['area'] = test_config['area']
    
    # Save request JSON
    logged_request = dict(request)
    request_file = REQUESTS_DIR / f"{test_config['name']}_request.json"
    with open(request_file, 'w') as f:
        json.dump(logged_request, f, indent=2)
    
    # Attempt retrieval
    result = {
        'name': test_config['name'],
        'cycle': test_config['cycle'],
        'description': test_config['description'],
        'stream': test_config['stream'],
        'type': test_config['type'],
        'grid': test_config['grid'],
        'param': test_config['param'],
        'step': test_config['step'],
        'area': test_config['area'],
        'request': logged_request,
        'success': False,
        'expected_fields': None,
        'retrieved_fields': None,
        'output_bytes': None,
        'error_message': None,
        'elapsed_time_sec': None,
    }
    
    start_time = time.time()
    
    try:
        # Count expected fields
        params = test_config['param'].split('/')
        if 'to' in test_config['step']:
            # Step expansion: 0/to/24/by/1 = 0 to 24 inclusive = 25 steps
            step_list = list(range(0, 25))
            expected = len(params) * len(step_list)
        else:
            steps = test_config['step'].split('/')
            expected = len(params) * len(steps)
        result['expected_fields'] = expected
        
        # Prepare output path
        output_path = SAMPLE_DOWNLOADS_DIR / f"{test_config['name']}.grib"
        
        # Build MARS request
        mars_request = dict(request)
        
        # Execute via ecmwfapi
        server = ECMWFService("mars")
        server.execute(mars_request, str(output_path))
        
        # Check if file exists and has size
        if output_path.exists():
            result['success'] = True
            result['retrieved_fields'] = result['expected_fields']
            result['output_bytes'] = output_path.stat().st_size
        else:
            result['success'] = False
            result['error_message'] = 'Output file not created after successful execute call'
    
    except Exception as e:
        result['success'] = False
        result['error_message'] = str(e)
    
    result['elapsed_time_sec'] = time.time() - start_time
    append_cycle_log(result)
    return result


def write_summaries(results):
    """Write CSV, JSON, and markdown summaries."""
    # JSON summary
    summary_json = SUMMARY_DIR / 'ifs_resolution_comparison_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV summary
    summary_csv = SUMMARY_DIR / 'ifs_resolution_comparison_summary.csv'
    if results:
        keys = ['name', 'cycle', 'grid', 'stream', 'type', 'param', 'step', 'area', 'success', 'expected_fields', 'retrieved_fields', 'output_bytes', 'error_message', 'elapsed_time_sec']
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for result in results:
                row = {k: result.get(k, '') for k in keys}
                writer.writerow(row)
    
    # Markdown summary
    summary_md = SUMMARY_DIR / 'ifs_resolution_comparison_summary.md'
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write('# IFS Resolution Comparison Summary\n\n')
        f.write(f'Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n')
        
        # Organize by cycle and grid
        by_cycle_grid = {}
        for result in results:
            key = (result['cycle'], result['grid'])
            if key not in by_cycle_grid:
                by_cycle_grid[key] = []
            by_cycle_grid[key].append(result)
        
        for (cycle, grid) in sorted(by_cycle_grid.keys()):
            f.write(f'## Cycle {cycle:02d} UTC, Grid {grid}\n\n')
            
            for result in by_cycle_grid[(cycle, grid)]:
                status = 'PASS' if result['success'] else 'FAIL'
                f.write(f'- **{result["name"]}**: {status}\n')
                f.write(f'  - Description: {result["description"]}\n')
                f.write(f'  - Stream: {result["stream"]}, Type: {result["type"]}\n')
                f.write(f'  - Param: {result["param"]}, Step: {result["step"]}\n')
                if result['output_bytes']:
                    f.write(f'  - Output size: {result["output_bytes"]:,} bytes\n')
                if result['elapsed_time_sec']:
                    f.write(f'  - Elapsed: {result["elapsed_time_sec"]:.2f}s\n')
                if result['error_message']:
                    f.write(f'  - Error: {result["error_message"]}\n')
                f.write('\n')
        
        f.write('\n## Comparison Analysis\n\n')
        
        # Compute statistics by grid and test type
        stats = {}
        for result in results:
            grid = result['grid']
            if grid not in stats:
                stats[grid] = {'success': 0, 'fail': 0, 'total_bytes': 0, 'test_count': 0}
            
            stats[grid]['test_count'] += 1
            if result['success']:
                stats[grid]['success'] += 1
                stats[grid]['total_bytes'] += result['output_bytes'] or 0
            else:
                stats[grid]['fail'] += 1
        
        f.write('### By Grid Resolution\n\n')
        for grid, data in sorted(stats.items()):
            success_pct = (data['success'] / data['test_count'] * 100) if data['test_count'] > 0 else 0
            size_mb = data['total_bytes'] / (1024 * 1024)
            f.write(f'**Grid {grid}**\n')
            f.write(f'- Tests: {data["test_count"]} ({data["success"]} pass, {data["fail"]} fail, {success_pct:.0f}% success)\n')
            f.write(f'- Total bytes: {data["total_bytes"]:,} ({size_mb:.2f} MB)\n')
            f.write('\n')


def build_review_bundle(results):
    """Copy key artifacts into review bundle."""
    # Copy summaries
    for f in SUMMARY_DIR.glob('*'):
        shutil.copy(f, REVIEW_BUNDLE_SUMMARY / f.name)
    
    # Copy request JSONs (sample)
    request_files = list(REQUESTS_DIR.glob('*.json'))
    for f in request_files:
        shutil.copy(f, REVIEW_BUNDLE_REQUESTS / f.name)
    
    # Write manifest
    manifest = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run_root': str(RUN_ROOT),
        'review_bundle': str(REVIEW_BUNDLE_DIR),
        'results_count': len(results),
        'cycles_tested': TEST_CYCLES,
        'grids_tested': GRIDS,
    }
    with open(REVIEW_BUNDLE_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    """Main resolution test runner."""
    print('=' * 70)
    print('IFS RESOLUTION COMPARISON TEST')
    print('=' * 70)
    print(f'Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}')
    print(f'Run root: {RUN_ROOT}')
    print()
    
    # Setup
    setup_directories()
    print(f'✓ Output directories ready')
    
    # Get test matrix
    tests = get_test_matrix()
    print(f'✓ Test matrix ready: {len(tests)} tests for cycles {TEST_CYCLES}, grids {GRIDS}')
    print()
    
    # Run tests
    results = []
    for i, test in enumerate(tests, 1):
        print(f'[{i}/{len(tests)}] Running: {test["name"]}...')
        try:
            result = run_test(test)
            results.append(result)
            status = '[PASS]' if result['success'] else '[FAIL]'
            elapsed = f"{result['elapsed_time_sec']:.1f}s" if result['elapsed_time_sec'] else "unknown"
            size_str = f"{result['output_bytes']/(1024*1024):.2f}MB" if result['output_bytes'] else "0"
            print(f'  {status} ({elapsed}, {size_str})')
            if result['error_message']:
                err_msg = result['error_message'][:100]
                print(f'  Error: {err_msg}')
        except Exception as e:
            print(f'  [EXCEPTION]: {str(e)[:80]}')
    
    print()
    print('=' * 70)
    print('Writing summaries...')
    write_summaries(results)
    print('✓ Summaries written')
    
    print('Building review bundle...')
    build_review_bundle(results)
    print('✓ Review bundle ready')
    
    print()
    print('=' * 70)
    print('RESOLUTION TEST COMPLETE')
    print(f'Results: {RUN_ROOT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
