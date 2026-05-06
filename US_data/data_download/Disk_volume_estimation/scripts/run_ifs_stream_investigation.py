#!/usr/bin/env python3
"""
IFS stream/type investigation for 06 and 18 UTC historical retrievals.

Determines why 06/18 UTC fail while 00/12 UTC succeed under:
  - class=od
  - stream=oper
  - type=fc
  - levtype=sfc

Tests systematically:
  1. Minimal request with current settings
  2. Parameter/step incremental expansion
  3. Alternative stream/type combinations
  4. Area subset impact on retrieval success
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

RUN_ROOT = Path(__file__).parent.parent / 'reports' / 'audit_2026_04_29' / 'run_04_ifs_stream_investigation'
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
TEST_CYCLES = [6, 18]

# Base request parameters (all cycles)
BASE_PARAMS = {
    'class': 'od',
    'date': SAMPLE_DATE,
    'levtype': 'sfc',
    'grid': '0.25/0.25',
}

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
    """Return list of test configurations for 06 and 18."""
    tests = []
    
    for cycle in TEST_CYCLES:
        cycle_time = f'{cycle:02d}:00:00'
        
        # Test 1: Minimal request with area subset
        tests.append({
            'name': f'cycle_{cycle:02d}_minimal_with_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': '167.128',
            'step': '0',
            'area': AREA_SUBSET,
            'description': 'Minimal: 2T, step=0, with area subset',
        })
        
        # Test 2: Minimal request without area
        tests.append({
            'name': f'cycle_{cycle:02d}_minimal_no_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': '167.128',
            'step': '0',
            'area': None,
            'description': 'Minimal: 2T, step=0, no area subset',
        })
        
        # Test 3: Single param, full step range, with area
        tests.append({
            'name': f'cycle_{cycle:02d}_single_param_full_steps_with_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': '167.128',
            'step': '0/to/24/by/1',
            'area': AREA_SUBSET,
            'description': 'Single param (2T), full step range, with area',
        })
        
        # Test 4: Single param, full step range, no area
        tests.append({
            'name': f'cycle_{cycle:02d}_single_param_full_steps_no_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': '167.128',
            'step': '0/to/24/by/1',
            'area': None,
            'description': 'Single param (2T), full step range, no area',
        })
        
        # Test 5: All params, step=0, with area
        tests.append({
            'name': f'cycle_{cycle:02d}_all_params_step0_with_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': CURRENT_PARAM_STRING,
            'step': '0',
            'area': AREA_SUBSET,
            'description': 'All current params, step=0, with area',
        })
        
        # Test 6: All params, step=0, no area
        tests.append({
            'name': f'cycle_{cycle:02d}_all_params_step0_no_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': CURRENT_PARAM_STRING,
            'step': '0',
            'area': None,
            'description': 'All current params, step=0, no area',
        })
        
        # Test 7: All params, full steps, with area
        tests.append({
            'name': f'cycle_{cycle:02d}_all_params_full_steps_with_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': CURRENT_PARAM_STRING,
            'step': '0/to/24/by/1',
            'area': AREA_SUBSET,
            'description': 'All current params, full steps, with area',
        })
        
        # Test 8: All params, full steps, no area
        tests.append({
            'name': f'cycle_{cycle:02d}_all_params_full_steps_no_area',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'oper',
            'type': 'fc',
            'param': CURRENT_PARAM_STRING,
            'step': '0/to/24/by/1',
            'area': None,
            'description': 'All current params, full steps, no area',
        })
        
        # Test 9: Alternative stream (scda) with minimal
        tests.append({
            'name': f'cycle_{cycle:02d}_alt_stream_scda_fc_minimal',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'scda',
            'type': 'fc',
            'param': '167.128',
            'step': '0',
            'area': None,
            'description': 'Alternative: stream=scda, type=fc, minimal',
        })
        
        # Test 10: Alternative stream (scda) control forecast
        tests.append({
            'name': f'cycle_{cycle:02d}_alt_stream_scda_cf_minimal',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'scda',
            'type': 'cf',
            'param': '167.128',
            'step': '0',
            'area': None,
            'description': 'Alternative: stream=scda, type=cf, minimal',
        })
        
        # Test 11: Alternative stream (scda) analysis
        tests.append({
            'name': f'cycle_{cycle:02d}_alt_stream_scda_an_minimal',
            'cycle': cycle,
            'cycle_time': cycle_time,
            'stream': 'scda',
            'type': 'an',
            'param': '167.128',
            'step': '0',
            'area': None,
            'description': 'Alternative: stream=scda, type=an, minimal',
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
    log_path = LOGS_DIR / f"ifs_{result['cycle']:02d}.log"
    lines = [
        f"name={result['name']}",
        f"cycle={result['cycle']:02d}",
        f"stream={result['stream']}",
        f"type={result['type']}",
        f"levtype=sfc",
        f"param={result['param']}",
        f"step={result['step']}",
        f"grid=0.25/0.25",
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
    
    if test_config['area']:
        request['area'] = test_config['area']
    
    # Save request JSON (before adding target)
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
        
        # Build MARS request (without target for execution)
        mars_request = {k: v for k, v in request.items()}
        
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
    summary_json = SUMMARY_DIR / 'ifs_stream_investigation_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # CSV summary
    summary_csv = SUMMARY_DIR / 'ifs_stream_investigation_summary.csv'
    if results:
        keys = ['name', 'cycle', 'stream', 'type', 'param', 'step', 'area', 'success', 'expected_fields', 'retrieved_fields', 'output_bytes', 'error_message', 'elapsed_time_sec']
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for result in results:
                row = {k: result.get(k, '') for k in keys}
                writer.writerow(row)
    
    # Markdown summary (with UTF-8 encoding to handle Unicode characters)
    summary_md = SUMMARY_DIR / 'ifs_stream_investigation_summary.md'
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write('# IFS Stream Investigation Summary\n\n')
        f.write(f'Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n')
        
        # Group by cycle
        by_cycle = {}
        for result in results:
            cycle = result['cycle']
            if cycle not in by_cycle:
                by_cycle[cycle] = []
            by_cycle[cycle].append(result)
        
        for cycle in sorted(by_cycle.keys()):
            f.write(f'## Cycle {cycle:02d} UTC\n\n')
            for result in by_cycle[cycle]:
                status = 'PASS' if result['success'] else 'FAIL'
                f.write(f'- **{result["name"]}**: {status}\n')
                f.write(f'  - Description: {result["description"]}\n')
                f.write(f'  - Stream: {result["stream"]}, Type: {result["type"]}\n')
                f.write(f'  - Param: {result["param"]}, Step: {result["step"]}\n')
                f.write(f'  - Area: {result["area"]}\n')
                if result['error_message']:
                    f.write(f'  - Error: {result["error_message"]}\n')
                f.write('\n')


def build_review_bundle(results):
    """Copy key artifacts into review bundle."""
    # Copy summaries
    for f in SUMMARY_DIR.glob('*'):
        shutil.copy(f, REVIEW_BUNDLE_SUMMARY / f.name)
    
    # Copy request JSONs (minimal sample)
    request_files = list(REQUESTS_DIR.glob('*.json'))
    for f in request_files[:5]:  # Keep first 5 as example
        shutil.copy(f, REVIEW_BUNDLE_REQUESTS / f.name)
    
    # Write manifest
    manifest = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run_root': str(RUN_ROOT),
        'review_bundle': str(REVIEW_BUNDLE_DIR),
        'results_count': len(results),
        'cycles_tested': TEST_CYCLES,
    }
    with open(REVIEW_BUNDLE_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    """Main investigation runner."""
    print('=' * 70)
    print('IFS STREAM INVESTIGATION')
    print('=' * 70)
    print(f'Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}')
    print(f'Run root: {RUN_ROOT}')
    print()
    
    # Setup
    setup_directories()
    print(f'✓ Output directories ready')
    
    # Get test matrix
    tests = get_test_matrix()
    print(f'✓ Test matrix ready: {len(tests)} tests for cycles {TEST_CYCLES}')
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
            print(f'  {status} ({elapsed})')
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
    print('INVESTIGATION COMPLETE')
    print(f'Results: {RUN_ROOT}')
    print('=' * 70)


if __name__ == '__main__':
    main()
