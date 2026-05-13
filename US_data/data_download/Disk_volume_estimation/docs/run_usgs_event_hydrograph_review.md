# Run USGS Event Hydrograph Review (Flash-NH)

This document explains how to run the bounded event-centered hydrograph review.

Smoke test (10 basins):

```bash
c:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation/.venv/Scripts/python.exe scripts/run_usgs_event_hydrograph_review.py --max-basins 10
```

Full run (default sampling):

```bash
c:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation/.venv/Scripts/python.exe scripts/run_usgs_event_hydrograph_review.py
```

Notes:
- The script reads the completed screening results at `reports/flashnh_usgs_rbi_screening_wy2024_v001`.
- It will write lightweight hourly CSVs, event plots, and summary tables to `reports/flashnh_usgs_event_hydrograph_review_v001/`.
- By default the script only saves raw USGS payloads if `--debug-raw` is passed.
- Do not commit generated outputs; commit only the script and this doc.
