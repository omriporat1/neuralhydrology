# Run USGS Event Hydrograph Review (Flash-NH)

This document explains how to run the bounded event-centered hydrograph review.
The refined workflow lives in `scripts/run_usgs_event_hydrograph_review_v002.py`.

No-download dry run using cached hourly series when available:

```bash
c:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation/.venv/Scripts/python.exe scripts/run_usgs_event_hydrograph_review_v002.py --max-review-basins 10 --no-download
```

Small selected-basin run that can reuse cache and fetch only missing selected basins:

```bash
c:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation/.venv/Scripts/python.exe scripts/run_usgs_event_hydrograph_review_v002.py --max-review-basins 10
```

Full bounded review with the default sample:

```bash
c:/PhD/Python/neuralhydrology/US_data/data_download/Disk_volume_estimation/.venv/Scripts/python.exe scripts/run_usgs_event_hydrograph_review_v002.py
```

Notes:
- The script reads the completed screening results at `reports/flashnh_usgs_rbi_screening_wy2024_v001`.
- It writes the refined review outputs to `reports/flashnh_usgs_event_hydrograph_review_v002/`.
- Existing hourly CSVs in `reports/flashnh_usgs_event_hydrograph_review_v001/hourly_series/` are reused as cache when available.
- The sample is stratified across RBI bands and force-includes the four highlighted basins: `01521500`, `07382000`, `02310700`, and `09513860`.
- Raw USGS JSON is only saved when `--debug-raw` is passed.
- Do not commit generated outputs; commit only the script and this doc.
