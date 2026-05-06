# Decision Log

## 2026-05-06

- Confirmed GFS `.idx` byte-range extraction end-to-end; the acquisition path is validated and frozen except for plotting.
- Confirmed IFS 00/12 UTC MARS retrievals succeed, while 06/18 UTC remain unresolved and should stay open as an access/request issue.
- Confirmed IMERG NC4 download is valid; crop logic still needs repair and final nonzero selected-CONUS confirmation.
- Retained RTMA despite size/time cost because it is scientifically important for the Stage 1 pipeline.

## 2026-05-06 IFS Stream Investigation

- Tested `oper/fc` requests for 06 UTC and 18 UTC on 2023-01-01 with `area` subset and without `area`; all returned `MARS_EXPECTED_FIELDS` with 0 retrieved fields.
- Tested `oper/fc` with the current variable set at `step=0` and `step=0/to/24/by/1`; both area-subset and full-domain forms failed for 06 UTC and 18 UTC.
- Tested `scda` alternatives with a minimal `2T` request at `step=0`.
- `scda/type=fc` succeeded for both 06 UTC and 18 UTC.
- `scda/type=an` succeeded for both 06 UTC and 18 UTC.
- `scda/type=cf` failed with `MARS_EXPECTED_FIELDS` and 0 retrieved fields.
- The area subset `50/-126/24/-66` did not explain the oper failures.
- **Stream fix implemented**: 00/12 UTC use `oper/fc`, while 06/18 UTC use `scda/fc`.
- Provisional recommendation: use the deterministic `scda` path for historical 06/18 UTC retrievals; keep `oper` as the working path for 00/12 UTC and do not assume only 2 historical IFS cycles exist.

## 2026-05-06 IFS Resolution Comparison

- Tested 2023-01-01 historical retrievals at two grid resolutions:
  - **0.25/0.25**: current configuration
  - **0.1/0.1**: proposed higher resolution
- For cycles 00 UTC (oper/fc) and 06 UTC (scda/fc), tested both minimal requests (1 variable, 1 step) and full requests (7 variables, 25 steps).
- **Results**: Both resolutions retrieved successfully with 100% pass rate.
  - 0.25/0.25: ~17.9 MB total for both cycles (8.9 MB per cycle average)
  - 0.1/0.1: ~110.5 MB total for both cycles (55.3 MB per cycle average)
  - Ratio: 0.1/0.1 is ~6.2× larger
- **Timing**: Higher resolution added ~5–15 seconds per request but remained well within operational tolerance.
- **Recommendation**: Adopt **0.1/0.1 resolution** to align with IFS's scientific value (higher resolution than GFS).
  - Justification: Estimated annual burden ~80 GB (acceptable for 2-year window); retrieval time penalty negligible; area subset remains supported.
  - Contingency: Revert to 0.25/0.25 requires single config change if burden becomes untenable.
- **Decision**: Update `IfsMarsConfig.grid` to `0.1/0.1` and document stream logic in code comments.