from __future__ import annotations

import pandas as pd
import requests

CFS_TO_M3S = 0.028316846592  # exact enough


def fetch_usgs_iv_discharge(
    site_no: str,
    start: str,
    end: str,
    timeout: int = 60,
) -> pd.Series:
    """
    Fetch USGS Instantaneous Values (IV) discharge (00060) for one site.

    Parameters
    ----------
    site_no : str
        USGS site number (gauge id), e.g. "0136230002"
    start, end : str
        ISO-like timestamps accepted by USGS, e.g. "2016-01-01", "2017-01-01"
        end is inclusive-ish depending on service; we’ll clip later.
    timeout : int
        HTTP timeout in seconds

    Returns
    -------
    pd.Series
        Discharge time series in m3/s, indexed by timezone-aware UTC timestamps.
        Missing values are NaN.
    """
    url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": "00060",
        # "siteStatus": "all",  # optional
        "startDT": start,
        "endDT": end,
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()

    ts_list = js.get("value", {}).get("timeSeries", [])
    if not ts_list:
        raise RuntimeError(
            f"No timeseries returned for site={site_no} start={start} end={end}. "
            f"Check site id, date range, and whether IV exists for that site."
        )

    # Often only one series, but be robust
    # We want variable 00060
    series = ts_list[0]
    values = series["values"][0]["value"]

    # Build dataframe
    df = pd.DataFrame(values)
    if df.empty:
        raise RuntimeError(f"Empty values for site={site_no} start={start} end={end}")

    # 'dateTime' is ISO with offset, e.g. 2016-01-01T00:00:00.000-05:00
    dt = pd.to_datetime(df["dateTime"], utc=True)
    q = pd.to_numeric(df["value"], errors="coerce")

    # Units can be cfs; convert to m3/s.
    # To be extra safe, check unit metadata:
    unit = (
        series.get("variable", {})
        .get("unit", {})
        .get("unitCode", "")
        .lower()
    )
    # Most common: "ft3/s"
    if "ft3" in unit or "cfs" in unit or unit == "":
        q = q * CFS_TO_M3S

    out = pd.Series(q.values, index=dt, name="q_usgs_iv_m3s").sort_index()
    return out
