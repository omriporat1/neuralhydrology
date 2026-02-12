from src.data_usgs import fetch_usgs_iv_discharge
from src.aggregation import to_hourly_candidates

gauge = "0136230002"  # any gauge you know works
year = 2016

start = f"{year-1}-12-31"
end = f"{year+1}-01-02"

q_iv = fetch_usgs_iv_discharge(gauge, start, end)

cands = to_hourly_candidates(q_iv, nearest_tolerance="10min")

print("nearest head:")
print(cands["nearest_10min"].head(3).index)

print("\nmean head:")
print(cands["mean"].head(3).index)
