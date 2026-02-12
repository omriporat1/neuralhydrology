from src.data_usgs import fetch_usgs_iv_discharge

def main():
    site = "0136230002"
    # fetch a small window first (faster + easier to debug)
    q = fetch_usgs_iv_discharge(site, "2016-01-01", "2016-01-10")
    print(q.head())
    print(q.tail())
    print("n=", len(q), "missing=", q.isna().sum())
    print("freq guess:", q.index.to_series().diff().value_counts().head())

if __name__ == "__main__":
    main()
