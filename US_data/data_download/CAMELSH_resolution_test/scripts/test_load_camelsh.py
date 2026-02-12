from src.data_camelsh import load_camelsh_hourly

def main():
    q = load_camelsh_hourly("0136230002", 2016)
    print(q.head())
    print(q.tail())
    print("n=", len(q), "missing=", q.isna().sum())

if __name__ == "__main__":
    main()