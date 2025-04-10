import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# === CONFIGURATION ===
combined_folder = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_2025-04-08'
flow_column = "Flow_m3_sec"
datetime_column = "datetime"

# Define training and validation periods
val_years = list(range(2015, 2018))
test_years = list(range(2019, 2022))

# === INIT RESULTS ===
rankings = []

# === HELPER FUNCTION TO ASSIGN HYDROLOGICAL YEAR ===
def assign_hydro_year(date):
    return date.year if date.month < 10 else date.year + 1

# === PROCESS EACH STATION FILE ===
if os.path.exists(combined_folder):
    all_files = [f for f in os.listdir(combined_folder) if f.endswith("_combined.csv")]
    for file in tqdm(all_files, desc="Processing stations"):
        station_id = file.split('_')[0]
        file_path = os.path.join(combined_folder, file)
        df = pd.read_csv(file_path, parse_dates=[datetime_column])

        if df.empty or flow_column not in df.columns:
            continue

        df = df.dropna(subset=[flow_column])
        df["hydro_year"] = df[datetime_column].apply(assign_hydro_year)

        # Group by hydrological year and get annual maxima
        annual_max = df.groupby("hydro_year")[flow_column].max().reset_index()
        annual_max["rank"] = annual_max[flow_column].rank(ascending=False, method="min").astype(int)
        annual_max["decile"] = pd.qcut(
            annual_max[flow_column].rank(ascending=True),
            10,
            labels=False,
            duplicates='drop'
        ) + 1
        annual_max["station_id"] = station_id

        rankings.append(annual_max)

    # === COMBINE RESULTS ===
    ranking_df = pd.concat(rankings)

    # === CREATE PIVOT TABLE ===
    rank_table = ranking_df.pivot(index="station_id", columns="hydro_year", values="rank")
    decile_table = ranking_df.pivot(index="station_id", columns="hydro_year", values="decile")

    # === FILTER VALIDATION AND TEST SETS ===
    val_df = ranking_df[ranking_df["hydro_year"].isin(val_years)]
    test_df = ranking_df[ranking_df["hydro_year"].isin(test_years)]

    # === HISTOGRAMS OF RANKS ===
    plt.figure(figsize=(10, 5))
    plt.hist(val_df["rank"], bins=range(1, val_df["rank"].max() + 2), edgecolor="black", alpha=0.7, label="Validation", align="left")
    plt.hist(test_df["rank"], bins=range(1, test_df["rank"].max() + 2), edgecolor="black", alpha=0.7, label="Test", align="left")
    plt.xlabel("Rank of Annual Maximum (1 = Highest of All Years)")
    plt.ylabel("Frequency Across Stations")
    plt.title("Histogram of Annual Maximum Ranks (Validation vs Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === HISTOGRAMS OF DECILES ===
    plt.figure(figsize=(10, 5))
    plt.hist(val_df["decile"], bins=range(1, 12), edgecolor="black", alpha=0.7, label="Validation", align="left")
    plt.hist(test_df["decile"], bins=range(1, 12), edgecolor="black", alpha=0.7, label="Test", align="left")
    plt.xlabel("Decile of Annual Maximum (1 = Lowest, 10 = Highest)")
    plt.ylabel("Frequency Across Stations")
    plt.title("Histogram of Annual Maximum Deciles (Validation vs Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === DISPLAY RANK TABLE ===
    print(rank_table)

else:
    print(f"Folder not found: {combined_folder}")
