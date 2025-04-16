import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import seaborn as sns

# === CONFIGURATION ===
combined_folder = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_2025-04-08'
flow_column = "Flow_m3_sec"
datetime_column = "datetime"
summary_file = "annual_maxima_summary.csv"

# Define validation and test periods
val_years = list(range(2015, 2018))
test_years = list(range(2019, 2022))

# === HELPER FUNCTION TO ASSIGN HYDROLOGICAL YEAR ===
def assign_hydro_year(date):
    return date.year if date.month < 10 else date.year + 1

# === LOAD OR CREATE ANNUAL MAXIMA SUMMARY ===
if os.path.exists(summary_file):
    print(f"Loading precomputed summary from {summary_file}")
    ranking_df = pd.read_csv(summary_file)
else:
    rankings = []
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

            # Exclude zero or missing values
            annual_max = annual_max[annual_max[flow_column] > 0]

            annual_max["rank"] = annual_max[flow_column].rank(ascending=False, method="min").astype(int)
            annual_max["decile"] = pd.qcut(
                annual_max[flow_column].rank(ascending=False),
                10,
                labels=False,
                duplicates='drop'
            ) + 1
            annual_max["station_id"] = station_id

            rankings.append(annual_max)

        # Combine and save
        ranking_df = pd.concat(rankings)
        ranking_df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
    else:
        print(f"Folder not found: {combined_folder}")
        exit()

# === NORMALIZE FLOW VALUES (Z-score per station) ===
ranking_df["flow_z"] = ranking_df.groupby("station_id")[flow_column].transform(
    lambda x: (x - x.mean()) / x.std()
)

# === CREATE PIVOT TABLES ===
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
plt.xlabel("Decile of Annual Maximum (1 = Highest, 10 = Lowest)")
plt.ylabel("Frequency Across Stations")
plt.title("Histogram of Annual Maximum Deciles (Validation vs Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === HEATMAP OF Z-SCORED FLOWS ===
plt.figure(figsize=(14, 8))
z_heatmap = ranking_df.pivot(index="station_id", columns="hydro_year", values="flow_z")
sns.heatmap(z_heatmap, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Heatmap of Z-Scored Annual Maxima by Station and Year")
plt.xlabel("Hydrological Year")
plt.ylabel("Station")
plt.tight_layout()
plt.show()

# === BOXPLOT OF Z-SCORED FLOWS PER YEAR ===
plt.figure(figsize=(12, 6))
sns.boxplot(x="hydro_year", y="flow_z", data=ranking_df)
plt.xticks(rotation=45)
plt.title("Distribution of Z-Scored Annual Maxima per Year")
plt.xlabel("Hydrological Year")
plt.ylabel("Z-score of Annual Max Flow")
plt.tight_layout()
plt.show()

# === LINE PLOT FOR SELECTED STATIONS ===
# choose 10 unique stations from the ranking_df["station_id"] randomly:
sample_stations = ranking_df["station_id"].drop_duplicates().sample(6, random_state=42).values
plt.figure(figsize=(12, 6))
for station in sample_stations:
    station_data = ranking_df[ranking_df["station_id"] == station]
    plt.plot(station_data["hydro_year"], station_data["flow_z"], label=station)

plt.legend()
plt.xlabel("Hydrological Year")
plt.ylabel("Z-score of Annual Max Flow")
plt.title("Z-Scored Annual Maxima Over Time (Selected Stations)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === SHOW SUMMARY TABLE ===
# print(rank_table)


