"""
This script reads the optimized Stations_With_3_Gauges_Wide_assessment.csv file and creates the unified IMS+IHS
data accordingly, only for the overlapping record periods of each station.

This means:
1. read the optimized Stations_With_3_Gauges_Wide_assessment.csv file
2. for each station, read the corresponding IMS and IHS data
3. clean the data from NaNs, negative values, and values above threshold
3. find the overlapping period between the IMS and IHS data
4. unify the data for the overlapping period, excluding rows with missing values
5. save the unified data to a new csv file
"""


# import relevant:
import os
import pandas as pd
import multiprocessing as mp
import pickle

# Define paths
station_info_file = r'C:\PhD\Data\IHS_IMS_wide\Stations_With_3_Gauges_Wide_assessment.csv'
stations_folder = r'C:\PhD\Data\IHS\stations_hydrographs'
rain_gauges_folder = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted'
output_folder = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_' + str(pd.Timestamp.now().date())
rain_gauge_pickle = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\rain_gauge_data.pkl'




# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load station-gauge mapping
df_mapping = pd.read_csv(station_info_file)

# remove from df_mapping all line lines where "assessment" equals 9:
df_mapping = df_mapping[df_mapping['assessment'] != 9]

# Sanity check: Ensure all files exist before processing
missing_files = []
all_gauge_ids = pd.unique(df_mapping[['gauge_id_1', 'gauge_id_2', 'gauge_id_3']].values.ravel())

for gauge_id in all_gauge_ids:
    if not os.path.exists(os.path.join(rain_gauges_folder, f"{gauge_id.astype(int)}.csv")):
        missing_files.append(f"Rain gauge file missing: {gauge_id}.csv")
for station_id in df_mapping['station_id']:
    if not os.path.exists(os.path.join(stations_folder, f"{station_id}.csv")):
        missing_files.append(f"Station file missing: {station_id}.csv")

if missing_files:
    print("The following files are missing. Please resolve before running the process:")
    for file in missing_files:
        print(file)
    exit(1)

# Check if the rain gauge data is already pickled
if os.path.exists(rain_gauge_pickle):
    with open(rain_gauge_pickle, 'rb') as f:
        rain_gauge_data = pickle.load(f)
        print("Loaded rain gauge data from pickle.")
else:
    # Load and filter all rain gauge data into a dictionary
    rain_gauge_data = {}
    count = 0
    for gauge_id in all_gauge_ids:
        gauge_id = int(gauge_id)
        gauge_file = os.path.join(rain_gauges_folder, f"{gauge_id}.csv")
        df_gauge = pd.read_csv(gauge_file, parse_dates=['datetime'], dayfirst=True)
        df_gauge['datetime'] = pd.to_datetime(df_gauge['datetime']).dt.tz_localize(None)
        df_gauge = df_gauge[(df_gauge['Rain'] >= 0) & (df_gauge['Rain'] <= 100)].dropna()
        df_gauge.drop_duplicates(subset=['datetime'], inplace=True)
        rain_gauge_data[gauge_id] = df_gauge
        print(f"Loaded and filtered rain gauge data for {gauge_id} - {count+1}/{len(all_gauge_ids)}")
        count += 1

    with open(rain_gauge_pickle, "wb") as f:
        pickle.dump(rain_gauge_data, f)
    print("All rain gauge data loaded and filtered.")


def process_station(row):
    """Process a single station and merge with rain gauge data."""
    station_id = row['station_id']
    gauge_ids = [row['gauge_id_1'], row['gauge_id_2'], row['gauge_id_3']]

    print(f"Processing station: {station_id}")

    station_file = os.path.join(stations_folder, f"{station_id}.csv")
    df_station = pd.read_csv(station_file, parse_dates=['Flow_sampling_time'])
    df_station.rename(columns={'Flow_sampling_time': 'datetime'}, inplace=True)
    df_station = df_station[(df_station['Flow_m3_sec'] >= 0) & (df_station['Flow_m3_sec'] <= 1000)]
    df_station = df_station[(df_station['Water_level_m'] >= 0)].dropna()
    df_station.drop_duplicates(subset=['datetime'], inplace=True)
    df_station['datetime'] = pd.to_datetime(df_station['datetime']).dt.tz_localize(None)

    # Merge with rain gauge data
    for i, gauge_id in enumerate(gauge_ids, start=1):
        if gauge_id in rain_gauge_data:
            df_station = df_station.merge(rain_gauge_data[gauge_id], on='datetime', how='left',
                                          suffixes=(None, f'_gauge_{i}'))
            df_station.rename(columns={'Rain': f'Rain_gauge_{i}'}, inplace=True)
        else:
            df_station[f'Rain_gauge_{i}'] = 999  # Set missing rain values to 0

    # Drop any remaining NaNs after merging
    df_station = df_station.dropna()

    # Save cleaned, combined file
    output_file = os.path.join(output_folder, f"{station_id}_combined.csv")
    df_station.to_csv(output_file, index=False)
    print(f"Saved combined file: {output_file}")


# Use multiprocessing to parallelize station processing
"""
if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.map(process_station, [row._asdict() for row in df_mapping.itertuples(index=False)])

    print("All stations processed.")
"""

if __name__ == "__main__":
    for row in df_mapping.itertuples(index=False):
        process_station(row._asdict())
    print("All stations processed.")
