# This code reads the Hydrograph files, unites them, translate all Hebrew to English, and saves the combined file.
# Then, it extracts the relevant stations list from the static parameters of caravan, and uses that to extract the
# data of the relevant stations. Next, it interpolates the data of each station into 10-minute intervals,
# using the CHMIP algorithm, and saves the data in a new folder in separate station files.

import pandas as pd
import os
from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
import random


def main():

    input_hydrograph_folder = 'C:/PhD/Data/IHS/solve_hebrew_2025_02_23/'
    output_hydrograph_folder = 'C:/PhD/Data/IHS/stations_hydrographs/'
    stations_list = r'C:\PhD\Data\Caravan\attributes\il\attributes_caravan_il.csv'
    keep_columns = ['Flow_sampling_time', 'Station_ID', 'Flow_m3_sec', 'Water_level_m', 'Flow_type', 'Data_type', 'Record_type']
    unified_file_exist = False
    start_date = pd.to_datetime("1999-10-01")
    end_date = pd.to_datetime("2023-09-30")

    if not unified_file_exist:

        # read all csv files from the directory
        all_files = os.listdir(input_hydrograph_folder)
        csv_files = [input_hydrograph_folder + file for file in all_files if file.endswith('.csv')]
        print(csv_files)

        # read all csv files into a single dataframe
        df = pd.concat([pd.read_csv(file) for file in csv_files])

        heb_to_eng = {
            'מדודים': 'measured',
            'משוחזרים': 'reconstructed',
            'אחר': 'other',
            'תקין+אחר': 'normal+other',
            'ביוב': 'sewage',
            'גאות': 'flow',
            'תקין': 'normal',
            'התחלת קטע': 'start',
            'נקודה פנימית': 'internal',
            'סיום קטע': 'end',
            'פעילה': 'active',
            'לא פעילה': 'inactive',
            'לא פעילה זמנית': 'temporarily_inactive',
        }
        df.replace(heb_to_eng, inplace=True)
        print(df.head())

        # save the unified dataframe to a new file in the original folder
        output_file_path = os.path.join(input_hydrograph_folder, 'hydrographs_unified.csv')
        df.to_csv(output_file_path, index=False)
        print(f"Exported unified data to {output_file_path}")

    else:
        df = pd.read_csv(os.path.join(input_hydrograph_folder, 'hydrographs_unified.csv'))
        print(df.head())

    # read the stations list from the column 'gauge_id' of the stations list file, remove the 'il_' prefix and
    # convert to list
    stations_df = pd.read_csv(stations_list)
    stations_list = stations_df['gauge_id'].str.replace('il_', '').tolist()
    print(stations_list)

    # extract the data of the relevant stations
    stations_data = {}
    for station_number in stations_list:
        single_station_data = df[df['Station_ID'] == int(station_number)].copy()
        stations_data[station_number] = single_station_data

        # keep only data in columns in the list "keep_columns"
        single_station_data = single_station_data[keep_columns]

        # extrapolate the data of each station into 10-minute intervals
        single_station_data['Flow_sampling_time'] = pd.to_datetime(single_station_data['Flow_sampling_time'], format='%d/%m/%Y %H:%M:%S')
        single_station_data.set_index('Flow_sampling_time', inplace=True)

        single_station_data = single_station_data.loc[(single_station_data.index >= start_date) &
                                                      (single_station_data.index <= end_date)]
        print("Datetime range:", single_station_data.index.min(), "to", single_station_data.index.max())

        single_station_data = single_station_data[~single_station_data.index.duplicated(keep='first')]

        resampled = single_station_data.resample('10min').asfreq()

        # Interpolate numeric columns using PCHIP
        for col in resampled.select_dtypes(include=['number']).columns:
            known_values = single_station_data[col].dropna()
            if len(known_values) >= 2:
                pchip_interpolator = PchipInterpolator(
                    known_values.index.astype(np.int64), known_values.values)
                resampled[col] = pchip_interpolator(resampled.index.astype(np.int64))
            else:
                resampled[col] = known_values.mean()  # fallback for insufficient data

        # Forward-fill non-numeric columns
        resampled[resampled.select_dtypes(exclude=['number']).columns] = \
            resampled.select_dtypes(exclude=['number']).ffill().bfill()

        single_station_data = resampled


        # save the data in a new folder in separate station files
        file_name = os.path.join(output_hydrograph_folder, f'{station_number}.csv')
        single_station_data.to_csv(file_name, index=True)
        print(f"Exported data for station {station_number} to {str(file_name)}")

    # Define the period for plotting
    plot_start = pd.to_datetime('2013-01-01')
    plot_end = pd.to_datetime('2013-01-30')

    # Ensure the output folder for plots exists
    plots_folder = os.path.join(output_hydrograph_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Randomly select 10 stations for plotting
    stations_to_plot = random.sample(stations_list, min(10, len(stations_list)))

    for station_number in stations_to_plot:
        station_df_original = stations_data[station_number].copy()
        station_df_resampled = pd.read_csv(os.path.join(output_hydrograph_folder, f'{station_number}.csv'),
                                           parse_dates=['Flow_sampling_time'],
                                           index_col='Flow_sampling_time')

        # Filter data for plotting period
        original_period = station_df_original[(station_df_original.index >= plot_start) &
                                              (station_df_original.index <= plot_end)]

        resampled_period = station_df_resampled[(station_df_resampled.index >= plot_start) &
                                                (station_df_resampled.index <= plot_end)]

        plt.figure(figsize=(15, 5))
        plt.plot(resampled_period.index, resampled_period['Flow_m3_sec'], label='Interpolated (PCHIP)', color='blue')
        plt.scatter(original_period.index, original_period['Flow_m3_sec'],
                    label='Observed Points', color='red', marker='o', s=20, zorder=5)

        plt.title(f'Hydrograph for Station {station_number} (Jan 1–30, 2013)')
        plt.xlabel('Date')
        plt.ylabel('Flow [m³/sec]')
        plt.grid(True)
        plt.legend()

        plot_file_name = os.path.join(plots_folder, f'station_{station_number}_hydrograph.png')
        plt.tight_layout()
        plt.savefig(plot_file_name, dpi=300)
        plt.close()

        print(f"Plot saved for station {station_number} at {plot_file_name}")


if __name__ == '__main__':
    main()
