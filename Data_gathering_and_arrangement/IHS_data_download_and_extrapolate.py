# This code reads the Hydrograph files, unites them, translate all Hebrew to English, and saves the combined file.
# Then, it extracts the relevant stations list from the static parameters of caravan, and uses that to extract the
# data of the relevant stations. Next, it extrapolates the data of each station into 10-minute intervals,
# and saves the data in a new folder in separate station files.

import pandas as pd
import os


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
        numeric_cols = resampled.select_dtypes(include=['number']).columns
        non_numeric_cols = resampled.select_dtypes(exclude=['number']).columns
        resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='time')
        resampled[non_numeric_cols] = resampled[non_numeric_cols].ffill()
        resampled[numeric_cols] = resampled[numeric_cols].bfill()
        resampled[non_numeric_cols] = resampled[non_numeric_cols].bfill()

        single_station_data = resampled


        # save the data in a new folder in separate station files
        file_name = os.path.join(output_hydrograph_folder, f'{station_number}.csv')
        single_station_data.to_csv(file_name, index=True)
        print(f"Exported data for station {station_number} to {str(file_name)}")


if __name__ == '__main__':
    main()
