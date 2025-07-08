import os
import pandas as pd

orig_files_path = 'C:/PhD/Data/IMS/Data_by_station/Data_by_station_orig/'
output_folder_path = 'C:/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/'


for filename in os.listdir(orig_files_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(orig_files_path, filename)
        df = pd.read_csv(file_path)
        station_id = os.path.splitext(filename)[0]
        df.insert(0, 'Station_ID', station_id)

        # df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S%z').dt.tz_convert('UTC')
        df['datetime'] = df['datetime'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S%z', utc=True))
        df['datetime'] = df['datetime'].dt.strftime('%d/%m/%Y %H:%M%z')

        output_file_path = os.path.join(output_folder_path, f"{station_id}.csv")

        df.to_csv(output_file_path, index=False)
        print(f"file {station_id} saved as {output_folder_path}")
