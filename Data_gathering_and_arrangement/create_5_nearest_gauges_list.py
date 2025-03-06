import pandas as pd


def main():
    # Load the CSV file
    file_path = r'C:\PhD\Data\IHS_IMS_wide\5_nearest_gauges_to_stations_wide_manual.csv'
    caravan_file_path = r"C:\PhD\Data\Caravan\attributes\il\attributes_other_il.csv"
    df = pd.read_csv(file_path)
    caravan_df = pd.read_csv(caravan_file_path)

    # Initialize a list to hold the transformed data
    rows = []

    # Iterate through each unique station
    for station_id, group in df.groupby('station_id'):
        # Initialize a dictionary to hold data for the current station
        # from caravan_df get the station name from column 'gauge_name' of station "gauge_id" = station_id
        station_name = caravan_df[caravan_df['gauge_id'] == station_id]['gauge_name'].iloc[0]
        row_data = {
            'station_id': station_id,
            'station_name': station_name,
            'station_x': group['station_x'].iloc[0],
            'station_y': group['station_y'].iloc[0]
        }

        # Iterate through the gauges associated with the current station
        for i, gauge in enumerate(group.itertuples(), start=1):
            row_data[f'gauge_id_{i}'] = gauge.gauge_id
            row_data[f'gauge_name_{i}'] = gauge.gauge_name
            row_data[f'distance_{i}'] = gauge.distance
            row_data[f'gauge_x_{i}'] = gauge.gauge_x
            row_data[f'gauge_y_{i}'] = gauge.gauge_y

        # Append the row data to the list
        rows.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    transformed_data = pd.DataFrame(rows)

    # Save the transformed data to a new CSV file
    output_path = r'C:\PhD\Data\IHS_IMS_wide\Stations_With_Gauges_Wide_Combined.csv'
    transformed_data.to_csv(output_path, index=False)

    print(f"File saved to {output_path}")


if __name__ == '__main__':
    main()
