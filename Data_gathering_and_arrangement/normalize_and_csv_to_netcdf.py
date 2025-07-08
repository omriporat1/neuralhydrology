import os
import pandas as pd
import xarray as xr
import pickle
import yaml
import numpy as np

# Define the paths
input_folder = f"C:/PhD/Data/Caravan/timeseries/csv/archive/il_before_2024_12_24"
attribute_file = f"C:/PhD/Data/Caravan/attributes/il/attributes_other_il.csv"
output_folder_csv = f"C:/PhD/Data/Caravan/timeseries/csv/il"
output_folder_netcdf = f"C:/PhD/Data/Caravan/timeseries/netcdf/il"
configuration_file = r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\LSTM_shared_RainDis2Dis\Feature_normalization\Best_HPC_minmax_norm.yml"

# Load periods from the YAML file
with open(configuration_file, "r") as file:
    periods = yaml.safe_load(file)

# Extract start and end dates
train_start_dates = periods["train_start_date"]
train_end_dates = periods["train_end_date"]

# Combine start and end dates into a list of tuples
date_ranges = list(zip(train_start_dates, train_end_dates))

# Ensure the output folder exists
os.makedirs(output_folder_netcdf, exist_ok=True)
os.makedirs(output_folder_csv, exist_ok=True)

# create dict to store normalization values - min and max for each feature of each basin:
norm_dict = {}


# Loop through each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        csv_path = os.path.join(input_folder, file_name)
        # Extract the numeric ID (XXXX) from the file name
        numeric_id = file_name.split('_')[0]  # Assumes the format is XXXX_combined.csv
        basin_id = 'il_' + numeric_id

        df = pd.read_csv(csv_path, na_values=['', ' '], parse_dates=['date'], dayfirst=True)

        # Convert the date column to datetime and set it as the index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # extract basin area from attribute_file:
        attribute_df = pd.read_csv(attribute_file)
        basin_area = attribute_df.loc[attribute_df['gauge_id'] == basin_id, 'area'].values[0]

        norm_dict[basin_id] = {
            "basin_area": basin_area,
            "features": {}
        }

        training_df = pd.DataFrame()  # Empty DataFrame to store the unified period

        for start_date, end_date in date_ranges:
            # Convert dates to datetime objects
            start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
            end_date = pd.to_datetime(end_date, format='%d/%m/%Y')

            # Filter the DataFrame for the current date range
            df_filtered = df.loc[start_date:end_date]

            # Skip if the filtered DataFrame is empty
            if df_filtered.empty:
                print(f"No data available for basin {basin_id} in the date range {start_date} to {end_date}.")
                continue

            # Append the filtered DataFrame to the combined DataFrame
            training_df = pd.concat([training_df, df_filtered])


        # normalize each of the feature between [0,1]:
        for feature in df.columns:
            if feature == 'date' or feature == 'code':
                continue

            min_val = training_df[feature].min()
            max_val = training_df[feature].max()
            std_val = training_df[feature].std()
            mean_val = training_df[feature].mean()

            # add a new column with the normalized values named like the original + _minmax_norm:
            if max_val - min_val == 0:
                df[f"{feature}_minmax_norm"] = 0
            else:
                df[f"{feature}_minmax_norm"] = (df[feature] - min_val) / (max_val - min_val)

            if std_val == 0:
                df[f"{feature}_zscore_norm"] = mean_val
            else:
                df[f"{feature}_zscore_norm"] = (df[feature] - mean_val) / std_val

            # Save min and max to the dictionary under the basin ID
            norm_dict[basin_id]["features"][feature] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val
            }
            if feature == 'Flow_m3_sec':
                df['unit_discharge_m3_sec_km'] = df[feature] / basin_area
        df['all_ones'] = 1
        df['random_0_1'] = np.random.rand(len(df))


        # Convert the DataFrame to an xarray Dataset
        ds = xr.Dataset.from_dataframe(df)



        # Define the output path for the netCDF file
        netcdf_file_name = f"il_{numeric_id}.nc"
        netcdf_path = os.path.join(output_folder_netcdf, netcdf_file_name)
        # Save the xarray Dataset as a netCDF file
        ds.to_netcdf(netcdf_path)

        # Define the output path for the CSV file
        csv_file_name = f"il_{numeric_id}.csv"
        csv_path = os.path.join(output_folder_csv, csv_file_name)
        # Save the DataFrame as a CSV file
        df.to_csv(csv_path)

        print(f"Converted {file_name} to {netcdf_file_name}")

# save the normalization dictionary to a file:
norm_dict_path = os.path.join(output_folder_csv, "normalization_dict.csv")
norm_df = pd.DataFrame(norm_dict)
norm_df.to_csv(norm_dict_path)

# save the normalization dictionary to a pickle file:
norm_dict_path = os.path.join(output_folder_netcdf, "normalization_dict.pkl")
with open(norm_dict_path, 'wb') as f:
    pickle.dump(norm_dict, f)


print(f"Normalization dictionary saved to {norm_dict_path}")
