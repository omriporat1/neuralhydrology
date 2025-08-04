import os
import pandas as pd

# Define the input and output directories
input_dir = r'C:\PhD\Data\IMS\Data_by_station\5_stations_formatted_sample'
output_dir = r'C:\PhD\Data\IMS\Data_by_station\5_stations_filtered_2022_2023'
os.makedirs(output_dir, exist_ok=True)

# Loop through each file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        
        # Read the file into a DataFrame
        data = pd.read_csv(file_path)
        
        # Filter the data for the years 2022 and 2023 without converting the datetime column
        filtered_data = data[data['datetime'].str.contains('2022|2023')]
        
        # Save the filtered data to the output directory
        output_path = os.path.join(output_dir, file_name)
        filtered_data.to_csv(output_path, index=False)
