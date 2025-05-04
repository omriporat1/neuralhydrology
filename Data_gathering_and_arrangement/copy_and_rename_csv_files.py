import os
import shutil

# Define folder paths
input_folder_path = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_2025-04-08'
output_folder_path = r'C:\PhD\Data\Caravan\timeseries\csv\il'

# Loop through all files in the folder
for filename in os.listdir(input_folder_path):
    if filename.endswith("_combined.csv"):
        # Extract the numeric ID
        numeric_id = filename.split('_')[0]
        # Create the new filename
        new_filename = f"il_{numeric_id}.csv"
        # Define full file paths
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, new_filename)
        # Copy and rename the file
        shutil.copyfile(input_file_path, output_file_path)
        print(f"Copied and renamed {filename} → {new_filename}")

print("All files copied and renamed.")
