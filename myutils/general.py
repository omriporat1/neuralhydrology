
import os

from pathlib import Path
from datetime import datetime

from neuralhydrology.utils.config import Config


def get_files(directory, extension='.csv'):
    """Walks through a directory tree and returns a list of all csv file's paths
    
    Args:
        directory: The root directory to start the search from
    """
    files = []
    dirs = []

    # Walk through all subdirectories of the given directory
    for root, dirs_in_dir, files_in_dir in os.walk(directory):
        for file in files_in_dir: # Check if the file is a csv file
            if file.endswith(extension):
                # If yes, add the full path to the list of files
                files.append(os.path.join(root, file))
        dirs.extend(dirs_in_dir)
        
    
    return files, dirs

def get_free_name(path, name: str, extension='.yaml'):

    file_name = f'{name}{extension}'

    free_name = file_name
    i = 1
    while os.path.exists(os.path.join(path, free_name)):
        free_name = f'{name}_{i}{extension}'
        i += 1
    
    return free_name


def map_state_dict(target_state_dict, source_state_dict):
    mapped_state_dict = {'src_to_tgt': {}, 'tgt_to_src': {}}
    for key, value in target_state_dict.items():
        names = key.split('.')
        matched = True
        for name in names:
            if not matched:
                break
            matched = False
            for skey, svalue in source_state_dict.items():
                tkeys = skey.split('.')
                if name in tkeys:
                    matched = True
                    break

        if matched and source_state_dict[skey].shape == value.shape:
            mapped_state_dict['src_to_tgt'][skey] = key
            mapped_state_dict['tgt_to_src'][key] = skey
    return mapped_state_dict

def get_last_run(config: Config):
    if config.run_dir is None:
        config.run_dir = Path('runs')
        
    main_directory = str(config.run_dir / config.experiment_name)
    # List all directories in the main directory
    directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    # Initialize variables to store the latest datetime and corresponding folder name
    latest_datetime = datetime.min
    latest_folder = None

    # Loop through each directory and parse the datetime from its name
    for directory in directories:
        try:
            # Extract the datetime part of the folder name (assuming the format you described)
            # Adjust the slicing if the experiment_name contains underscores
            parts = directory.split('_')
            date_part = parts[-2]  # This is 'DDMM'
            time_part = parts[-1]  # This is 'HHMMSS'
            day = int(date_part[:2])
            month = int(date_part[2:4])
            year = int(date_part[4:8])
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            # Create a datetime object from the extracted parts
            folder_datetime = datetime(year, month, day, hour, minute, second)
            
            # Update the latest folder if this folder's datetime is later
            if folder_datetime > latest_datetime:
                latest_datetime = folder_datetime
                latest_folder = directory

        except ValueError:
            continue  # Skip folders that do not match the expected format

    return latest_folder
