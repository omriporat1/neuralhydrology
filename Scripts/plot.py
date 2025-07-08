import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
from pathlib import Path
import matplotlib.pyplot as plt
from utils.configs import get_last_run
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config

def process_tensorboard_folders(folders: list[Path]):
    """
    Process TensorBoard log files from multiple folders.
    Parameters:
        folders (list[str]): List of folders, each containing a TensorBoard log file.
    """
    curves = []
    for folder in folders:
        if not os.path.isdir(folder.absolute()):
            print(f"Invalid folder: {folder.absolute()}")
            continue
        try:
            curve = extract_curves_from_folder(str(folder.absolute()))
            curves.append(curve)
        except Exception as e:
            print(f"Error processing {folder}: {e}")

        if curves:
            plot_learning_curves(curves, title="Learning Curves for All Runs")

def extract_curves_from_folder(folder_path):
    """
    Extract loss curves from a single folder containing a TensorBoard log file.
    Parameters:
        folder_path (str): Path to the folder containing a TensorBoard log file.
    Returns:
        dict: A dictionary containing extracted curves and metadata.
    """
    params = parse_hyperparameters(os.path.basename(folder_path))
    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    train_loss, valid_loss, steps = [], [], []

    if 'train/avg_loss' in event_acc.Tags()['scalars']:
        for e in event_acc.Scalars('train/avg_loss'):
            train_loss.append(e.value)
            steps.append(e.step - 1)

    if 'valid/avg_loss' in event_acc.Tags()['scalars']:
        for e in event_acc.Scalars('valid/avg_loss'):
            valid_loss.append(e.value)

    return {
        **params,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'steps': steps
    }

def parse_hyperparameters(folder_name):
    """
    Dynamically extract hyperparameters from the folder name.
    Parameters:
        folder_name (str): Name of the folder containing the run.
    Returns:
        dict: A dictionary of hyperparameters found in the folder name.
    """
    patterns = {
        "hidden_size": r"hidden_size(\d+)",
        "seq_length": r"seq_length(\d+)",
        "batch_size": r"batch_size(\d+)",
        "learning_rate": r"learning_rate(\d+)",
    }
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        if match:
            if key == "learning_rate":
                matched_value = match.group(1)
                params[key] = float(f"0.{matched_value}")*10
            else:
                # Convert other parameters to integers
                params[key] = int(match.group(1))
    return params

def plot_learning_curves(curves, title="Learning Curves"):
    """
    Plot learning curves for training and validation loss.
    Parameters:
        curves (list[dict]): Extracted curves containing steps, train_loss, and valid_loss.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for curve in curves:
        params_str = ', '.join([f"{k}={v}" for k, v in curve.items() if k not in ['train_loss', 'valid_loss', 'steps']])
        plt.plot(curve['steps'], curve['train_loss'], label=f"Train ({params_str})", linestyle='-')
        plt.plot(curve['steps'], curve['valid_loss'], label=f"Validation ({params_str})", linestyle='--')

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # List of directories containing TensorBoard logs
    log_folders = []
    
    working_conf = Config(Path('RT_flood/check_loss_config.yaml'))
    last_run_path = Path('runs', working_conf.experiment_name, get_last_run(working_conf))
    log_folders.append(last_run_path)
    print(log_folders)
    # log_folders = [ Path('runs' , 'test_validation_loss', 'test_validation_loss_0302_112849')]
    process_tensorboard_folders(log_folders)