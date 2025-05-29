# generate_config_csv.py
import pandas as pd
import numpy as np

N = 60  # Number of random configs

param_space = {
    'batch_size': [256, 512, 1024],
    'hidden_size': [128, 256, 512],
    'learning_rate': [0.0005, 0.001, 0.005],
    'output_dropout': [0.2, 0.4, 0.6],
    'seq_length': [36, 72, 144],
    'statics_embedding': [5, 10, 15, 20]
}

samples = pd.DataFrame({
    k: np.random.choice(v, size=N)
    for k, v in param_space.items()
})

samples.to_csv("random_search_configurations.csv", index_label="job_id")
