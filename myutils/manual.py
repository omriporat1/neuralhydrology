import pandas as pd
from general import *

def clean_series(df: pd.Series):

    df = df.copy()

    df = df.dropna(axis=0)

    df['date'] = pd.to_datetime(df['date'])

    df['DateDiff'] = df['date'].diff().dt.days

    # Assign a unique group/batch ID to each series of consecutive days
    df['Batch'] = (df['DateDiff'] > 1).cumsum()

    df = df.drop('DateDiff', axis=1)
    df = df.drop('date', axis=1)

    return df


def series_to_samples(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    dataset = dataset.drop('date', axis=1)

    for i in range(len(dataset)-lookback):
        if pd.isna(dataset.iloc[i + lookback, -1]):
            continue
        feature = dataset[i : i + lookback]
        target = dataset.iloc[i + lookback, -1]
        X.append(feature)
        y.append(target)

    return np.array(X), np.array(y)

def build_dataset(name, lookback = 4):
    src_path = 'data/Caravan/timeseries/csv'

    # files, _ = get_files(os.path.join(src_path, name), extension='.csv')

    # X, y = np.ndarray((0, lookback, in_features)), np.ndarray((0, lookback, in_features))
    # for file in files:
    #     df = pd.read_csv(file)
    #     X, y = series_to_samples(df, lookback)
    #     print(X.shape, y.shape)

    # if data

    
def get_model(config):

    kwargs = {}
    kwargs.update(config['Model'])

    kwargs.pop('name')

    if config['Model']['name'] == 'LSTM':
        return LSTM(**kwargs)
    
def load_Data(config):
    datasets = config['Datasets']
    print(datasets)

    for key, value in datasets.items():
        if value:
            build_dataset(key)
