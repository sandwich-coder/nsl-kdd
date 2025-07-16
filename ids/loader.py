from environment import *
import pandas as pd

def _make_nsl_kdd(is_normal):
    df = pd.read_csv('datasets/nsl-kdd/flow.csv')
    df = df.drop(columns = ['attack_name', 'attack_step', 'unknown'])

    categorical = [
        'protocol',
        'service',
        'flag',
        ]
    df = pd.get_dummies(df, columns = categorical)

    normal = df[df['attack_flag'] == 0].copy()
    normal = normal.drop(columns = ['attack_flag'])
    normal = normal.to_numpy(dtype = 'float64')
    
    anomalous = df[df['attack_flag'] == 1].copy()
    anomalous = anomalous.drop(columns = ['attack_flag'])
    anomalous = anomalous.to_numpy(dtype = 'float64')


    # - refined -

    high = np.quantile(normal, 0.95, axis = 0)
    high_valid = normal <= high
    high_valid = np.all(high_valid, axis = 1)
    valid = high_valid.copy()
    normal = normal[valid]
    normal = (normal - normal.min()) / (normal.max() - normal.min())

    high = np.quantile(anomalous, 0.95, axis = 0)
    high_valid = anomalous <= high
    high_valid = np.all(high_valid, axis = 1)
    valid = high_valid.copy()
    anomalous = anomalous[valid]
    anomalous = (anomalous - anomalous.min()) / (anomalous.max() - anomalous.min())

    if is_normal:
        return normal
    else:
        return anomalous


class Loader:
    """
    reference = [
        _make_nsl_kdd,
        ]
    """
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, normal = True):
        if not isinstance(name, str):
            raise TypeError('Name of the dataset should be a string.')
        if not isinstance(normal, bool):
            raise TypeError('\'normal\' should be boolean.')

        if name == 'nsl-kdd':
            return _make_nsl_kdd(normal)
