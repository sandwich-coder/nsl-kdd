from environment import *
logger = logging.getLogger(name = __name__)

def _make_nsl_kdd(attack, merge):
    df = pd.read_csv('datasets/nsl-kdd/train.csv', header = 0, index_col = None)
    df_ = pd.read_csv('datasets/nsl-kdd/test.csv', header = 0, index_col = None)

    temp = pd.concat([df, df_], axis = 'index')
    categorical = [
        'protocol_type',
        'service',
        'flag',
        ]
    temp = pd.get_dummies(temp, columns = categorical)
    df = temp.iloc[:df.shape[0], :]
    df_ = temp.iloc[df.shape[0]:, :]

    normal = df[df['attack'] == 'normal'].copy()
    normal.drop(columns = ['attack'], inplace = True)
    normal = normal.to_numpy(dtype = 'float64', copy = False)

    normal_ = df_[df_['attack'] == 'normal'].copy()
    normal_.drop(columns = ['attack'], inplace = True)
    normal_ = normal_.to_numpy(dtype = 'float64', copy = False)

    anomalous = df[df['attack'] != 'normal'].copy()
    anomalous.drop(columns = ['attack'], inplace = True)
    anomalous = anomalous.to_numpy(dtype = 'float64', copy = False)

    anomalous_ = df_[df_['attack'] != 'normal'].copy()
    anomalous_.drop(columns = ['attack'], inplace = True)
    anomalous_ = anomalous_.to_numpy(dtype = 'float64', copy = False)

    if attack:
        if merge:
            return np.concatenate([anomalous, anomalous_], axis = 0)
        else:
            return anomalous, anomalous_
    else:
        if merge:
            return np.concatenate([normal, normal_], axis = 0)
        else:
            return normal, normal_




class Loader:
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, attack = False, merge = False):
        if not isinstance(name, str):
            raise TypeError('The name of dataset should be a string.')
        if not isinstance(attack, bool):
            raise TypeError('\'attack\' should be boolean.')
        if not isinstance(merge, bool):
            raise TypeError('Whether to merge should be boolean.')

        if name == 'nsl-kdd':
            return _make_nsl_kdd(attack, merge)
