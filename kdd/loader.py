from common import *
logger = logging.getLogger(name = __name__)
from sklearn.model_selection import train_test_split

def _make_nsl_kdd(attack, resplit, raw):
    df = pd.read_csv('datasets/nsl-kdd/train.csv', header = 0, index_col = None)
    df_ = pd.read_csv('datasets/nsl-kdd/test.csv', header = 0, index_col = None)
    categorical = [
        'protocol_type',
        'service',
        'flag',
        ]

    # The below code doesn't work. There seems some property issue.
    # merged.loc[:, categorical] = merged.loc[:, categorical].astype('category', copy = False)
    # The 'loc' property should be used with care for assignment operations.
    # The 'copy=False' option is virtually meaningless, since copy is required except very special cases.
    # The real intent of the option is to avoid copy when the input dtype and the target dtype are the same.
    #merge, total categories, and split
    merged = pd.concat([df, df_], axis = 'index')
    for l in categorical:
        merged[l] = merged[l].astype('category', copy = False)
    if resplit:
        df, df_ = train_test_split(
            merged,
            test_size = 0.2,
            shuffle = True,
            stratify = merged['attack'],
            )
    else:
        df = merged.iloc[:df.shape[0], :]
        df_ = merged.iloc[df.shape[0]:, :]


    if raw:

        normal = df[df['attack'] == 'normal'].copy()
        normal_ = df_[df_['attack'] == 'normal'].copy()

        anomalous = df[df['attack'] != 'normal'].copy()
        anomalous_ = df_[df_['attack'] != 'normal'].copy()

        if attack:
            return anomalous, anomalous_
        else:
            return normal, normal_



    #one-hot
    df = pd.get_dummies(df, columns = categorical, dtype = 'int64')
    df_ = pd.get_dummies(df_, columns = categorical, dtype = 'int64')
    logger.info('The categorical features are one-hot-encoded.')

    normal = df[df['attack'] == 'normal'].copy()    # Without copy the pandas prints a warning, not because it returns a view but it MIGHT return a view. I don't know what this means.
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
        return anomalous, anomalous_
    else:
        return normal, normal_




class Loader:
    """
    reference = [
        '_make_nsl_kdd',
        ]
    """
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, attack = False, resplit = False, raw = False):
        if not isinstance(name, str):
            raise TypeError('The name of dataset should be a string.')
        if not isinstance(attack, bool):
            raise TypeError('\'attack\' should be boolean.')
        if not isinstance(resplit, bool):
            raise TypeError('\'resplit\' should be boolean.')
        if not isinstance(raw, bool):
            raise TypeError('\'raw\' should be boolean.')

        if name == 'nsl-kdd':
            return _make_nsl_kdd(attack, resplit, raw)
