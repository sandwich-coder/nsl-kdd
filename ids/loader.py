from environment import *
logger = logging.getLogger(name = __name__)

import pandas as pd


class Loader:
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, normal = True, train = True):
        if not isinstance(name, str):
            raise TypeError('The name should be a string.')
        if not isinstance(normal, bool):
            raise TypeError('\'normal\' should be boolean.')
        if not isinstance(train, bool):
            raise TypeError('\'train\' should be boolean.')
        if not train:
            kind = 'test'
        else:
            kind = 'train'

        if name == 'cic17':
            orderless = [
                'attack_flag',
                'attack_name',
                'attack_step',
                'destination port',
                ]
            df = pd.read_csv('datasets/{name}/{kind}-flow.csv'.format(
                name = name,
                kind = kind,
                ))
            if not normal:
                df = df[df['attack_flag'] == 1]
            else:
                df = df[df['attack_flag'] == 0]
            df = df.drop(columns = orderless)
            array = df.to_numpy(dtype = 'float64', copy = True)
            array = (array - array.min()) / (array.max() - array.min())
            array = (array - np.float64(0.5)) * np.float64(2)
        elif name == 'nsl-kdd':
            orderless = [
                'protocol',
                'service',
                'flag',
                'land',
                'logged_in',
                'root_shell',
                'is_host_login',
                'is_guest_login',
                'attack_name',
                'unknown',
                'attack_flag',
                'attack_step',
                ]
            df = pd.read_csv('datasets/{name}/{kind}-flow.csv'.format(
                name = name,
                kind = kind,
                ))
            if not normal:
                df = df[df['attack_flag'] == 1]
            else:
                df = df[df['attack_flag'] == 0]
            df = df.drop(columns = orderless)
            array = df.to_numpy(dtype = 'float64', copy = True)
            array = (array - array.min()) / (array.max() - array.min())
            array = (array - np.float64(0.5)) * np.float64(2)
        elif name == 'ton-iot':
            orderless = [
                'l4_src_port',
                'l4_dst_port',
                'protocol',
                'l7_proto',
                'attack_flag',
                'attack_name',
                'attack_step',
                ]
            df = pd.read_csv('datasets/{name}/{kind}-flow.csv'.format(
                name = name,
                kind = kind,
                ))
            if not normal:
                df = df[df['attack_flag'] == 1]
            else:
                df = df[df['attack_flag'] == 0]
            df = df.drop(columns = orderless)
            array = df.to_numpy(dtype = 'float64', copy = True)
            array = (array - array.min()) / (array.max() - array.min())
            array = (array - np.float64(0.5)) * np.float64(2)

        return array
