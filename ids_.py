#experiment

import sys, os, subprocess

from copy import deepcopy as copy
import types
import time
import logging
import numpy as np

from scipy import integrate
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['axes.labelsize'] = 'x-small'
mpl.rcParams['xtick.labelsize'] = 'xx-small'
mpl.rcParams['ytick.labelsize'] = 'xx-small'
mpl.rcParams['legend.fontsize'] = 'x-small'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.linewidth'] = 0.5
import torch
from torch import optim, nn

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# - loaded -

df = pd.read_csv('datasets/nsl-kdd/train-flow.csv')
categorical = [
    'protocol',
    'service',
    'flag',
    ]
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
X = normal.copy()

anomalous = df[df['attack_flag'] == 1].copy()
anomalous = anomalous.drop(columns = ['attack_flag'])
anomalous = anomalous.to_numpy(dtype = 'float64')


# - processed -

low = np.quantile(normal, 0.05, axis = 0)

low_valid = normal >= low
low_valid = np.all(low_valid, axis = 1)
valid = low_valid.copy()
data = normal[valid].copy()

scaler = MinMaxScaler(feature_range = (-1, 1))
scaler.fit(data)

data = torch.tensor(
    scaler.transform(data),
    dtype = torch.float32,
    )
