# -- Experiment Script --

import sys, os, subprocess

from copy import deepcopy as copy
import inspect, code
import types
import time
import logging
logger = logging.getLogger(name = 'experiment')
import numpy as np
from scipy import linalg as la

import pandas as pd
import torch
from torch import optim, nn
import xgboost as xgb
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams.update({
    'figure.figsize':(10, 10),
    'axes.titlesize':'medium',
    'axes.labelsize':'x-small',
    'xtick.labelsize':'xx-small',
    'ytick.labelsize':'xx-small',
    'legend.fontsize':'x-small',
    'lines.markersize':1,
    'lines.linewidth':0.5,
    })
from rich.console import Console
from rich.text import Text
from rich.table import Table

from tqdm import tqdm
import seaborn as sb
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


...
