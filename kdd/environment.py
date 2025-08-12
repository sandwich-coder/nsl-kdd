from copy import deepcopy as copy
import inspect, code
import types
import time
import logging
import numpy as np
from scipy import linalg as la

import pandas as pd
import torch
from torch import optim, nn
import xgboost as xgb
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
from rich.console import Console
from rich.table import Table
