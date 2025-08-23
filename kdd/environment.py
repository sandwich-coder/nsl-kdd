from copy import deepcopy as copy
import inspect, code
import types
import time
import logging
import numpy as np
from scipy import linalg as la
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams.update({
    'figure.figsize':[10, 10],
    'figure.edgecolor':'none',
    'figure.facecolor':'none',
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.edgecolor':'none',
    'axes.facecolor':'none',
    'lines.markersize':1,
    'lines.linewidth':0.5,
    'legend.edgecolor':'none',
    'legend.facecolor':'none',
    })

import pandas as pd
import torch
from torch import optim, nn
import xgboost as xgb
from rich.console import Console
from rich.text import Text
from rich.table import Table
