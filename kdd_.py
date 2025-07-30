#experiment

import inspect, code
from copy import deepcopy as copy
import types
import time
import logging
logging.basicConfig(level = 'INFO')
logger = logging.getLogger('experiment')
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

from scipy import stats
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import seaborn as sb
from tqdm import tqdm


# - loaded -

dataset = 'nsl-kdd'

df = pd.read_csv('datasets/nsl-kdd/train.csv', header = 0, index_col = None)
df_ = pd.read_csv('datasets/nsl-kdd/test.csv', header = 0, index_col = None)
categorical = [
    'protocol_type',
    'service',
    'flag',
    ]

for l in categorical:
    df[l] = df[l].astype('category')

#one-hot
merged = pd.concat([df, df_], axis = 'index')
merged = pd.get_dummies(merged, columns = categorical)
df_hot = merged.iloc[:df.shape[0], :]
df_hot_ = merged.iloc[df.shape[0]:, :]
logger.info('The categorical features are one-hot encoded.')

#normal-train
df_normal = df_hot[df_hot['attack'] == 'normal'].copy()
df_normal.drop(columns = ['attack'], inplace = True)

#normal-test
df_normal_ = df_hot_[df_hot_['attack'] == 'normal'].copy()
df_normal_.drop(columns = ['attack'], inplace = True)

#anomalous-train
df_anomalous = df_hot[df_hot['attack'] != 'normal'].copy()
df_anomalous.drop(columns = ['attack'], inplace = True)

#anomalous-test
df_anomalous_ = df_hot_[df_hot_['attack'] != 'normal'].copy()
df_anomalous_.drop(columns = ['attack'], inplace = True)
del df_hot, df_hot_, categorical


# - prepared -

#to arrays
normal = df_normal.to_numpy(dtype = 'float64', copy = True)
normal_ = df_normal_.to_numpy(dtype = 'float64', copy = True)
anomalous = df_anomalous.to_numpy(dtype = 'float64', copy = True)
anomalous_ = df_anomalous_.to_numpy(dtype = 'float64', copy = True)
del df_normal, df_normal_, df_anomalous, df_anomalous_

#training set
X = normal.copy()
X_ = normal_.copy()


# - model -

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Sequential(nn.Linear(122, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, 10), nn.Sigmoid()),
            )
        self._decoder = nn.Sequential(
            nn.Sequential(nn.Linear(10, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, 122), nn.Sigmoid()),
            )

        with torch.no_grad():
            nn.init.xavier_uniform_(self._encoder[-1][0].weight)
            nn.init.xavier_uniform_(self._decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, data):

        output = self._encoder(data)
        output = self._decoder(output)

        return output


    def process(self, X, train = True):

        if not train:
            pass
        else:
            self._scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
            self._scaler.fit(X)

        data = self._scaler.transform(X)
        data = torch.tensor(data, dtype = torch.float32)

        return data


    def unprocess(self, output):

        _ = output.numpy()
        Y = _.astype('float64')
        Y = self._scaler.inverse_transform(Y)

        return Y


    def flow(self, X):
        self.eval()

        Y = self.process(X, train = False)
        Y = self(Y)
        Y = Y.detach()
        Y = self.unprocess(Y)

        return Y



# - trained -

#device
if not torch.cuda.is_available():
    device = torch.device('cpu')
    logger.info('CPU is assigned to \'device\' as fallback.')
else:
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')

#autoencoder
ae = Autoencoder()

#processed
data = ae.process(X)
logger.info('data size: {}'.format(tuple(data.size())))

#to gpu
data = data.to(device)
ae.to(device)
logger.info('\'device\' is allocated to the data and model.')

#configured
optimizer = optim.AdamW(
    ae.parameters(),
    lr = 0.0001,
    eps = 1e-7,
    )
LossFn = nn.MSELoss
loss_fn = LossFn()

#training
loader = DataLoader(
    data,
    batch_size = 32,
    shuffle = True,
    )
batchloss = []
logger.info(' - Training Start -')
if logger.getEffectiveLevel() > 20:
    pass
else:
    print('Epoch |     Loss')
    print('===== | ========')
for l in range(30):
    ae.train()
    last_epoch = []
    if logger.getEffectiveLevel() > 20:
        iteration = loader
    else:
        iteration = tqdm(loader, leave = False, ncols = 70)
    for ll in iteration:

        out = ae(ll)
        loss = loss_fn(out, ll)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        last_epoch.append(loss.detach())    ###


    last_epoch = torch.stack(last_epoch, dim = 0)
    last_epoch = last_epoch.cpu()
    last_epoch = last_epoch.numpy()
    last_epoch = last_epoch.astype('float64')

    if logger.getEffectiveLevel() > 20:
        pass
    else:
        print(' {epoch:>4} | {epochloss:<8}'.format(
            epoch = l + 1,
            epochloss = last_epoch.mean(axis = 0, dtype = 'float64').round(decimals = 6),
            ))

    batchloss.append(last_epoch)

batchloss = np.concatenate(batchloss, axis = 0)
logger.info(' - Training finished - ')

#bach to cpu
ae.cpu()
del data, optimizer, loss_fn, loader, last_epoch, iteration, out, loss

#descent plot
fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
ax = fig.add_subplot()
ax.set_box_aspect(0.7)
ax.set_title('Losses')
ax.set_xlabel('batch')
ax.set_ylabel('loss')
pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')
plot = ax.plot(
    np.arange(1, len(batchloss)+1, dtype = 'int64'), batchloss,
    marker = 'o', markersize = 0.05,
    linestyle = '--', linewidth = 0.02,
    color = 'slategrey',
    label = 'final: {}'.format(
        batchloss[-1].round(decimals = 4).tolist(),
        ),
    )
ax.legend()
descent = fig
del batchloss, fig, ax, plot


# - threshold -

loss_fn = nn.L1Loss(reduction = 'none')    #different from that for training
normal_data = ae.process(X, train = False)

## Try averaging or medianing the losses for multiple perturbations.
#perturbation
noise = torch.normal(
    mean = torch.median(normal_data),
    std = (torch.quantile(normal_data, 0.75, dim = 0) - torch.quantile(normal_data, 0.25, dim = 0)) / 5,
    )
noise = torch.reshape(noise, [1, noise.size(dim = 0)])
normal_data = normal_data + noise

normal_loss = loss_fn(ae(normal_data).detach(), normal_data)    ###
_ = normal_loss.numpy()
normal_loss = _.astype('float64')
normal_loss = normal_loss.mean(axis = 1, dtype = 'float64')
threshold = np.quantile(normal_loss, 0.99, axis = 0).tolist()
del normal_data, normal_loss


# - anomaly detection (train) -

result = df.copy()
result = result.astype({'attack':'category'}, copy = True)

normal_index = df[df['attack'] == 'normal']
normal_index = normal_index.index
normal_index = normal_index.to_numpy(dtype = 'int64', copy = False)

normal_data = ae.process(normal, train = False)
normal_data = normal_data + noise    #perturbation
normal_loss = loss_fn(ae(normal_data).detach(), normal_data)    ###
_ = normal_loss.numpy()
normal_loss = _.astype('float64')
normal_loss = normal_loss.mean(axis = 1, dtype = 'float64')

anomalous_index = df[df['attack'] != 'normal']
anomalous_index = anomalous_index.index
anomalous_index = anomalous_index.to_numpy(dtype = 'int64', copy = False)

anomalous_data = ae.process(anomalous, train = False)
anomalous_data = anomalous_data + noise    #perturbation
anomalous_loss = loss_fn(ae(anomalous_data).detach(), anomalous_data)    ###
_ = anomalous_loss.numpy()
anomalous_loss = _.astype('float64')
anomalous_loss = anomalous_loss.mean(axis = 1, dtype = 'float64')

result.loc[normal_index, 'detection'] = normal_loss >= threshold
result.loc[anomalous_index, 'detection'] = anomalous_loss >= threshold
del normal_index, normal_data, normal_loss, anomalous_index, anomalous_data, anomalous_loss

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(0.7)
ax.set_title('Detection (train)')
ax.set_ylabel('proportion (%)')
pp.setp(ax.get_xticklabels(), rotation = 60, ha = 'right')

sb.histplot(
    data = result, x = 'attack',
    hue = 'detection',
    stat = 'percent',
    common_norm = True, multiple = 'dodge',
    shrink = 0.8,
    palette = {True:'tab:red', False:'tab:blue'},
    hue_order = [True, False],
    ax = ax,
    )
del fig, ax


# - anomaly detection (test) -

result_ = df_.copy()
result_ = result_.astype({'attack':'category'}, copy = True)

normal_index_ = df_[df_['attack'] == 'normal']
normal_index_ = normal_index_.index
normal_index_ = normal_index_.to_numpy(dtype = 'int64', copy = False)

normal_data_ = ae.process(normal_, train = False)
normal_data_ = normal_data_ + noise    #perturbation
normal_loss_ = loss_fn(ae(normal_data_).detach(), normal_data_)    ###
_ = normal_loss_.numpy()
normal_loss_ = _.astype('float64')
normal_loss_ = normal_loss_.mean(axis = 1, dtype = 'float64')

anomalous_index_ = df_[df_['attack'] != 'normal']
anomalous_index_ = anomalous_index_.index
anomalous_index_ = anomalous_index_.to_numpy(dtype = 'int64', copy = False)

anomalous_data_ = ae.process(anomalous_, train = False)
anomalous_data_ = anomalous_data_ + noise    #perturbation
anomalous_loss_ = loss_fn(ae(anomalous_data_).detach(), anomalous_data_)    ###
_ = anomalous_loss_.numpy()
anomalous_loss_ = _.astype('float64')
anomalous_loss_ = anomalous_loss_.mean(axis = 1, dtype = 'float64')

result_.loc[normal_index_, 'detection'] = normal_loss_ >= threshold
result_.loc[anomalous_index_, 'detection'] = anomalous_loss_ >= threshold
del normal_index_, normal_data_, normal_loss_, anomalous_index_, anomalous_data_, anomalous_loss_

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(0.7)
ax.set_title('Detection (test)')
ax.set_ylabel('proportion (%)')
pp.setp(ax.get_xticklabels(), rotation = 60, ha = 'right')

sb.histplot(
    data = result_, x = 'attack',
    hue = 'detection',
    stat = 'percent',
    common_norm = True, multiple = 'dodge',
    shrink = 0.8,
    palette = {True:'tab:red', False:'tab:blue'}, hue_order = [True, False],
    ax = ax,
    )
del fig, ax
