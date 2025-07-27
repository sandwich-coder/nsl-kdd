#experiment

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
from rich.logging import RichHandler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
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
df = merged.iloc[:df.shape[0], :]
df_ = merged.iloc[df.shape[0]:, :]
logger.info('The categorical features are one-hot encoded.')

#normal-train
normal = df[df['attack'] == 'normal'].copy()
normal.drop(columns = ['attack'], inplace = True)
normal = normal.to_numpy(dtype = 'float64', copy = False)

#normal-test
normal_ = df_[df_['attack'] == 'normal'].copy()
normal_.drop(columns = ['attack'], inplace = True)
normal_ = normal_.to_numpy(dtype = 'float64', copy = False)

#anomalous-train
anomalous = df[df['attack'] != 'normal'].copy()
anomalous.drop(columns = ['attack'], inplace = True)
anomalous = anomalous.to_numpy(dtype = 'float64', copy = False)

#anomalous-test
anomalous_ = df_[df_['attack'] != 'normal'].copy()
anomalous_.drop(columns = ['attack'], inplace = True)
anomalous_ = anomalous_.to_numpy(dtype = 'float64', copy = False)
del df, df_, categorical, merged


# - prepared -

#train
mixed = np.concatenate([normal, anomalous], axis = 0)
truth = np.ones(mixed.shape[0], dtype = 'int64')
truth[:len(normal)] = 0
truth = truth.astype('bool')

#test
mixed_ = np.concatenate([normal_, anomalous_], axis = 0)
truth_ = np.ones(mixed_.shape[0], dtype = 'int64')
truth_[:len(normal_)] = 0
truth_ = truth_.astype('bool')


#training set
X = normal.copy()
X_ = normal_.copy()


# - model -

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Sequential(nn.Linear(122, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, 5), nn.Sigmoid()),
            )
        self._decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 40), nn.GELU()),
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
        iteration = tqdm(loader, leave = False)
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
    marker = 'o', markersize = 0.3,
    linestyle = '--', linewidth = 0.05,
    color = 'slategrey',
    label = 'final: {}'.format(
        batchloss[-1].round(decimals = 4).tolist(),
        ),
    )
ax.legend()
descent = fig
del batchloss, fig, ax, plot