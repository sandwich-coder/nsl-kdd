#experiment

import sys, os, subprocess

from copy import deepcopy as copy
import types
import time
import logging
logger = logging.getLogger(name = 'experiment')
logging.basicConfig(level = 'INFO')
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader


#gpu check
if not torch.cuda.is_available():
    device = torch.device('cpu')
    logger.info('CPU is assigned to \'device\' as fallback.')
else:
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')


# - loaded -

#raw
df1 = pd.read_csv('datasets/nsl-kdd/train-flow.csv')
df2 = pd.read_csv('datasets/nsl-kdd/test-flow.csv')
df = pd.concat([df1, df2], axis = 'index')
df = df.drop(columns = ['attack_name', 'attack_step', 'unknown'])
del df1, df2

#one-hot
categorical = [
    'protocol',
    'service',
    'flag',
    ]
df = pd.get_dummies(df, columns = categorical)
del categorical

normal = df[df['attack_flag'] == 0].copy()
normal = normal.drop(columns = ['attack_flag'])
normal = normal.to_numpy(dtype = 'float64')
normal = (normal - normal.min()) / (normal.max() - normal.min())

anomalous = df[df['attack_flag'] == 1].copy()
anomalous = anomalous.drop(columns = ['attack_flag'])
anomalous = anomalous.to_numpy(dtype = 'float64')
anomalous = (anomalous - anomalous.min()) / (anomalous.max() - anomalous.min())


# - refined -

high = np.quantile(normal, 0.95, axis = 0)
high_valid = normal <= high
high_valid = np.all(high_valid, axis = 1)
valid = high_valid.copy()
normal = normal[valid]
normal = (normal - np.float64(0.5)) * np.float64(2)

high = np.quantile(anomalous, 0.95, axis = 0)
high_valid = anomalous <= high
high_valid = np.all(high_valid, axis = 1)
valid = high_valid.copy()
anomalous = anomalous[valid]
anomalous = (anomalous - np.float64(0.5)) * np.float64(2)
del high, high_valid, valid

normal, normal_ = train_test_split(normal, test_size = 0.2)
anomalous, anomalous_ = train_test_split(anomalous, test_size = 0.2)

contaminated = np.concatenate([normal, anomalous], axis = 0)
truth = np.zeros([len(contaminated)], dtype = 'int64')
truth[len(normal):] = 1
truth = truth.astype('bool')

contaminated_ = np.concatenate([normal_, anomalous_], axis = 0)
truth_ = np.zeros([len(contaminated_)], dtype = 'int64')
truth_[len(normal_):] = 1
truth_ = truth_.astype('bool')

#train set
X = normal.copy()
X_ = normal_.copy()


# - model -

class Autoencoder(nn.Module):
    def __init__(self):

        #initialized
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.scaler = None

        encoder = nn.Sequential(
            nn.Sequential(nn.Linear(122, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, 5), nn.Tanh()),
            )
        decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, 122), nn.Tanh()),
            )

        with torch.no_grad():
            nn.init.xavier_uniform_(encoder[-1][0].weight)
            nn.init.xavier_uniform_(decoder[-1][0].weight)

        #pushed
        self.encoder = encoder
        self.decoder = decoder

    def __repr__(self):
        return 'autoencoder'

    def forward(self, data):
        data = torch.clone(data)

        output = self.encoder(data)
        output = self.decoder(output)
        
        return output


    def process(self, X, train = True):
        X = X.copy()
        scaler = self.scaler    #pulled

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)

        processed = scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)

        #pushed
        self.scaler = scaler

        return processed

    def unprocess(self, processed):
        processed = torch.clone(processed)
        scaler = self.scaler    #pulled

        _ = processed.numpy()
        unprocessed = _.astype('float64')
        unprocessed = scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        X = X.copy()
        
        self.eval()

        Y = self.process(X, train = False)
        Y = self(Y)
        Y = Y.detach()    ###
        Y = self.unprocess(Y)
        
        return Y




# - training -

ae = Autoencoder()

data = ae.process(X)

#to gpu
data = data.to(device)
ae.to(device)
logger.info('\'device\' is allocated to \'data\' and \'model\'.')

optimizer = optim.AdamW(
    ae.parameters(),
    lr = 0.0001,
    eps = 1e-7,
    )
loss_fn = nn.MSELoss()

loader = DataLoader(
    data,
    batch_size = 32,
    shuffle = True,
    )
batchloss = []
logger.info('Training begins.')
for l in range(10):
    ae.train()
    last_epoch = []
    if logger.getEffectiveLevel() > 20:
        iteration = loader
    else:
        iteration = tqdm(loader, leave = False, ncols = 70)
    for t in iteration:

        output = ae(t)
        loss = loss_fn(output, t)

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
        print('Eposh {epoch:>3} | loss: {epochloss:<7}'.format(
            epoch = l + 1,
            epochloss = last_epoch.mean(axis = 0, dtype = 'float64').round(decimals = 6),
            ))

    batchloss.append(last_epoch)

batchloss = np.concatenate(batchloss, axis = 0)
logger.info(' - Training finished - ')
del iteration

#back to cpu
ae.cpu()
data = data.cpu()


# - result -

Y = ae.flow(X)
Y_ = ae.flow(X_)

error = (ae.flow(normal) - normal) ** 2
error = error.sum(axis = 1, dtype = 'float64')
normal_error = np.sqrt(error, dtype = 'float64')

error = (ae.flow(anomalous) - anomalous) ** 2
error = error.sum(axis = 1, dtype = 'float64')
anomalous_error = np.sqrt(error, dtype = 'float64')

error = (ae.flow(normal_) - normal_) ** 2
error = error.sum(axis = 1, dtype = 'float64')
normal_error_ = np.sqrt(error, dtype = 'float64')

error = (ae.flow(anomalous_) - anomalous_) ** 2
error = error.sum(axis = 1, dtype = 'float64')
anomalous_error_ = np.sqrt(error, dtype = 'float64')
del error

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(1)
ax.set_title('Reconstruction Errors')
ax.set_xticks([])
pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

temp = 25 / len(normal_error) ** 0.5
if temp > 1:
    temp = 1
plot_1 = ax.plot(
    np.linspace(0, 1, num = len(normal_error), dtype = 'float64'), normal_error,
    marker = 'o', markersize = 3 * temp,
    linestyle = '',
    alpha = 0.8,
    color = 'tab:blue',
    label = 'normal',
    )

temp = 25 / len(anomalous_error) ** 0.5
if temp > 1:
    temp = 1
plot_2 = ax.plot(
    np.linspace(0, 1, num = len(anomalous_error), dtype = 'float64'), anomalous_error,
    marker = 'o', markersize = 3 * temp,
    linestyle = '',
    alpha = 0.8,
    color = 'tab:red',
    label = 'anomalous',
    )

ax.legend()
errors = fig
del fig, ax, plot_1, plot_2

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(1)
ax.set_title('Reconstruction Errors (test)')
ax.set_xticks([])
pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

temp = 25 / len(normal_error_) ** 0.5
if temp > 1:
    temp = 1
plot_1 = ax.plot(
    np.linspace(0, 1, num = len(normal_error_), dtype = 'float64'), normal_error_,
    marker = 'o', markersize = 3 * temp,
    linestyle = '',
    alpha = 0.8,
    color = 'tab:blue',
    label = 'normal',
    )

temp = 25 / len(anomalous_error_) ** 0.5
if temp > 1:
    temp = 1
plot_2 = ax.plot(
    np.linspace(0, 1, num = len(anomalous_error_), dtype = 'float64'), anomalous_error_,
    marker = 'o', markersize = 3 * temp,
    linestyle = '',
    alpha = 0.8,
    color = 'tab:red',
    label = 'anomalous',
    )

ax.legend()
errors_ = fig
del fig, ax, plot_1, plot_2


# - test -

threshold = np.quantile(normal_error, 0.99, axis = 0)

error = (ae.flow(contaminated) - contaminated) ** 2
error = error.sum(axis = 1, dtype = 'float64')
error = np.sqrt(error, dtype = 'float64')
prediction = error >= threshold

error = (ae.flow(contaminated_) - contaminated_) ** 2
error = error.sum(axis = 1, dtype = 'float64')
error = np.sqrt(error, dtype = 'float64')
prediction_ = error >= threshold
del error


print('\n')
print(' --- Train --- ')
print('')
print('      Precision: {precision}'.format(
    precision = round(precision_score(truth, prediction), ndigits = 3),
    ))
print('         Recall: {recall}'.format(
    recall = round(recall_score(truth, prediction), ndigits = 3),
    ))
print('             F1: {f1}'.format(
    f1 = round(f1_score(truth, prediction), ndigits = 3),
    ))

print('\n')
print(' --- Test --- ')
print('      Precision: {precision}'.format(
    precision = round(precision_score(truth_, prediction_), ndigits = 3),
    ))
print('         Recall: {recall}'.format(
    recall = round(recall_score(truth_, prediction_), ndigits = 3),
    ))
print('             F1: {f1}'.format(
    f1 = round(f1_score(truth_, prediction_), ndigits = 3),
    ))
