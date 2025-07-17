#experiment

import sys, os, subprocess

from copy import deepcopy as copy
import types
import time
import logging
logger = logging.getLogger(name = 'experiment')
logging.basicConfig(level = 'INFO')
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

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sb


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
df = pd.read_csv('datasets/nsl-kdd/train.csv', header = 0, index_col = None)
df_ = pd.read_csv('datasets/nsl-kdd/test.csv', header = 0, index_col = None)

#one-hot
merged = pd.concat([df, df_], axis = 'index')
categorical = [
    'protocol_type',
    'service',
    'flag',
    ]
merged = pd.get_dummies(merged, columns = categorical)
df = merged.iloc[:df.shape[0], :]
df_ = merged.iloc[df.shape[0]:, :]
del merged


# - separated -

normal = df[df['attack'] == 'normal'].drop(columns = ['attack']).to_numpy(dtype = 'float64')
normal_ = df_[df_['attack'] == 'normal'].drop(columns = ['attack']).to_numpy(dtype = 'float64')

anomalous = df[df['attack'] != 'normal'].drop(columns = ['attack']).to_numpy(dtype = 'float64')
anomalous_ = df_[df_['attack'] != 'normal'].drop(columns = ['attack']).to_numpy(dtype = 'float64')


"""
# - what if (turns out to make visual illusion of the reconstruction losses) -

def qcut(A, lowq, highq):
    A = A.copy()

    low = np.quantile(A, lowq, axis = 0)
    high = np.quantile(A, highq, axis = 0)

    low_valid = A >= low
    high_valid = A <= high

    valid = low_valid & high_valid
    valid = np.all(valid, axis = 1)
    return A[valid]

anomalous = qcut(anomalous, 0, 0.99)
anomalous_ = qcut(anomalous_, 0, 0.99)
"""

#training set
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
            nn.Sequential(nn.Linear(122, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, 13), nn.GELU()),
            nn.Sequential(nn.Linear(13, 4), nn.Sigmoid()),
            )
        decoder = nn.Sequential(
            nn.Sequential(nn.Linear(4, 13), nn.GELU()),
            nn.Sequential(nn.Linear(13, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, 122), nn.Sigmoid()),
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
            scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
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

#model
ae = Autoencoder()

#loss function
LossFn = nn.MSELoss

#processed
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

loader = DataLoader(
    data,
    batch_size = 32,
    shuffle = True,
    )
loss_fn = LossFn()
batchloss = []
logger.info('Training begins.')
for l in range(100):
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
del optimizer, loader, loss_fn, batchloss, last_epoch, iteration, output, loss

#back to cpu
ae.cpu()
data = data.cpu()


# - test -

## Is it necessary to use the same metric to compare the reconstruction losses? Can it make a big difference?
loss_fn = LossFn(reduction = 'none')

normal_data = ae.process(normal, train = False)
normal_data_ = ae.process(normal_, train = False)

anomalous_data = ae.process(anomalous, train = False)
anomalous_data_ = ae.process(anomalous_, train = False)

with torch.no_grad():

    normal_loss = loss_fn(ae(normal_data), normal_data)
    normal_loss = normal_loss.numpy()
    normal_loss = normal_loss.mean(axis = 1, dtype = 'float64')

    normal_loss_ = loss_fn(ae(normal_data_), normal_data_)
    normal_loss_ = normal_loss_.numpy()
    normal_loss_ = normal_loss_.mean(axis = 1, dtype = 'float64')

    anomalous_loss = loss_fn(ae(anomalous_data), anomalous_data)
    anomalous_loss = anomalous_loss.numpy()
    anomalous_loss = anomalous_loss.mean(axis = 1, dtype = 'float64')

    anomalous_loss_ = loss_fn(ae(anomalous_data_), anomalous_data_)
    anomalous_loss_ = anomalous_loss_.numpy()
    anomalous_loss_ = anomalous_loss_.mean(axis = 1, dtype = 'float64')


losses_normal = pd.DataFrame({
    'loss':normal_loss,
    'label':['normal'] * normal_loss.shape[0],
    })
losses_anomalous = pd.DataFrame({
    'loss':anomalous_loss,
    'label':['anomalous'] * anomalous_loss.shape[0],
    })
losses = pd.concat([losses_normal, losses_anomalous], axis = 'index')

losses_normal_ = pd.DataFrame({
    'loss':normal_loss_,
    'label':['normal'] * normal_loss_.shape[0],
    })
losses_anomalous_ = pd.DataFrame({
    'loss':anomalous_loss_,
    'label':['anomalous'] * anomalous_loss_.shape[0],
    })
losses_ = pd.concat([losses_normal_, losses_anomalous_], axis = 'index')
del normal_data, normal_data_, anomalous_data, anomalous_data_, normal_loss, normal_loss_, anomalous_loss, anomalous_loss_, losses_normal, losses_anomalous, losses_normal_, losses_anomalous_

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(0.3)
ax.set_title('Reconstruction Losses (train)')
ax.set_xlabel('loss')
ax.set_ylabel('proportion (%)')
pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

sb.histplot(
    data = losses,
    x = 'loss',
    hue = 'label',
    binwidth = 0.01,
    binrange = [0, 1],
    stat = 'percent', common_norm = False,
    ax = ax,
    )

ax.axvline(
    x = losses[losses['label'] == 'normal']['loss'].quantile(0.9),
    ymin = 0, ymax = 0.9,
    linestyle = '--', color = 'orange',
    label = '0.9q',
    )
ax.axvline(
    x = losses[losses['label'] == 'normal']['loss'].quantile(0.99),
    ymin = 0, ymax = 0.9,
    linestyle = '--', color = 'black',
    label = '0.99q',
    )

ax.legend()
reconstructions = fig
del fig, ax

fig = pp.figure(layout = 'constrained')
ax = fig.add_subplot()
ax.set_box_aspect(0.3)
ax.set_title('Reconstruction Losses (test)')
ax.set_xlabel('loss')
ax.set_ylabel('proportion (%)')
pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

sb.histplot(
    data = losses_,
    x = 'loss',
    hue = 'label',
    binwidth = 0.01,
    binrange = [0, 4],
    stat = 'percent', common_norm = False,
    ax = ax,
    )

ax.axvline(
    x = losses_[losses_['label'] == 'normal']['loss'].quantile(0.9),
    ymin = 0, ymax = 0.9,
    linestyle = '--', color = 'orange',
    label = '0.9q',
    )
ax.axvline(
    x = losses_[losses_['label'] == 'normal']['loss'].quantile(0.99),
    ymin = 0, ymax = 0.9,
    linestyle = '--', color = 'black',
    label = '0.99q',
    )

ax.legend()
reconstructions_ = fig
del fig, ax

os.makedirs('figures', exist_ok = True)
reconstructions.savefig('figures/reconstructions-train.png', dpi = 300)
reconstructions_.savefig('figures/reconstructions-test.png', dpi = 300)


# - detections -

threshold = losses[losses['label'] == 'normal']['loss'].quantile(0.99)

result = losses.set_axis(['truth', 'prediction'], axis = 'columns', copy = True)
loss = result['truth'].to_numpy(dtype = 'float64', copy = True)
result['truth'] = np.where(loss <= threshold, 'normal', 'anomalous')

result_ = losses_.set_axis(['truth', 'prediction'], axis = 'columns', copy = True)
loss = result_['truth'].to_numpy(dtype = 'float64', copy = True)
result_['truth'] = np.where(loss <= threshold, 'normal', 'anomalous')
del loss

detections = result.replace({
    'anomalous':True,
    'normal':False,
    })
detections = detections.astype('bool', copy = False)

detections_ = result_.replace({
    'anomalous':True,
    'normal':False,
    })
detections_ = detections_.astype('bool', copy = False)


# - results -

print('\n')
print(' --- Train --- ')
print('')
print(result.value_counts())
print('')
print('      Precision: {precision}'.format(
    precision = round(precision_score(detections['truth'], detections['prediction']), ndigits = 3),
    ))
print('         Recall: {recall}'.format(
    recall = round(recall_score(detections['truth'], detections['prediction']), ndigits = 3),
    ))
print('             F1: {f1}'.format(
    f1 = round(f1_score(detections['truth'], detections['prediction']), ndigits = 3),
    ))

print('\n')
print(' --- Test --- ')
print('')
print(result_.value_counts())
print('')
print('      Precision: {precision}'.format(
    precision = round(precision_score(detections_['truth'], detections_['prediction']), ndigits = 3),
    ))
print('         Recall: {recall}'.format(
    recall = round(recall_score(detections_['truth'], detections_['prediction']), ndigits = 3),
    ))
print('             F1: {f1}'.format(
    f1 = round(f1_score(detections_['truth'], detections_['prediction']), ndigits = 3),
    ))
