import sys, os, subprocess

#python check
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('This module is intended to be run on Python 3.12.')


#console inputs
from commandline import (
    LATENT,
    RESPLIT,
    Q_THRESHOLD,
    LOGGING_LEVEL,
    )

#packages
from common import *
logger = logging.getLogger(name = 'main')
logging.basicConfig(level = LOGGING_LEVEL)

from loader import Loader
from models import Autoencoder
from utils import Sampler


# - gpu driver check -

sh = 'nvidia-smi'
sh_ = subprocess.run('which ' + sh, shell = True, capture_output = True, text = True).stdout
if sh_ == '':
    supported_cuda = None
else:
    sh_ = subprocess.run(
        sh,
        shell = True, capture_output = True, text = True,
        ).stdout
    supported_cuda = sh_.split()
    supported_cuda = supported_cuda[supported_cuda.index('CUDA') + 2]

if None in [torch.version.cuda, supported_cuda]:
    if torch.version.cuda is None:
        logger.warning('The installed pytorch is not built with CUDA. Install a CUDA-enabled.')
    if supported_cuda is None:
        logger.warning('The nvidia driver does not exist.')
elif float(supported_cuda) < float(torch.version.cuda):
    logger.warning('The supported CUDA is lower than installed. Upgrade the driver.')
else:
    logger.info('- Nvidia driver checked -')


dataset = 'nsl-kdd' #For later extensions with multiple datasets

#loaded
loader = Loader()
normal, normal_ = loader.load(dataset, attack = False, resplit = RESPLIT)
anomalous, anomalous_ = loader.load(dataset, attack = True, resplit = RESPLIT)

#for traditional ML
normal_df, normal_df_ = loader.load(dataset, attack = False, resplit = RESPLIT, raw = True)
anomalous_df, anomalous_df_ = loader.load(dataset, attack = True, resplit = RESPLIT, raw = True)


# - prepared -

sampler = Sampler()

mix_ = np.concatenate([
    normal_,
    anomalous_,
    ], axis = 0)
truth_ = np.ones(len(mix_), dtype = 'int64')
truth_[:len(normal_)] = 0
truth_ = truth_.astype('bool')

#training set
X = normal.copy()

#model
ae = Autoencoder()


# - training -

#compiled
ae.compile(LossAD = nn.L1Loss) ## Figure out why the result is better when the detection loss is MAE, different from the training loss, MSE.

#trained
ae.fit(X, latent = LATENT, q_threshold = Q_THRESHOLD)

#detection
print('\n\n --- Result ---\n')
detection_, roc_curve_, reconstructions_ = ae.detect(mix_, truth_, return_rocplot = True, return_histplot = True)

#saved
os.makedirs('figures', exist_ok = True)
roc_curve_.savefig('figures/roc_curve.png', dpi = 600)
reconstructions_.savefig('figures/reconstructions.png', dpi = 600)
