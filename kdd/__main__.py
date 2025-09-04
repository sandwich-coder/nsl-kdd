import sys, os, subprocess

#python check
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('This module is intended to be run on Python 3.12.')


#console inputs
from commandline import (
    resplit,
    q_threshold,
    logging_level,
    )

#packages
from common import *
logger = logging.getLogger(name = 'main')
logging.basicConfig(level = logging_level)

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
normal, normal_ = loader.load(dataset, attack = False, resplit = resplit)
anomalous = np.concatenate(
    [*loader.load(dataset, attack = True, resplit = resplit)],
    axis = 0,
    )

#for traditional ML
normal_df, normal_df_ = loader.load(dataset, attack = False, resplit = resplit, raw = True)
anomalous_df, anomalous_df_ = loader.load(dataset, attack = True, resplit = resplit, raw = True)


# - prepared -

mix_ = np.concatenate(
    [normal_, anomalous],
    axis = 0,
    )
truth_ = np.ones(len(mix_), dtype = 'int64')
truth_[:len(normal_)] = 0
truth_ = truth_.astype('bool')

#training set
X = normal.copy()

#model
ae = Autoencoder()


# - training -

#trained
ae.compile(LossAD = nn.L1Loss)

####comparison
latents = [2, 4, 9, 18, 36]
for l in latents:
    ae.fit(X, latent = l, q_threshold = q_threshold)

    #detection
    print('\n\n --- Result ---\n')
    prediction_, reconstructions_ = ae.detect(mix_, truth_, return_histplot = True)

    #saved
    os.makedirs('figures', exist_ok = True)
    reconstructions_.savefig('figures/reconstruction-{latent}.png'.format(latent = l), dpi = 300)
