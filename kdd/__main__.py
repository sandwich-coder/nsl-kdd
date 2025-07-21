import sys, os, subprocess
import argparse

#python check
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('This module is intended to be run on Python 3.12.')


# - console input -

parser = argparse.ArgumentParser()
parser.add_argument('--resplit', help = 'whether to merge the train and test sets and resplit randomly, retaining the attack type distribution.', default = 'False')
parser.add_argument('--qthreshold', help = 'the quantile threshold above which the reconstruction loss is deemed as anomalous', default = '0.99')
parser.add_argument('--log', help = 'logging level', default = 'INFO')
args = parser.parse_args()

if args.resplit == 'True':
    resplit = True
elif args.resplit == 'False':
    resplit = False
q_threshold = float(args.qthreshold)
logging_level = args.log


from environment import *
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
        logger.info('The installed pytorch is not built with CUDA. Install a CUDA-enabled.')
    if supported_cuda is None:
        logger.info('The nvidia driver does not exist.')
elif float(supported_cuda) < float(torch.version.cuda):
    logger.info('The supported CUDA is lower than installed. Upgrade the driver.')
else:
    logger.info('- Nvidia driver checked -')


dataset = 'nsl-kdd'

#loaded
loader = Loader()
normal, normal_ = loader.load(dataset, attack = False, resplit = resplit)
anomalous, anomalous_ = loader.load(dataset, attack = True, resplit = resplit)

#for traditional ML
normal_df, normal_df_ = loader.load(dataset, attack = False, resplit = resplit, raw = True)
anomalous_df, anomalous_df_ = loader.load(dataset, attack = True, resplit = resplit, raw = True)


# - prepared -

mixed = np.concatenate([normal, anomalous], axis = 0)
truth = np.ones(mixed.shape[0], dtype = 'int64')
truth[:len(normal)] = 0
truth = truth.astype('bool')

mixed_ = np.concatenate([normal_, anomalous_], axis = 0)
truth_ = np.ones(mixed_.shape[0], dtype = 'int64')
truth_[:len(normal_)] = 0
truth_ = truth_.astype('bool')


#training set
X = normal.copy()
X_ = normal_.copy()

#model
ae = Autoencoder()

#trained
ae.compile()
descent = ae.fit(X, return_descentplot = True, q_threshold = q_threshold)

#detection

print('\n')
print(' --- Train ---\n')
prediction, reconstructions = ae.detect(mixed, truth, return_histplot = True)
print('\n')
print(' --- Test ---\n')
prediction_, reconstructions_ = ae.detect(mixed_, truth_, return_histplot = True)

#saved
os.makedirs('figures', exist_ok = True)
descent.savefig('figures/descent.png', dpi = 300)
reconstructions.savefig('figures/reconstructions-train.png', dpi = 600)
reconstructions_.savefig('figures/reconstructions-test.png', dpi = 600)
