import sys, os, subprocess

#python check
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('This module is intended to be run on Python 3.12.')


from environment import *
logger = logging.getLogger(name = 'main')
logging.basicConfig(level = 'INFO')
from sklearn.model_selection import train_test_split

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
normal, normal_ = loader.load(dataset, benign = True)
anomalous, anomalous_ = loader.load(dataset, benign = False)


# - prepared -

sampler = Sampler()

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
descent = ae.fit(X, return_descentplot = True, q_threshold = 0.9)

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
