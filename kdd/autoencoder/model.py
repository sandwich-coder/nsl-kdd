from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.preprocessing import MinMaxScaler

from._trainer import Trainer
from .dimension_estimator import DimensionEstimator


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self._in_features = 122
        self._latent = 5
        self._encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self._in_features, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, self._latent), nn.Sigmoid()),
            )
        self._decoder = nn.Sequential(
            nn.Sequential(nn.Linear(self._latent, 40), nn.GELU()),
            nn.Sequential(nn.Linear(40, self._in_features), nn.Sigmoid()),
            )

        with torch.no_grad():
            nn.init.xavier_uniform_(self._encoder[-1][0].weight)
            nn.init.xavier_uniform_(self._decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, data):
        if data.size(dim = 1) != self._in_features:
            raise ValueError('The number of features must match the input layer.')

        output = self._encoder(data)
        output = self._decoder(output)

        return output


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')

        if not train:
            pass
        else:
            self._scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
            self._scaler.fit(X)

        processed = self._scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)

        return processed

    def unprocess(self, processed):
        if not isinstance(processed, torch.Tensor):
            raise TypeError('The input should be a \'torch.Tensor\'.')
        if processed.requires_grad:
            raise ValueError('The input must not be on a graph. \nThis method doesn\'nt automatically detach such Tensors.')
        if processed.dim() != 2:
            raise ValueError('The input must be tabular.')
        if processed.dtype != torch.float32:
            logger.warning('The dtype doesn\'t match.')
            processed = processed.to(torch.float32)

        _ = processed.numpy()
        unprocessed = _.astype('float64')
        unprocessed = self._scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')

        self.eval()

        Y = self.process(X, train = False)
        Y = self(Y)
        Y = Y.detach()    ###
        Y = self.unprocess(Y)

        return Y


    def compile(self, LossFn = nn.MSELoss):
        if not issubclass(LossFn, nn.Module):
            raise TypeError('The loss function should be a subclass of \'torch.nn.Module\'.')

        self._trainer = Trainer(LossFn = LossFn)

    def fit(self, X, return_descentplot = False, auto_latent = False):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(return_descentplot, bool):
            raise TypeError('\'return_descentplot\' should be boolean.')
        if not isinstance(auto_latent, bool):
            raise TypeError('Whether to enable the dimension estimation should be boolean.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')

        if auto_latent:
            estimator = DimensionEstimator()
            dimension = estimator(X, exact = True, trim = True)
            logger.info('intrinsic dimension: {}'.format(round(dimension, ndigits = 3)))
            self._latent = round(dimension)
        logger.info('The latent dimension is set to {}'.format(self._latent))

        self._trainer.train(X, self)

        if return_descentplot:
            return self._trainer.plot_descent()


    def get_LossFn(self):
        return self._trainer.LossFn
