from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self):

        #initialized
        super().__init__()
        self._encoder = None
        self._decoder = None
        self._scaler = None
        self._in_features = None

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
        self._encoder = encoder
        self._decoder = decoder
        self._in_features = encoder[0][0].weight.size(dim = 1)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, data):
        if data.size(dim = 1) != self._in_features:
            raise ValueError('The number of features must match the input layer.')
        data = torch.clone(data)
        encoder = self._encoder    #pulled
        decoder = self._decoder    #pulled

        output = encoder(data)
        output = decoder(output)

        return output


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        if not train and self._scaler is None:
            raise NotImplementedError('The scaler has not been constructed.')
        X = X.copy()
        scaler = self._scaler    #pulled

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
            scaler.fit(X)

        processed = scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)

        #pushed
        self._scaler = scaler

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
        if self._scaler is None:
            raise NotImplementedError('The scaler has not been constructed.')
        processed = torch.clone(processed)
        scaler = self._scaler    #pulled

        _ = processed.numpy()
        unprocessed = _.astype('float64')
        unprocessed = scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        self.eval()

        Y = self.process(X, train = False)
        Y = self(Y)
        Y = Y.detach()    ###
        Y = self.unprocess(Y)

        return Y
