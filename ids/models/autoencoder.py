from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self, in_features):
        if not isinstance(in_features, int):
            raise TypeError('The number of features should be an integer.')
        if in_features < 1:
            raise ValueError('The number of features must be positive.')

        #initialized
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.in_features = None
        self.scaler = None

        encoder = nn.Sequential(
            nn.Sequential(nn.Linear(in_features, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, 5), nn.Tanh()),
            )
        decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 32), nn.ReLU()),
            nn.Sequential(nn.Linear(32, in_features), nn.Tanh()),
            )

        with torch.no_grad():
            nn.init.xavier_uniform_(encoder[-1][0].weight)
            nn.init.xavier_uniform_(decoder[-1][0].weight)

        #pushed
        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features


    def forward(self, t):
        if t.size(dim = 1) != self.in_features:
            raise ValueError('The number of features must match the input layer.')    # Checking of the in-features should be placed in the 'forward' instead of the 'process' and 'unprocess'.
        t = torch.clone(t)

        t = self.encoder(t)
        t = self.decoder(t)

        return t


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        if not train:
            if self.scaler is None:
                raise NotImplementedError('The scaler has not been constructed.')
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


    # This method solely aims to be the inverse. It doesn't add any other functionality.
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
        if self.scaler is None:
            raise NotImplementedError('The scaler has not been constructed.')
        processed = torch.clone(processed)
        scaler = self.scaler    #pulled

        _ = processed.numpy()
        unprocessed = _.astype('float64')
        unprocessed = scaler.inverse_transform(unprocessed)
        return unprocessed


    #process->forward->unprocess
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
