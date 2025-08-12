from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from._trainer import Trainer
from .dimension_estimator import DimensionEstimator


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
    def __repr__(self):
        return 'autoencoder'

    def forward(self, data):
        if data.size(dim = 1) != self._in_features:
            raise ValueError('The number of features must match the input layer.')
        assert hasattr(self, '_encoder'), 'no encoder'
        assert hasattr(self, '_decoder'), 'no decoder'

        output = self._encoder(data)
        output = self._decoder(output)

        return output


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            raise ValueError('The input must be of \'numpy.float64\'.')

        if not train:
            assert hasattr(self, '_scaler'), 'no scaler'
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
            raise ValueError('The input must be of \'torch.float32\'.')
        assert hasattr(self, '_scaler'), 'no scaler'

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
            raise ValueError('The input must be of \'numpy.float64\'.')

        self.eval()

        Y = self.process(X, train = False)
        Y = self(Y)
        Y = Y.detach()    ###
        Y = self.unprocess(Y)

        return Y


    def compile(self, LossFn = nn.MSELoss, LossAD = None):
        if not issubclass(LossFn, nn.Module):
            raise TypeError('The loss function should be a subclass of \'torch.nn.Module\'.')
        if LossAD is not None:
            if not issubclass(LossAD, nn.Module):
                raise TypeError('The loss function for detection should be a subclass of \'torch.nn.Module\'.')

        self._trainer = Trainer(LossFn)
        if LossAD is not None:
            self._LossAD = LossAD
        else:
            self._LossAD = LossFn


    def fit(self, X, return_descentplot = False, auto_latent = False, q_threshold = 0.99):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(return_descentplot, bool):
            raise TypeError('\'return_descentplot\' should be boolean.')
        if not isinstance(auto_latent, bool):
            raise TypeError('Whether to enable the dimension estimation should be boolean.')
        if not isinstance(q_threshold, float):
            raise TypeError('The detection threshold should be a float.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            raise ValueError('The input must be of \'numpy.float64\'.')
        if not 0 < q_threshold < 1:
            raise ValueError('The detection threshold must be between 0 and 1.')
        assert hasattr(self, '_trainer'), 'no trainer'
        assert hasattr(self, '_LossAD'), 'no detection loss'

        if auto_latent:
            estimator = DimensionEstimator()
            dimension = estimator(X, exact = True, trim = True)
            logger.info('intrinsic dimension: {}'.format(round(dimension, ndigits = 2)))
            self._latent = round(dimension)
            logger.info('The latent dimension is set to {}'.format(self._latent))
        else:
            self._latent = 9

        self._in_features = 122
        self._encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self._in_features, 128), nn.GELU()),
            nn.Sequential(nn.Linear(128, 64), nn.GELU()),
            nn.Sequential(nn.Linear(64, 32), nn.GELU()),
            nn.Sequential(nn.Linear(32, self._latent), nn.Sigmoid()),
            )
        self._decoder = nn.Sequential(
            nn.Sequential(nn.Linear(self._latent, 32), nn.GELU()),
            nn.Sequential(nn.Linear(32, 64), nn.GELU()),
            nn.Sequential(nn.Linear(64, 128), nn.GELU()),
            nn.Sequential(nn.Linear(128, self._in_features), nn.Sigmoid()),
            )

        with torch.no_grad():
            nn.init.xavier_uniform_(self._encoder[-1][0].weight)
            nn.init.xavier_uniform_(self._decoder[-1][0].weight)

        self._trainer.train(X, self)

        #threshold set
        loss_fn = self._LossAD(reduction = 'none')    #the loss function for detection
        normal_data = self.process(X, train = False)
        normal_loss = loss_fn(self(normal_data).detach(), normal_data)    ###
        _ = normal_loss.numpy()
        normal_loss = _.astype('float64')
        normal_loss = normal_loss.mean(axis = 1, dtype = 'float64')
        self._threshold = np.quantile(normal_loss, q_threshold, axis = 0).tolist()

        if return_descentplot:
            return self._trainer.plot_descent()


    def detect(self, mix, truth = None, return_histplot = False):
        if not isinstance(mix, np.ndarray):
            raise TypeError('The dataset should be a \'numpy.ndarray\'.')
        if not isinstance(truth, np.ndarray) and truth is not None:
            raise TypeError('\'truth\' should be a \'numpy.ndarray\'.')
        if not isinstance(return_histplot, bool):
            raise TypeError('\'return_histplot\' should be boolean.')
        if mix.ndim != 2:
            raise ValueError('The dataset must be tabular.')
        if mix.shape[1] != self._in_features:
            raise ValueError('The dataset must have the same feature count.')
        if mix.dtype != np.float64:
            raise ValueError('The dataset must be of \'numpy.float64\'.')
        if truth.ndim != 1 and truth is not None:
            raise ValueError('\'truth\' must be 1-dimensional.')
        if truth.dtype != np.bool and truth is not None:
            raise ValueError('\'truth\' must be of \'numpy.bool\'.')
        if len(truth) != len(mix) and truth is not None:
            raise ValueError('\'truth\' must have the same length as the dataset.')
        if truth is None and return_histplot:
            raise ValueError('\'return_histplot\' is valid only when the truth is given.')
        assert hasattr(self, '_threshold'), 'no threshold'
        returns = []

        #prepared
        loss_fn = self._LossAD(reduction = 'none')    #the loss function for detection
        data = self.process(mix, train = False)

        #loss measured
        loss = loss_fn(self(data).detach(), data)    ###
        _ = loss.numpy()
        loss = _.astype('float64')
        loss = loss.mean(axis = 1, dtype = 'float64')

        #detection
        detection = loss >= self._threshold
        returns.append(detection)

        if truth is not None:

            if return_histplot:
                fig = pp.figure(layout = 'constrained')
                ax = fig.add_subplot()
                ax.set_box_aspect(0.6)
                ax.set_title('Reconstruction Loss')
                ax.set_xlabel('loss')
                ax.set_ylabel('proportion (%)')
                pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

                bincount = 500
                binrange = [
                    np.quantile(loss, 0, axis = 0),
                    np.quantile(loss, 0.99, axis = 0),
                    ]

                #normal pmf
                normal_loss = loss[~truth]    # Simple indexing, which includes slicing, returns a view, or a shallow copy in other words. However, "fancy indexing" returns a copy instead of a view, differing from the numpy's usual indexing behavior one would expect. Therefore the names are separated without copying.
                prob_normal, edge_normal = np.histogram(normal_loss, range = binrange, bins = bincount)
                prob_normal = prob_normal / prob_normal.sum(axis = 0, dtype = 'int64')

                #anomalous pmf
                anomalous_loss = loss[truth]
                prob_anomalous, edge_anomalous = np.histogram(anomalous_loss, range = binrange, bins = bincount)
                prob_anomalous = prob_anomalous / prob_anomalous.sum(axis = 0, dtype = 'int64')

                #histogram plots
                plot_1 = ax.stairs(
                    prob_normal * 100,
                    edges = edge_normal,
                    fill = True,
                    color = 'tab:blue', alpha = 0.4,
                    label = 'normal',
                    )
                plot_2 = ax.stairs(
                    prob_anomalous * 100,
                    edges = edge_anomalous,
                    fill = True,
                    color = 'tab:red', alpha = 0.4,
                    label = 'anomalous',
                    )

                #Q points
                ax.axvline(
                    x = np.quantile(normal_loss, 0.9),
                    linestyle = '-.', linewidth = 0.2,
                    color = 'tab:grey',
                    label = 'Q 0.9',
                    )
                ax.axvline(
                    x = np.quantile(normal_loss, 0.99),
                    linestyle = '-.', linewidth = 0.2,
                    color = 'purple',
                    label = 'Q 0.99',
                    )
                ax.axvline(
                    x = self._threshold,
                    linestyle = '--', linewidth = 0.2,
                    color = 'black',
                    label = 'threshold',
                    )

                ax.legend()
                returns.append(fig)


            #false-negative rate
            positives = truth.astype('int64').sum(axis = 0, dtype = 'int64').tolist()
            fn = truth & ~detection
            fn = fn.astype('int64').sum(axis = 0, dtype = 'int64').tolist()
            fn_rate = fn / positives

            #false-positive rate
            negatives = (~truth).astype('int64').sum(axis = 0, dtype = 'int64').tolist()
            fp = ~truth & detection
            fp = fp.astype('int64').sum(axis = 0, dtype = 'int64').tolist()
            fp_rate = fp / negatives

            print('false negative: {rate:>4}%'.format(
                rate = round(fn_rate * 100, ndigits = 1),
                ))
            print('false positive: {rate:>4}%'.format(
                rate = round(fp_rate * 100, ndigits = 1),
                ))
            print('     Precision: {precision}'.format(
                precision = round(precision_score(truth, detection), ndigits = 3),
                ))
            print('        Recall: {recall}'.format(
                recall = round(recall_score(truth, detection), ndigits = 3),
                ))
            print('            F1: {f1}'.format(
                f1 = round(f1_score(truth, detection), ndigits = 3),
                ))

        if len(returns) > 1:
            return returns
        elif len(returns) == 1:
            return returns[0]
