from common import *
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


    def fit(self, X, latent = 'auto', return_descentplot = False, q_threshold = 0.99):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(latent, int) and latent != 'auto':
            raise TypeError('The latent dimension should be an integer.')
        if not isinstance(return_descentplot, bool):
            raise TypeError('\'return_descentplot\' should be boolean.')
        if not isinstance(q_threshold, float):
            raise TypeError('The detection threshold should be a float.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            raise ValueError('The input must be of \'numpy.float64\'.')
        if latent != 'auto':
            if latent < 1:
                raise ValueError('The latent dimension must be positive.')
        if not 0 < q_threshold < 1:
            raise ValueError('The detection threshold must be between 0 and 1.')
        assert hasattr(self, '_trainer'), 'no trainer'
        assert hasattr(self, '_LossAD'), 'no detection loss'

        #dimension estimation
        if latent == 'auto':
            estimator = DimensionEstimator()
            dimension = estimator(X, exact = True, trim = True)
            logger.info('intrinsic dimension: {dimension:.2f}'.format(dimension = dimension))
            latent = round(dimension)
        logger.info('The bottleneck is set to {latent}'.format(latent = latent))

        fold = 2
        self._latent = latent #stored for the loss plot

        self._in_features = 122
        self._encoder = nn.Sequential()
        fan_in = self._in_features
        while fan_in / fold ** 2 >= latent:
            fan_out = fan_in // fold
            self._encoder.append(
                nn.Sequential(nn.Linear(fan_in, fan_out), nn.GELU()),
                ),
            fan_in = fan_out
        self._encoder.append(
            nn.Sequential(nn.Linear(fan_in, latent), nn.Sigmoid()),
            )

        self._decoder = nn.Sequential()
        fan_in = latent
        while fan_in * fold ** 2 <= self._in_features:
            fan_out = fan_in * fold
            self._decoder.append(
                nn.Sequential(nn.Linear(fan_in, fan_out), nn.GELU()),
                )
            fan_in = fan_out
        self._decoder.append(
            nn.Sequential(nn.Linear(fan_in, self._in_features), nn.Sigmoid()),
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

        ... ## The ROC plot is to be added.

        if truth is not None:

            if return_histplot:
                fig = pp.figure(layout = 'constrained', facecolor = 'ivory')
                ax = fig.add_subplot()
                ax.set_box_aspect(0.5)
                ax.set_title('Reconstruction Loss (bottleneck: {latent})'.format(latent = self._latent))
                ax.set_xlabel('loss')
                ax.set_ylabel('proportion (%)')
                pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

                bincount = 300
                binrange = [
                    np.quantile(loss, 0, axis = 0),
                    np.quantile(loss, 0.99, axis = 0),
                    ]

                #normal pmf
                normal_loss = loss[~truth] # Simple indexing, which includes slicing, returns a view, or a shallow copy in other words. However, "fancy indexing" returns a copy instead of a view, differing from the numpy's usual indexing behavior one would expect. Therefore the names are separated without copying.
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
                vline = ax.axvline(
                    x = self._threshold,
                    marker = '', color = 'black', alpha = 0.8,
                    linestyle = '--', linewidth = 1,
                    label = 'threshold',
                    )

                #cmf plots
                ax.set_xmargin(0); ax.set_ymargin(0) #for those of axes coordinates
                plot_3 = ax.plot(
                    np.linspace(0, 1, num = len(prob_normal) + 1, endpoint = True),
                    np.cumsum(
                        np.concatenate([np.array([0.]), prob_normal], axis = 0),
                        axis = 0, dtype = 'float64',
                        ),
                    marker = '', color = 'tab:blue', alpha = 0.5,
                    linestyle = '-', linewidth = 1,
                    label = 'normal cmf',
                    transform = ax.transAxes,
                    )
                plot_4 = ax.plot(
                    np.linspace(0, 1, num = len(prob_anomalous) + 1, endpoint = True),
                    np.cumsum(
                        np.concatenate([np.array([0.]), prob_anomalous], axis = 0),
                        axis = 0, dtype = 'float64',
                        ),
                    marker = '', color = 'tab:red', alpha = 0.5,
                    linestyle = '-', linewidth = 1,
                    label = 'anomalous cmf',
                    transform = ax.transAxes,
                    )

                ax.legend(handles = [
                    vline,
                    plot_1,
                    plot_3[0],
                    plot_2,
                    plot_4[0],
                    ], loc = 'upper right')
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


            # - table -

            console = Console()

            table = Table()
            table.add_column('Metric', justify = 'center')
            table.add_column('Score', justify = 'right')

            table.add_row(
                Text('false negative', style = 'red'),
                Text(format(fn_rate, '.1%'), style = 'bold red'),
                )
            table.add_row('false positive', format(fp_rate, '.1%'))
            table.add_row('Precision', format(precision_score(truth, detection), '.3f'))
            table.add_row('Recall', format(recall_score(truth, detection), '.3f'))
            table.add_row('F1', format(f1_score(truth, detection), '.3f'))

            console.print(table)


        if len(returns) > 1:
            return returns
        elif len(returns) == 1:
            return returns[0]
