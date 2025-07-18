from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sb


class AnomalyDetector:
    def __init__(self, X, ae, LossFn, quantile = 0.99):
        if not isinstance(X, np.ndarray):
            raise TypeError('The normal should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not issubclass(LossFn, nn.Module):
            raise TypeError('The loss function must be a subclass of \'torch.nn.Module\'.')
        if not isinstance(quantile, float):
            raise TypeError('The loss quantile should be a float.')
        if not 0 < quantile < 1:
            raise ValueError('The quantile must be between 0 and 1.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        #initialized
        self._ae = None
        self._LossFn = None
        self._threshold = None
        self._in_features = None

        loss_fn = LossFn(reduction = 'none')
        normal_data = ae.process(X, train = False)
        
        normal_loss = loss_fn(ae(normal_data).detach(), normal_data)    ###
        _ = normal_loss.numpy()
        normal_loss = _.astype('float64')
        normal_loss = normal_loss.mean(axis = 1, dtype = 'float64')

        threshold = np.quantile(normal_loss, quantile, axis = 0).tolist()

        #pushed
        self._ae = ae
        self._LossFn = LossFn
        self._threshold = threshold
        self._in_features = X.shape[1]

    def __repr__(self):
        return 'anomaly detector'

    def predict(self, A, truth = None, return_histplot = False):
        if not isinstance(A, np.ndarray):
            raise TypeError('The dataset should be a \'numpy.ndarray\'.')
        if not (truth is None or isinstance(truth, np.ndarray)):
            raise TypeError('\'truth\' should be a \'numpy.ndarray\'.')
        if not isinstance(return_histplot, bool):
            raise TypeError('\'return_histplot\' should be boolean.')
        if A.ndim != 2:
            raise ValueError('The dataset must be tabular.')
        if A.shape[1] != self._in_features:
            raise ValueError('The dataset must have the same feature count.')
        if not (truth is None or truth.dtype == np.bool):
            raise ValueError('\'truth\' must be of \'numpy.bool\'.')
        if not (truth is None or truth.ndim == 1):
            raise ValueError('\'truth\' must be 1-dimensional.')
        if not (truth is None or len(truth) == len(A)):
            raise ValueError('\'truth\' must have the same length as the dataset.')
        if truth is None and return_histplot:
            raise ValueError('\'return_histplot\' is valid only when the truth is given.')
        if A.dtype != np.float64:
            logging.warning('The dtype doesn\'t match.')
            A = A.astype('float64')
        if self._ae is None:
            raise NotImplementedError('The autoencoder has not been constructed.')
        if self._LossFn is None:
            raise NotImplementedError('The loss function has not been constructed.')
        if self._threshold is None:
            raise NotImplementedError('The threshold has not been set.')
        A = A.copy()
        ae = self._ae    #pulled
        LossFn = self._LossFn    #pulled
        threshold = self._threshold    #pulled

        returns = []

        loss_fn = LossFn(reduction = 'none')
        data = ae.process(A, train = False)

        loss = loss_fn(ae(data).detach(), data)    ###
        _ = loss.numpy()
        loss = _.astype('float64')
        loss = loss.mean(axis = 1, dtype = 'float64')

        prediction = loss >= threshold
        returns.append(prediction)

        if truth is not None:

            if return_histplot:
                fig = pp.figure(layout = 'constrained')
                ax = fig.add_subplot()
                ax.set_box_aspect(0.6),
                ax.set_title('Reconstruction Losses')
                ax.set_xlabel('loss')
                ax.set_ylabel('proportion (%)')
                pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

                losses = pd.DataFrame({
                    'loss': loss,
                    'truth': truth,
                    })
                sb.histplot(
                    data = losses,
                    x = 'loss',
                    hue = 'truth',
                    bins = 100,
                    binrange = [0, 1],
                    stat = 'percent', common_norm = False,
                    ax = ax,
                    )

                ax.axvline(
                    x = losses[losses['truth'] == False]['loss'].quantile(0.9),
                    ymin = 0.3, ymax = 0.7,
                    linestyle = '--',
                    color = 'tab:grey',
                    label = 'Q 0.9',
                    )
                ax.axvline(
                    x = losses[losses['truth'] == False]['loss'].quantile(0.99),
                    ymin = 0.3, ymax = 0.7,
                    linestyle = '--',
                    color = 'tab:brown',
                    label = 'Q 0.99',
                    )
                ax.axvline(
                    x = threshold,
                    ymin = 0.1, ymax = 0.9,
                    linestyle = '-.',
                    color = 'black',
                    label = 'threshold',
                    )
                ax.legend()

                returns.append(fig)


            print('      Precision: {precision}'.format(
                precision = round(precision_score(truth, prediction), ndigits = 3),
                ))
            print('         Recall: {recall}'.format(
                recall = round(recall_score(truth, prediction), ndigits = 3),
                ))
            print('             F1: {f1}'.format(
                f1 = round(f1_score(truth, prediction), ndigits = 3),
                ))

        if len(returns) > 1:
            return returns
        elif len(returns) == 1:
            return returns[0]
