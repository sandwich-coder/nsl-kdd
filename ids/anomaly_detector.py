from environment import *
logger = logging.getLogger(name = __name__)
from sklearn.metrics import precision_score, recall_score, f1_score

def _compress(X, ae):
    Z = ae.process(X, train = False)
    Z = ae.encoder(Z)
    Z = Z.detach()    ###
    Z = Z.numpy()
    Z = Z.astype('float64')
    return Z


class AnomalyDetector:
    def __init__(self, normal, anomalous, ae, manual = False):
        if not isinstance(normal, np.ndarray):
            raise TypeError('The normal should be a \'numpy.ndarray\'.')
        if not isinstance(anomalous, np.ndarray):
            raise TypeError('The anomalous should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not isinstance(manual, bool):
            raise TypeError('\'manual\' should be boolean.')
        if not (normal.ndim == anomalous.ndim == 2):
            raise ValueError('The arrays must be tabular.')
        if normal.shape[1] != anomalous.shape[1]:
            raise ValueError('The normal and anomalous must have the same number of features.')
        if normal.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            normal = normal.astype('float64')
        if anomalous.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            anomalous = anomalous.astype('float64')
        normal = normal.copy()
        anomalous = anomalous.copy()

        #initialized
        self.ae = None
        self.metric = None
        self.threshold = None

        #Euclidean distance
        def diff(X, Y):
            error = (Y - X) ** 2
            error = error.sum(axis = 1, dtype = 'float64')
            error = np.sqrt(error, dtype = 'float64')
            return error

        normal_error = diff(
            normal,
            ae.flow(normal),
            )
        anomalous_error = diff(
            anomalous,
            ae.flow(anomalous),
            )

        if manual:
            fig = pp.figure(layout = 'constrained')
            ax = fig.add_subplot()
            ax.set_box_aspect(1)
            ax.set_title('Reconstruction Errors')
            ax.set_xticks([])
            pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

            temp = 25 / len(normal_error) ** 0.5
            if temp > 1:
                temp = 1
            plot_1 = ax.plot(
                np.linspace(0, 1, num = len(normal_error), dtype = 'float64'), normal_error,
                marker = 'o', markersize = 3 * temp,
                linestyle = '',
                alpha = 0.8,
                color = 'tab:blue',
                label = 'normal',
                )
            temp = 25 / len(anomalous_error) ** 0.5
            if temp > 1:
                temp = 1
            plot_2 = ax.plot(
                np.linspace(0, 1, num = len(anomalous_error), dtype = 'float64'), anomalous_error,
                marker = 'o', markersize = 3 * temp,
                linestyle = '',
                alpha = 0.8,
                color = 'tab:red',
                label = 'anomalous',
                )

            ax.legend()

            fig.show()
            threshold = input('threshold: ')
            threshold = float(threshold)

        else:
            threshold = np.quantile(normal_error, 0.99, axis = 0)

        #pushed
        self.ae = ae
        self.metric = diff
        self.threshold = threshold

    def __repr__(self):
        return 'anomaly detector'

    def predict(self, contaminated, label = None):
        if not isinstance(contaminated, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if label is not None:
            if not isinstance(label, np.ndarray):
                raise TypeError('The label should be boolean.')
        if contaminated.ndim != 2:
            raise ValueError('The input must tabular.')
        if label is not None:
            if label.ndim != 1:
                raise ValueError('The labels must be 1-dimensional.')
        if label is not None:
            if len(label) != len(contaminated):
                raise ValueError('The labels must have the same length as the input.')
        if contaminated.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            contaminated = contaminated.astype('float64')
        if self.ae is None:
            raise NotImplementedError('The autoencoder has not been constructed.')
        if self.metric is None:
            raise NotImplementedError('The metric has not been constructed.')
        if self.threshold is None:
            raise NotImplementedError('The threshold has not been constructed.')
        contaminated = contaminated.copy()
        ae = self.ae    #pulled
        metric = self.metric    #pulled
        threshold = self.threshold    #pulled

        error = metric(
            contaminated,
            ae.flow(contaminated),
            )

        prediction = error >= threshold

        if label is not None:

            print('')
            print('      Precision: {precision}'.format(
                precision = round(precision_score(label, prediction), ndigits = 3),
                ))
            print('         Recall: {recall}'.format(
                recall = round(recall_score(label, prediction), ndigits = 3),
                ))
            print('             F1: {f1}'.format(
                f1 = round(f1_score(label, prediction), ndigits = 3),
                ))
            print('\n')


        return prediction
