from environment import *
logger = logging.getLogger(name = __name__)
from torch.utils.data import DataLoader
from tqdm import tqdm

if not torch.cuda.is_available():
    device = torch.device('cpu')
    logger.warning('CPU is assigned to \'device\' as fallback.')
else:
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')

Optimizer = optim.AdamW
learning_rate = 0.0001
epsilon = 1e-6

batch_size = 16
epochs = 30


class Trainer:
    """
    reference = [
        'device',
        'Optimizer',
        'learning_rate',
        'epsilon',
        'batch_size',
        'epochs',
        ]
    """
    def __init__(self, LossFn):
        self._LossFn = LossFn
    def __repr__(self):
        return 'trainer'

    @property
    def LossFn(self):
        return self._LossFn

    def train(self, X, ae):
        self._ae = ae

        #processed
        data = self._ae.process(X)
        logger.info('data size: {}'.format(tuple(data.size())))

        #to gpu
        data = data.to(device)
        self._ae.to(device)
        logger.info('\'device\' is allocated to the data and model.')

        #configured
        optimizer = Optimizer(
            self._ae.parameters(),
            lr = learning_rate,
            eps = epsilon,
            )
        loss_fn = self._LossFn()

        loader = DataLoader(
            data,
            batch_size = batch_size,
            shuffle = True,
            )
        self._batchloss = []
        logger.info(' - Training -')
        if logger.getEffectiveLevel() > 20:
            pass
        else:
            print('Epoch |   Loss    ')
            print('===== | ==========')
        for l in range(epochs):
            self._ae.train()
            last_epoch = []
            if logger.getEffectiveLevel() > 20:
                iteration = loader
            else:
                iteration = tqdm(loader, leave = False, ncols = 70)
            for ll in iteration:

                out = self._ae(ll)
                loss = loss_fn(out, ll)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                last_epoch.append(loss.detach())    ###


            last_epoch = torch.stack(last_epoch, dim = 0)
            last_epoch = last_epoch.cpu()
            last_epoch = last_epoch.numpy()
            last_epoch = last_epoch.astype('float64')

            if logger.getEffectiveLevel() > 20:
                pass
            else:
                print('{epoch:>5} | {epochloss:.4E}'.format(
                    epoch = l + 1,
                    epochloss = last_epoch.mean(axis = 0, dtype = 'float64').tolist(),
                    ))    # I prefer the keyword-based 'format' method than the f-string, as the former can handle "temporary" values easier.

            self._batchloss.append(last_epoch)

        self._batchloss = np.concatenate(self._batchloss, axis = 0)
        logger.info(' - Finished -')

        #back to cpu
        self._ae.cpu()


    def plot_descent(self):
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3), facecolor = 'ivory')
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Training Loss')
        ax.set_xlabel('batch')
        ax.set_ylabel('loss')
        pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

        plot = ax.plot(
            np.arange(1, len(self._batchloss)+1, dtype = 'int64'), self._batchloss,
            marker = 'o', markersize = 0.01,
            linestyle = '--', linewidth = 0.005,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = self._batchloss[-1].round(decimals = 4).tolist(),
                ),
            )
        ax.legend()

        return fig
