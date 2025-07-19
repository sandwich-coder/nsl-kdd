from environment import *
logger = logging.getLogger(name = __name__)
from torch.utils.data import DataLoader
from tqdm import tqdm

if not torch.cuda.is_available():
    device = torch.device('cpu')
    logger.info('CPU is assigned to \'device\' as fallback.')
else:
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')

Optimizer = optim.AdamW
learning_rate = 0.0001
epsilon = 1e-7

batch_size = 32
epochs = 100


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

        #to gpu
        data = data.to(device)
        self._ae.to(device)
        logger.info('\'device\' is allocated to the dataset and model.')

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
        logger.info(' - Training Start -')
        for lll in range(epochs):
            self._ae.train()
            last_epoch = []
            if logger.getEffectiveLevel() > 20:
                iteration = loader
            else:
                iteration = tqdm(loader, leave = False, ncols = 70)
            for llll in iteration:

                out = self._ae(llll)
                loss = loss_fn(out, llll)

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
                print('Epoch {epoch:>3} | loss: {epochloss:<7}'.format(
                    epoch = lll + 1,
                    epochloss = last_epoch.mean(axis = 0, dtype = 'float64').round(decimals = 6),
                    ))

            self._batchloss.append(last_epoch)

        self._batchloss = np.concatenate(self._batchloss, axis = 0)
        logger.info(' - Training finished -')

        #back to cpu
        self._ae.cpu()


    def plot_descent(self):
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Losses')
        ax.set_xlabel('batch')
        ax.set_ylabel('loss')
        pp.setp(ax.get_yticklabels(), rotation = 90, va = 'center')

        plot = ax.plot(
            np.arange(1, len(self._batchloss)+1, dtype = 'int64'), self._batchloss,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.05,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = self._batchloss[-1].round(decimals = 4).tolist(),
                ),
            )
        ax.legend()

        return fig
