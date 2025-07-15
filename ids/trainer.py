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

learning_rate = 0.0001
epsilon = 1e-7
batch_size = 32
epochs = 3


class Trainer:
    """
    reference = [
        'device',
        'learning_rate',
        'epsilon',
        'batch_size',
        'epochs',
        ]
    """
    def __init__(self, Optimizer = optim.AdamW, LossFn = nn.MSELoss):
        if not issubclass(Optimizer, optim.Optimizer):
            raise TypeError('The optimizer should be a subclass of \'torch.nn.optim.Optimizer\'.')
        if not issubclass(LossFn, nn.Module):
            raise TypeError('\'LossFn\' should be a subclass of \'torch.nn.Module\'.')

        #initialized
        self.Optimizer = None
        self.loss_fn = None
        self.batchloss = None
        self.trained_array = None
        self.trained_ae = None

        #pushed
        self.Optimizer = Optimizer
        self.loss_fn = LossFn()

    def __repr__(self):
        return 'trainer'
    
    def train(self, X, ae):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if X.dtype != np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        if self.Optimizer is None:
            raise NotImplementedError('The optimizer has not been constructed.')
        if self.loss_fn is None:
            raise NotImplementedError('The loss function has not been constructed.')
        X = X.copy()
        Optimizer = self.Optimizer    #pulled
        loss_fn = self.loss_fn    #pulled

        #processed
        data = ae.process(X)

        #to gpu
        data = data.to(device)
        ae.to(device)
        logger.info('\'device\' is allocated to \'data\' and \'model\'.')

        optimizer = Optimizer(
            ae.parameters(),
            lr = learning_rate,
            eps = epsilon,
            )

        loader = DataLoader(
            data,
            batch_size = batch_size,
            shuffle = True,
            )
        batchloss = []
        logger.info('Training begins.')
        for lll in range(epochs):
            ae.train()
            last_epoch = []
            if logger.getEffectiveLevel() > 20:
                temp = loader
            else:
                temp = tqdm(loader, leave = False, ncols = 70)
            for t in temp:

                output = ae(t)
                loss = loss_fn(output, t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                last_epoch.append(loss.detach())    ###


            # Even without the epoch logging, retrieving and conversion of the losses to arrays every epoch is needed to prevent the Tensors' RAM explosion.
            last_epoch = torch.stack(last_epoch, dim = 0)
            last_epoch = last_epoch.cpu()
            last_epoch = last_epoch.numpy()
            last_epoch = last_epoch.astype('float64')

            #epoch logging
            if logger.getEffectiveLevel() > 20:
                pass
            else:
                print('Epoch {epoch:>3} | loss: {epochloss:<7}'.format(
                    epoch = lll + 1,
                    epochloss = last_epoch.mean(axis = 0, dtype = 'float64').round(decimals = 6),
                    ))

            batchloss.append(last_epoch)

        batchloss = np.concatenate(batchloss, axis = 0)

        logger.info(' - Training finished - ')

        #back to cpu
        ae.cpu()

        #pushed
        self.batchloss = batchloss.copy()
        self.trained_array = X.copy()
        self.trained_ae = ae


    def plot_losses(self):
        if self.batchloss is None:
            raise NotImplementedError('No training has been done.')
        batchloss = self.batchloss.copy()    #pulled
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Losses')
        ax.set_xlabel('batch')
        ax.set_ylabel('loss')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot = ax.plot(
            np.arange(1, len(batchloss)+1, dtype = 'int64'), batchloss,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.1,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = batchloss[-1].round(decimals = 4).tolist(),
                ),
            )
        ax.legend()

        return fig
