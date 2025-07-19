from environment import *
logger = logging.getLogger(name = __name__)
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from tqdm import tqdm


class DimensionEstimator:
    def __init__(self):
        pass
    def __repr__(self):
        return 'dimension estimator'

    def __call__(
        self,
        X,
        exact = False,
        trim = False,
        divisions = 10,
        batch_count = 1000,
        ):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(exact, bool):
            raise TypeError('Whether exact without rounding should be boolean.')
        if not isinstance(trim, bool):
            raise TypeError('Whether outliers be removed should be boolean.')
        if not isinstance(divisions, int):
            raise TypeError('The number of divisions should be an integer.')
        if not isinstance(batch_count, int):
            raise TypeError('The number of batches should be an integer.')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        if divisions < 2:
            raise ValueError('\'divisions\' must be greater than 1.')
        if batch_count < 1:
            raise ValueError('\'batch_count\' must be positive.')
        assert X.dtype == np.float64, 'The dtype doesn\'t match.'
        if divisions > 100:
            logger.error('The number of divisions higher than 100 will be meaningless.')
        if trim:
            logger.info('The dataset is trimmed by the isolation forest.')

        #trimmed
        if trim:
            forest = IsolationForest()
            forest.fit(X)
            valid = forest.predict(X) > 0
            X = X[valid]

        #oriented
        pca = PCA(svd_solver = 'full')
        pca.fit(X)
        X = pca.transform(X)

        # A tile is always adjacent to all the others in the case of two divisions.
        if divisions == 2:
            binary = np.where(X >= 0, 1, -1)
            binary = binary.astype('int64')

            occupied = np.unique(binary, axis = 0)
            dimension = np.log(len(occupied), dtype = 'float64') / np.log(2, dtype = 'float64')
            if exact:
                dimension = dimension.tolist()
            else:
                dimension = dimension.round().astype('int64')
                dimension = dimension.tolist()
            return dimension


        #quantized
        width = (X[:, 0].max(axis = 0) - X[:, 0].min(axis = 0)) / np.float64(divisions)
        if divisions % 2 != 0:
            tile = X / width
        else:
            tile = X / width - np.float64(0.5)
        tile = tile.round().astype('int64')
        tile = np.unique(tile, axis = 0)


        # - counted -

        batch = copy(np.array_split(tile, batch_count, axis = 0))
        if batch_count > len(tile):
            batch = batch[:len(tile)]

        adjacency = []
        for lll in tqdm(batch, ncols = 70):

            distance = cdist(lll, tile, metric = 'chebyshev')
            is_adjacent = np.isclose(distance, 1, atol = 0)

            adjacency_batch = is_adjacent.astype('int64')
            adjacency_batch = adjacency_batch.sum(axis = 1, dtype = 'int64')
            adjacency.append(adjacency_batch)

        adjacency = np.concatenate(adjacency, axis = 0)


        dimension = np.log(adjacency.mean(axis = 0) + 1, dtype = 'float64') / np.log(3, dtype = 'float64')
        if exact:
            dimension = dimension.tolist()
        else:
            dimension = dimension.round().astype('int64')
            dimension = dimension.tolist()
        return dimension
