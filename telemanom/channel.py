import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
import os
import logging

logger = logging.getLogger('telemanom')


class Channel:
    def __init__(self, config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.name = "Channel"
        self.config = config
        self.scaler = None    # MinMaxScaler
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name)
        try:
            out = out + "\nTraining data shape: " + str(self.X_train.shape) + ", " + str(self.y_train.shape)
        except Exception as e1:
            logger.debug(str(e1))
            pass
        try:
            out = out + "\n  Test data shape: " + str(self.X_test.shape) + ", " + str(self.y_test.shape)
        except Exception as e2:
            logger.debug(str(e2))
            pass
        try:
            out = out + "\n  Pred data shape: " + str(self.y_hat.shape)
        except Exception as e3:
            logger.debug(str(e3))
            pass
        try:
            out = out + "\n  Original data shape: " + str(self.train.shape) + ", " + str(self.test.shape)
        except Exception as e4:
            logger.debug(str(e4))
            pass
        return out

    # set Channel scaler from Model after model restore
    def set_scaler(self, scaler_first, scaler_full):
        self.scaler = scaler

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """
        if train:
            if self.scaler is None:
                self.scaler = MinMaxScaler().fit(arr)

        if self.scaler is not None:
            arr = self.scaler.transform(arr)

        data = []

        print(arr.shape)
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            snippet = arr[i:i + self.config.l_s + self.config.n_predictions]
            data.append(snippet)

        data = np.array(data)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, :-self.config.n_predictions, :]
            self.y_train = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0

    def load_data(self, Path=None):
        """
        Load train and test data from local.
        """
        if Path is None:
            Path = ''

        try:
            self.train = np.load(os.path.join(Path, "data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join(Path, "data", "test", "{}.npy".format(self.id)))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)
