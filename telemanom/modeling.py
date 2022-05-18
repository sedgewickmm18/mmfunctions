from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.metrics import MeanSquaredError,MeanAbsoluteError

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy as sp
import os
import logging

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')

model_metric = MeanSquaredError()


class Model:
    def __init__(self, config, run_id, Channel=None, Path=None, Train=True):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.name = "Model"
        self.config = config
        self.chan_id = None
        if Channel is not None:
            self.chan_id = Channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.scaler = None
        self.model = None
        self.history = None

        if Path is None:
            Path = ""

        # bypass default training in constructor
        if not Train:
            self.new_model((None, Channel.X_train.shape[2]))
        elif not self.config.train:
            try:
                self.load(Path)
            except FileNotFoundError:
                path = os.path.join(Path, 'data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))
                self.train_new(Channel)
                self.save(Path)
        else:
            self.train_new(Channel)
            self.save(Path)

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name) + "\n" + str(self.model.summary())
        return out

    def load(self, Path=None):
        """
        Load model for channel.
        """

        if Path is None:
            Path = ""

        logger.info('Loading pre-trained model')

        try:
            self.model = load_model(os.path.join(Path, 'data', self.config.use_id,
                                             'models', self.chan_id + '.h5'), compile=False)
        except Exception as e:
            return False

        # compile with custom loss metric
        self.model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer,
                           metrics=[model_metric])

        # reset y_hat
        self.y_hat = np.array([])
        return True

    def new_model(self, Input_shape):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if self.model is not None:
            return

        # reset y_hat
        self.y_hat = np.array([])

        self.model = Sequential()

        self.model.add(LSTM(
            self.config.layers[0],
            input_shape=Input_shape,
            return_sequences=True))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(LSTM(
            self.config.layers[1],
            return_sequences=False))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(Dense(
            self.config.n_predictions))
        self.model.add(Activation('linear'))

        self.model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer,
                           metrics=[model_metric])

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        # scaling - archived
        self.scaler = channel.scaler

        # instatiate model with input shape from training data
        self.new_model((None, channel.X_train.shape[2]))

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=1)]

        self.history = self.model.fit(channel.X_train,
                                      channel.y_train,
                                      batch_size=self.config.lstm_batch_size,
                                      epochs=self.config.epochs,
                                      validation_split=self.config.validation_split,
                                      callbacks=cbs,
                                      verbose=True)

    def train_more(self, channel, more_epochs):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        # instatiate model with input shape from training data
        #self.new_model((None, channel.X_train.shape[2]))

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        self.history = self.model.fit(channel.X_train,
                                      channel.y_train,
                                      batch_size=self.config.lstm_batch_size,
                                      #epochs=self.config.epochs,
                                      epochs=more_epochs,
                                      validation_split=self.config.validation_split,
                                      callbacks=cbs,
                                      verbose=True)

    def save(self, Path=None):
        """
        Save trained model.
        """
        if Path is None:
            Path = ""

        self.model.save(os.path.join(Path, 'data', self.run_id, 'models',
                                     '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions

            '''
            try:
                print ('Aggregate: ', start_idx, type(y_hat_batch),
                        y_hat_batch.shape, ' append ', y_hat_t[0])
            except Exception:
                print ('Aggregate: ', start_idx, type(y_hat_batch))
                pass
            '''

            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel, Train=False, Path=None):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        if Train:
            num_batches = int((channel.y_train.shape[0] - self.config.l_s)
                              / self.config.batch_size)
        else:
            num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                              / self.config.batch_size)

        logger.debug("predict: num_batches ", num_batches)

        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                if Train:
                    idx = channel.y_train.shape[0]
                else:
                    idx = channel.y_test.shape[0]

            if Train:
                X_train_batch = channel.X_train[prior_idx:idx]
                y_hat_batch = self.model.predict(X_train_batch)
            else:
                X_test_batch = channel.X_test[prior_idx:idx]
                y_hat_batch = self.model.predict(X_test_batch)

            try:
                logger.debug("predict: batch ", i, " - ", y_hat_batch.shape)
            except Exception:
                logger.debug("predict: batch ", i)
                pass

            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        if Train:
            channel.y_train_hat = self.y_hat
            #if self.scaler_first is not None:
            #    channel.y_train_hat = self.scaler_first.inverse_transform(channel.y_train_hat.reshape(-1, 1))
        else:
            channel.y_hat = self.y_hat
            #if self.scaler_first is not None:
            #    channel.y_hat = self.scaler_first.inverse_transform(channel.y_hat.reshape(-1, 1))

        if Path is None:
            Path = ""

        np.save(os.path.join(Path, 'data', self.run_id, 'y_hat', '{}.npy'
                                   .format(self.chan_id)), self.y_hat)

        return channel
