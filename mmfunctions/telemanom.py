import numpy as np
import scipy as sp
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from iotfunctions.base import BaseEstimatorFunction
from iotfunctions.ui import (UIMultiItem)
# from sklearn.utils.validation import check_X_y  # , check_array, check_is_fitted
import logging
import pandas as pd
import more_itertools as mit
import os
import yaml

from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'
_IS_PREINSTALLED = False


def STR(x):
    if x is None:
        return ''
    return str(x)

#
# Channel, Error, Model inherited from NASA's Telemanom project
#   https://arxiv.org/pdf/1802.04431.pdf
#   https://github.com/khundman/telemanom
#


class TelemanomConfig:
    """
    Loads parameters from config.yaml into global object
    """

    def __init__(self, path_to_config=None):
        self.path_to_config = path_to_config
        self.dictionary = None
        if path_to_config is not None:
            self.get_from_yaml(path_to_config)

    def get_from_yaml(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)

        with open(self.path_to_config, "r") as f:
            try:
                self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)
            except Exception as e:
                logger.error('Loading YAML configuration file failed with ' + str(e))
                pass

        if self.dictionary is not None:
            for k, v in self.dictionary.items():
                setattr(self, k, v)

    def __str__(self):
        out = yaml.dump(self.dictionary)
        return out


class TelemanomChannel:
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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name)
        out = out + '\nTraining data shape: ' + STR(self.X_train.shape) + ', ' + STR(self.y_train.shape) + \
            '\n  Test data shape: ' + STR(self.X_test.shape) + ', ' + str(self.y_test.shape) + \
            '\n  Pred data shape: ' + STR(self.y_hat.shape) + \
            '\n  Original data shape: ' + STR(self.train.shape) + ', ' + STR(self.test.shape)
        return out

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
            self.y_train = data[:, -self.config.n_predictions:, 0]  # telemetry value to predict is at position 0
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]  # telemetry value to predict is at position 0

    def set_data(self, Train=None, Test=None):
        """
        Set train and test data and shape it properly into overlapping windows
        """
        self.train = Train
        self.test = Test

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)


class TelemanomErrors:
    def __init__(self, channel, config, run_id):
        """
        Batch processing of errors between actual and predicted values
        for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            run_id (str): Datetime referencing set of predictions in use

        Attributes:
            config (obj): see Args
            window_size (int): number of trailing batches to use in error
                calculation
            n_windows (int): number of windows in test values for channel
            i_anom (arr): indices of anomalies in channel test values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in test values
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq
            e (arr): errors in prediction (predicted - actual)
            e_s (arr): exponentially-smoothed errors in prediction
            normalized (arr): prediction errors as a percentage of the range
                of the channel values
        """

        self.config = config
        self.name = "Errors"
        self.window_size = self.config.window_size
        self.n_windows = int((channel.y_test.shape[0] -
                              (self.config.batch_size * self.window_size))
                             / self.config.batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []

        # raw prediction error
        self.e = [abs(y_h-y_t[0]) for y_h, y_t in
                  zip(channel.y_hat, channel.y_test)]

        smoothing_window = int(self.config.batch_size * self.config.window_size
                               * self.config.smoothing_perc)
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError('len(y_hat) != len(y_test): {}, {}'
                             .format(len(channel.y_hat), len(channel.y_test)))

        # smoothed prediction error
        self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window)\
            .mean().values.flatten()

        # for values at beginning < sequence length, just use avg
        if not channel.id == 'C-2':  # anomaly occurs early in window
            self.e_s[:self.config.l_s] = \
                [np.mean(self.e_s[:self.config.l_s * 2])] * self.config.l_s

        self.normalized = np.mean(self.e / np.ptp(channel.y_test))
        logger.info("normalized prediction error: {0:.2f}"
                    .format(self.normalized))

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name)
        return out

    def adjust_window_size(self, channel):
        """
        Decrease the historical error window size (h) if number of test
        values is limited.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int((channel.y_test.shape[0]
                                 - (self.config.batch_size * self.window_size))
                                 / self.config.batch_size)
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError('Batch_size ({}) larger than y_test (len={}). '
                                 'Adjust in config.yaml.'
                                 .format(self.config.batch_size,
                                         channel.y_test.shape[0]))

    def merge_scores(self):
        """
        If anomalous sequences from subsequent batches are adjacent they
        will automatically be combined. This combines the scores for these
        initial adjacent sequences (scores are calculated as each batch is
        processed) where applicable.
        """

        merged_scores = []
        score_end_indices = []

        for i, score in enumerate(self.anom_scores):
            if not score['start_idx']-1 in score_end_indices:
                merged_scores.append(score['score'])
                score_end_indices.append(score['end_idx'])

    def process_batches(self, channel):
        """
        Top-level function for the Error class that loops through batches
        of values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        self.adjust_window_size(channel)

        for i in range(0, self.n_windows+1):
            prior_idx = i * self.config.batch_size
            idx = (self.config.window_size * self.config.batch_size) \
                + (i * self.config.batch_size)
            if i == self.n_windows:
                idx = channel.y_test.shape[0]

            window = ErrorWindow(channel, self.config, prior_idx, idx, self, i)

            window.find_epsilon()
            window.find_epsilon(inverse=True)

            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.prune_anoms()
            window.prune_anoms(inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.i_anom = np.sort(np.unique(
                np.append(window.i_anom, window.i_anom_inv))).astype('int')
            window.score_anomalies(prior_idx)

            # update indices to reflect true indices in full set of values
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores

        if len(self.i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups = [list(group) for group in
                      mit.consecutive_groups(self.i_anom)]
            self.E_seq = [(int(g[0]), int(g[-1])) for g in groups
                          if not g[0] == g[-1]]

            # additional shift is applied to indices so that they represent the
            # position in the original data array, obtained from the .npy files,
            # and not the position on y_test (See PR #27).
            self.E_seq = [(e_seq[0] + self.config.l_s,
                           e_seq[1] + self.config.l_s) for e_seq in self.E_seq]

            self.merge_scores()


class ErrorWindow:
    def __init__(self, channel, config, start_idx, end_idx, errors, window_num):
        """
        Data and calculations for a specific window of prediction errors.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        self.config = config
        self.anom_scores = []

        self.window_num = window_num

        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        self.e_s = errors.e_s[start_idx:end_idx]

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e)
                                 for e in self.e_s])

        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        self.y_test = channel.y_test[start_idx:end_idx]
        self.sd_values = np.std(self.y_test)

        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
        self.inter_range = self.perc_high - self.perc_low

        # ignore initial error values until enough history for processing
        self.num_to_ignore = self.config.l_s * 2
        # if y_test is small, ignore fewer
        if len(channel.y_test) < 2500:
            self.num_to_ignore = self.config.l_s
        if len(channel.y_test) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
            buffer = np.arange(1, self.config.error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,
                                            np.array([i+buffer for i in i_anom])
                                             .flatten(),
                                            np.array([i-buffer for i in i_anom])
                                             .flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) \
                    / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) \
                    / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) \
                    / (len(E_seq) ** 2 + len(i_anom))

                # sanity checks / guardrails
                if score >= max_score and len(E_seq) <= 5 and \
                        len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all, inverse=False):
        """
        Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            errors_all (obj): Errors class object containing list of all
            previously identified anomalies in test set
        """

        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        # Check: scale of errors compared to values too small?
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s)
                > (.05 * self.inter_range)) or not max(self.e_s) > 0.05:
            return

        i_anom = np.argwhere((e_s >= epsilon) &
                             (e_s > 0.05 * self.inter_range)).reshape(-1,)

        if len(i_anom) == 0:
            return
        buffer = np.arange(1, self.config.error_buffer+1)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # if it is first window, ignore initial errors (need some history)
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.config.batch_size]

        i_anom = np.sort(np.unique(i_anom))

        # capture max of non-anomalous values below the threshold
        # (used in filtering process)
        batch_position = self.window_num * self.config.batch_size
        window_indices = np.arange(0, len(e_s)) + batch_position
        adj_i_anom = i_anom + batch_position
        window_indices = np.setdiff1d(window_indices,
                                      np.append(errors_all.i_anom, adj_i_anom))
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))

        # group anomalous indices into continuous sequences
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        """
        Remove anomalies that don't meet minimum separation from the next
        closest anomaly or error value

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """

        E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse \
            else self.non_anom_max_inv

        if len(E_seq) == 0:
            return

        E_seq_max = np.array([max(e_s[e[0]:e[1]+1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        for i in range(0, len(E_seq_max_sorted)-1):
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) \
                    / E_seq_max_sorted[i] < self.config.p:
                i_to_remove = np.append(i_to_remove, np.argwhere(
                    E_seq_max == E_seq_max_sorted[i]))
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)

        if len(E_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(E_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1]+1)
                                          for e_seq in E_seq])

        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]

        for e_seq in groups:

            score_dict = {
                "start_idx": e_seq[0] + prior_idx,
                "end_idx": e_seq[-1] + prior_idx,
                "score": 0
            }

            score = max([abs(self.e_s[i] - self.epsilon)
                         / (self.mean_e_s + self.sd_e_s) for i in
                         range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv)
                             / (self.mean_e_s + self.sd_e_s) for i in
                             range(e_seq[0], e_seq[-1] + 1)])

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)


class TelemanomModel:
    def __init__(self, config, run_id, channel, Train=True):
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
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.history = None

        # bypass default training in constructor
        if not Train:
            self.new_model((None, channel.X_train.shape[2]))
        elif not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))
                self.train_new(channel)
                self.save(path)
        else:
            self.train_new(channel)
            self.save(path)

    def __str__(self):
        out = '\n%s:%s' % (self.__class__.__name__, self.name) + "\n" + str(self.model.summary())
        return out

    def load(self):
        """
        Load model for channel.
        """

        logger.info('Loading pre-trained model')
        self.model = load_model(os.path.join('data', self.config.use_id,
                                             'models', self.chan_id + '.h5'))

    def new_model(self, Input_shape):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        if self.model is not None:
            return

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
                           optimizer=self.config.optimizer)

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        # instatiate model with input shape from training data
        self.new_model((None, channel.X_train.shape[2]))

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        self.history = self.model.fit(channel.X_train,
                                      channel.y_train,
                                      batch_size=self.config.lstm_batch_size,
                                      epochs=self.config.epochs,
                                      validation_split=self.config.validation_split,
                                      callbacks=cbs,
                                      verbose=True)

    def save(self):
        """
        Save trained model.
        """

        self.model.save(os.path.join('data', self.run_id, 'models',
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
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel, Train=False):
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

        logger.debug("predict: num_batches " + str(num_batches))

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
                    idx = channel.y_test.shape[0]
                else:
                    idx = channel.y_train.shape[0]

            if Train:
                X_train_batch = channel.X_train[prior_idx:idx]
                y_hat_batch = self.model.predict(X_train_batch)
            else:
                X_test_batch = channel.X_test[prior_idx:idx]
                y_hat_batch = self.model.predict(X_test_batch)

            logger.debug(str(type(y_hat_batch)))
            if type(y_hat_batch) == list:
                logger.debug('predict: batch ' + str(i) + ' - ' + str(len(y_hat_batch)))
            else:
                logger.debug('predict: batch ' + str(i) + ' - ' + str(y_hat_batch.shape))

            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        if Train:
            channel.y_train_hat = self.y_hat
        else:
            if self.config.FFT:
                logger.info('FFT modelling')
                channel.y_hat = sp.fft.irfft(self.y_hat)
            else:
                channel.y_hat = self.y_hat

        return channel


class TelemanomEstimator(BaseEstimator):

        def __init__(self, conf=None):
            if conf is None:
                self.conf = TelemanomConfig('./config.yaml')
            else:
                self.conf = conf
            self.conf.dictionary['l_s'] = 250
            self.conf.dictionary['epochs'] = 80
            self.conf.dictionary['dropout'] = 0.2
            self.conf.l_s = 250
            self.conf.dropout = 0.2
            self.conf.lstm_batch_size = 80
            self.conf.batch_size = 100
            self.chan = TelemanomChannel(self.conf, "MyDevice")
            self.model = None
            self.errors = None
            return

        def fit(self, X, y=None):
            """
            Predict future data points of the first column of X with X
                When y is not None, fuse y to X to predict
            """
            # Check that X and y have correct shape
            # X, y = check_X_y(X, y)

            # make sure it's a numpy array
            X_ = X
            try:
                X_ = X.values
            except Exception:
                pass

            self.chan.train = X_
            msg = 'Fit: feature data shape: ' + str(X_.shape) + ', target data shape: '
            if y is None:
                msg = msg + '<nothing>'
            else:
                try:
                    self.chan.train = np.vstack((y, X_.reshape(X_.shape[0],))).T
                except Exception:
                    pass
                msg = msg + str(y.shape) + ', running with: ' + str(self.chan.train.shape)

            logger.info(msg)

            self.chan.shape_data(self.chan.train)

            self.model = TelemanomModel(self.conf, self.conf.use_id, self.chan, False)

            self.model.train_new(self.chan)

            return self

        def predict(self, X, y=None):

            # make sure it's a numpy array
            X_ = X
            try:
                X_ = X.values
            except Exception:
                pass

            self.chan.test = X_
            msg = 'Predict: feature data shape: ' + str(X_.shape) + ', target data shape: '
            if y is None:
                msg = msg + '<nothing>'
            else:
                try:
                    self.chan.test = np.vstack((y, X_.reshape(X_.shape[0],))).T
                except Exception:
                    pass
                msg = msg + str(y.shape) + ', running with: ' + str(self.chan.test.shape)

            logger.info(msg)

            self.chan.shape_data(self.chan.test, train=False)

            # clear old predictions
            self.model.y_hat = np.array([])

            self.model.batch_predict(self.chan)

            gap = len(self.chan.y_test) - len(self.model.y_hat)
            if gap > 0:
                padding = np.zeros(gap)
                self.chan.y_hat = np.append(padding, self.model.y_hat)

            self.errors = TelemanomErrors(self.chan, self.conf, self.conf.use_id)

            print(self.errors.E_seq, ' \n', self.errors.anom_scores)

            padding = np.zeros(self.conf.l_s + self.conf.n_predictions)  # start at 250 + 10
            return np.append(padding, self.chan.y_hat)
            # return self.chan.y_hat

        def score(self, X=None, y=None):

            return 0.9


class LSTMRegressor(BaseEstimatorFunction):
    '''
    Regressor based on LSTM
    '''
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True
    estimators_per_execution = 1
    num_rounds_per_estimator = 1
    test_size = 0.05

    def LSTMPipeline(self):
        tconf = TelemanomConfig('./config.yaml')
        steps = [('scaler', StandardScaler()), ('lstm', TelemanomEstimator(tconf))]
        return Pipeline(steps)

    def set_estimators(self):
        # lstm
        # self.estimators['lstm_regressor'] = (self.LSTMPipeline, self.params)
        self.estimators['lstm_regressor'] = (TelemanomEstimator, self.params)
        logger.info('LSTMRegressor start searching for best model')

    def execute_train_test_split(self, df):
        return (df, df)

    def __init__(self, features, targets, predictions=None,
                 n_estimators=None, num_leaves=None, learning_rate=None, max_depth=None):
        super().__init__(features=features, targets=targets, predictions=predictions)
        self.experiments_per_execution = 1
        self.auto_train = False
        self.params = {}

        self.stop_auto_improve_at = -2

    def fit_with_search_cv(self, estimator, params, df_train, target, features):
        logger.info('Fit directly')
        estimator.best_params_ = {}
        estimator = estimator.fit(X=df_train[features], y=df_train[target])
        self.estimator = estimator
        return estimator

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            # per entity - copy for later inplace operations
            # dfe = df_copy.loc[[entity]].dropna(how='all')
            # dfe = df_copy.loc[[entity]].copy()
            # try:
                dfe = super()._execute(df_copy.loc[[entity]], entity)
                print(df_copy.columns)
                # for c in self.predictions:
                df_copy.loc[entity, self.predictions] = dfe[self.predictions]
                # df_copy = df_copy.loc[[entity]] = dfe
                print(df_copy.columns)
            # except Exception as e:
                # logger.info('GBMRegressor for entity ' + str(entity) + ' failed with: ' + str(e))
                # continue
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)

    @classmethod
    def get_input_items(cls):
        return ['features', 'targets']
