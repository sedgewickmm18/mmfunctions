# *****************************************************************************
# © Copyright IBM Corp. 2018-2020.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

"""
The Built In Functions module contains preinstalled functions
"""

import itertools as it
import datetime as dt
import logging

# for gradient boosting
import lightgbm
import numpy as np
import pandas as pd
import scipy as sp
from pyod.models.cblof import CBLOF
#  for Spectral Analysis
from scipy import signal, fftpack
#   for KMeans
#from skimage import util as skiutil  # for nifty windowing
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.covariance import MinCovDet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   minmax_scale, PowerTransformer)
from sklearn.utils import check_array
# for Matrix Profile
import stumpy

#import statsmodels.api as sm
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEstimatorFunction, BaseSimpleAggregator)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMulti, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)

# VAE
import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False

Error_SmallWindowsize = 0.0001
Error_Generic = 0.0002

FrequencySplit = 0.3
DefaultWindowSize = 12
SmallEnergy = 1e-20

KMeans_normalizer = 1
Spectral_normalizer = 100 / 2.8
FFT_normalizer = 1
Saliency_normalizer = 1
Generalized_normalizer = 1 / 300


# from
# https://stackoverflow.com/questions/44790072/sliding-window-on-time-series-data
def view_as_windows(temperature, length, step):
    logger.info('VIEW ' + str(temperature.shape) + ' ' + str(length) + ' ' + str(step))

    def moving_window(x, length, _step=1):
        if type(step) != 'int' or _step < 1:
            logger.info('MOVE ' + str(_step))
            _step = 1
        streams = it.tee(x, length)
        return zip(*[it.islice(stream, i, None, _step) for stream, i in zip(streams, it.count(step=1))])

    x_=list(moving_window(temperature, length, step))
    return np.asarray(x_)



def custom_resampler(array_like):
    # initialize
    if 'gap' not in dir():
        gap = 0

    if array_like.values.size > 0:
        gap = 0
        return 0
    else:
        gap += 1
        return gap


def min_delta(df):
    # minimal time delta for merging

    if len(df.index.names) > 1:
        df2 = df.copy()
        df2.index = df2.index.droplevel(list(range(1, df.index.nlevels)))
    else:
        df2 = df

    try:
        mindelta = df2.index.to_series().diff().min()
    except Exception as e:
        logger.debug('Min Delta error: ' + str(e))
        mindelta = pd.Timedelta('5 seconds')

    if mindelta == dt.timedelta(seconds=0) or pd.isnull(mindelta):
        mindelta = pd.Timedelta('5 seconds')

    return mindelta, df2


def set_window_size_and_overlap(windowsize, trim_value=2 * DefaultWindowSize):
    # make sure it exists
    if windowsize is None:
        windowsize = DefaultWindowSize

    # make sure it is positive and not too large
    trimmed_ws = np.minimum(np.maximum(windowsize, 1), trim_value)

    # overlap
    if trimmed_ws == 1:
        ws_overlap = 0
    else:
        # larger overlap - half the window
        ws_overlap = trimmed_ws // 2

    return trimmed_ws, ws_overlap


def dampen_anomaly_score(array, dampening):
    if dampening is None:
        dampening = 0.9  # gradient dampening

    if dampening >= 1:
        return array

    if dampening < 0.01:
        return array

    if array.size <= 1:
        return array

    gradient = np.gradient(array)

    # dampened
    grad_damp = np.float_power(abs(gradient), dampening) * np.sign(gradient)

    # reconstruct (dampened) anomaly score by discrete integration
    integral = []
    x = array[0]
    for x_el in np.nditer(grad_damp):
        x = x + x_el
        integral.append(x)

    # shift array slightly to the right to position anomaly score
    array_damp = np.roll(np.asarray(integral), 1)
    array_damp[0] = array_damp[1]

    # normalize
    return array_damp / dampening / 2


# Saliency helper functions
# copied from https://github.com/y-bar/ml-based-anomaly-detection
#   remove the boring part from an image resp. time series
def series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


# Saliency class
#  see https://www.inf.uni-hamburg.de/en/inst/ab/cv/research/research1-visual-attention.html
class Saliency(object):
    def __init__(self, amp_window_size, series_window_size, score_window_size):
        self.amp_window_size = amp_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def transform_saliency_map(self, values):
        """
        Transform a time-series into spectral residual, which is method in computer vision.
        For example, See https://docs.opencv.org/master/d8/d65/group__saliency.html
        :param values: a list or numpy array of float values.
        :return: silency map and spectral residual
        """

        freq = np.fft.fft(values)
        mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)

        # remove the boring part of a timeseries
        spectral_residual = np.exp(np.log(mag) - series_filter(np.log(mag), self.amp_window_size))

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        # and apply inverse fourier transform
        saliency_map = np.fft.ifft(freq)
        return saliency_map

    def transform_spectral_residual(self, values):
        saliency_map = self.transform_saliency_map(values)
        spectral_residual = np.sqrt(saliency_map.real ** 2 + saliency_map.imag ** 2)
        return spectral_residual


def merge_score(dfEntity, dfEntityOrig, column_name, score, mindelta):
    """
    Fit interpolated score to original entity slice of the full dataframe
    """

    # equip score with time values, make sure it's positive
    score[score < 0] = 0
    dfEntity[column_name] = score

    # merge
    dfEntityOrig = pd.merge_asof(dfEntityOrig, dfEntity[column_name], left_index=True, right_index=True,
                                 direction='nearest', tolerance=mindelta)

    if column_name + '_y' in dfEntityOrig:
        merged_score = dfEntityOrig[column_name + '_y'].to_numpy()
    else:
        merged_score = dfEntityOrig[column_name].to_numpy()

    return merged_score


#####
#  experimental function to interpolate over larger gaps
####
class Interpolator(BaseTransformer):
    """
    Interpolates NaN and data to be interpreted as NaN (for example 0 as invalid sensor reading)
    The window size is typically set large enough to allow for "bridging" gaps
    Missing indicates sensor readings to be interpreted as invalid.
    """

    def __init__(self, input_item, windowsize, missing, output_item):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, self.windowoverlap = set_window_size_and_overlap(windowsize)

        self.missing = missing

        self.output_item = output_item

        self.inv_zscore = None

        self.whoami = 'Interpolator'

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # remove Nan
        dfe = dfe[dfe[self.input_item].notna()]

        # remove self.missing
        dfe = dfe[dfe[self.input_item] != self.missing]

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        # replace NaN with self.missing
        temperature = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, temperature

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return (df_copy)

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()

            # get rid of entityid part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta) + ' Index: ' + str(dfe_orig.index))

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug('Module Interpolator, Entity: ' + str(entity) + ', Input: ' + str(
                self.input_item) + ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(
                self.output_item) + ', Inputsize: ' + str(temperature.size) + ', Fullsize: ' + str(
                dfe_orig[self.input_item].values.shape))

            if temperature.size <= self.windowsize:
                logger.debug(str(temperature.size) + ' <= ' + str(self.windowsize))
                dfe[self.output_item] = Error_SmallWindowsize
            else:
                logger.debug(str(temperature.size) + str(self.windowsize))
                temperatureII = None

                try:
                    # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                    #   extend it to cover the full original length
                    temperatureII = merge_score(dfe, dfe_orig, self.output_item, temperature, mindelta)

                except Exception as e:
                    logger.error('Spectral failed with ' + str(e))

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = temperatureII

        msg = 'Interpolator'
        self.trace_append(msg)

        return (df_copy)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to interpolate'))

        inputs.append(
            UISingle(name='windowsize', datatype=int, description='Minimal size of the window for interpolating data.'))
        inputs.append(UISingle(name='missing', datatype=int, description='Data to be interpreted as not-a-number.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='output_item', datatype=float, description='Interpolated data'))
        return (inputs, outputs)


#######################################################################################
# Scalers
#######################################################################################

class Standard_Scaler(BaseEstimatorFunction):
    """
    Learns and applies standard scaling
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def set_estimators(self):
        self.estimators['standard_scaler'] = (StandardScaler, self.params)
        logger.info('Standard Scaler initialized')

    def __init__(self, features=None, targets=None, predictions=None):
        super().__init__(features=features, targets=targets, predictions=predictions, keep_current_models=True)

        # do not run score and call transform instead of predict
        self.is_scaler = True
        self.experiments_per_execution = 1
        self.normalize = True  # support for optional scaling in subclasses
        self.prediction = self.predictions[0]  # support for subclasses with univariate focus

        self.params = {}

    # used by all the anomaly scorers based on it
    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data for ' + self.prediction + ' column')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.prediction]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, temperature

    # dummy function for scaler, can be replaced with anomaly functions
    def kexecute(self, entity, df_copy):
        return df_copy

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            try:
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            # support for optional scaling in subclasses
            if self.normalize:
                dfe = super()._execute(df_copy.loc[[entity]], entity)
                df_copy.loc[entity, self.predictions] = dfe[self.predictions]

            df_copy = self.kexecute(entity, df_copy)

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


class Robust_Scaler(BaseEstimatorFunction):
    """
    Learns and applies robust scaling, scaling after outlier removal
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def set_estimators(self):
        self.estimators['robust_scaler'] = (RobustScaler, self.params)
        logger.info('Robust Scaler initialized')

    def __init__(self, features=None, targets=None, predictions=None):
        super().__init__(features=features, targets=targets, predictions=predictions, keep_current_models=True)

        # do not run score and call transform instead of predict
        self.is_scaler = True
        self.experiments_per_execution = 1

        self.params = {}

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            # per entity - copy for later inplace operations
            try:
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.predictions] = dfe[self.predictions]

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


class MinMax_Scaler(BaseEstimatorFunction):
    """
    Learns and applies minmax scaling
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def set_estimators(self):
        self.estimators['minmax_scaler'] = (MinMaxScaler, self.params)
        logger.info('MinMax Scaler initialized')

    def __init__(self, features=None, targets=None, predictions=None):
        super().__init__(features=features, targets=targets, predictions=predictions, keep_current_models=True)

        # do not run score and call transform instead of predict
        self.is_scaler = True
        self.experiments_per_execution = 1

        self.params = {}

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            try:
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.predictions] = dfe[self.predictions]

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


#######################################################################################
# Anomaly Scorers
#######################################################################################

class SpectralAnomalyScore(BaseTransformer):
    """
    An unsupervised anomaly detection function.
     Applies a spectral analysis clustering techniqueto extract features from time series data and to create z scores.
     Moves a sliding window across the data signal and applies the anomalymodelto each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, self.windowoverlap = set_window_size_and_overlap(windowsize)

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.inv_zscore = None

        self.whoami = 'Spectral'

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, temperature

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return (df_copy)

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()

            # get rid of entityid part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta) + ' Index: ' + str(dfe_orig.index))

            # one dimensional time series - named temperature for catchyness
            temperature = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug(
                'Module Spectral, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) + ', Windowsize: ' + str(
                    self.windowsize) + ', Output: ' + str(self.output_item) + ', Overlap: ' + str(
                    self.windowoverlap) + ', Inputsize: ' + str(temperature.size))

            if temperature.size <= self.windowsize:
                logger.debug(str(temperature.size) + ' <= ' + str(self.windowsize))
                dfe[self.output_item] = Error_SmallWindowsize
            else:
                logger.debug(str(temperature.size) + str(self.windowsize))

                dfe[self.output_item] = Error_Generic
                if self.inv_zscore is not None:
                    dfe[self.inv_zscore] = Error_Generic

                zScoreII = None
                inv_zScoreII = None
                try:
                    # Fourier transform:
                    #   frequency, time, spectral density
                    frequency_temperature, time_series_temperature, spectral_density_temperature = signal.spectrogram(
                        temperature, fs=self.frame_rate, window='hanning', nperseg=self.windowsize,
                        noverlap=self.windowoverlap, detrend='l', scaling='spectrum')

                    # cut off freqencies too low to fit into the window
                    frequency_temperatureb = (frequency_temperature > 2 / self.windowsize).astype(int)
                    frequency_temperature = frequency_temperature * frequency_temperatureb
                    frequency_temperature[frequency_temperature == 0] = 1 / self.windowsize

                    signal_energy = np.dot(spectral_density_temperature.T, frequency_temperature)

                    signal_energy[signal_energy < SmallEnergy] = SmallEnergy
                    inv_signal_energy = np.divide(np.ones(signal_energy.size), signal_energy)

                    dfe[self.output_item] = 0.0005

                    ets_zscore = abs(sp.stats.zscore(signal_energy)) * Spectral_normalizer
                    inv_zscore = abs(sp.stats.zscore(inv_signal_energy))

                    logger.debug(
                        'Spectral z-score max: ' + str(ets_zscore.max()) + ',   Spectral inv z-score max: ' + str(
                            inv_zscore.max()))

                    # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                    #   extend it to cover the full original length
                    dfe[self.output_item] = 0.0006
                    linear_interpolate = sp.interpolate.interp1d(time_series_temperature, ets_zscore, kind='linear',
                                                                 fill_value='extrapolate')

                    zScoreII = merge_score(dfe, dfe_orig, self.output_item,
                                           abs(linear_interpolate(np.arange(0, temperature.size, 1))), mindelta)

                    if self.inv_zscore is not None:
                        linear_interpol_inv_zscore = sp.interpolate.interp1d(time_series_temperature, inv_zscore,
                                                                             kind='linear', fill_value='extrapolate')

                        inv_zScoreII = merge_score(dfe, dfe_orig, self.inv_zscore,
                                                   abs(linear_interpol_inv_zscore(np.arange(0, temperature.size, 1))),
                                                   mindelta)

                except Exception as e:
                    logger.error('Spectral failed with ' + str(e))

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

                if self.inv_zscore is not None:
                    df_copy.loc[idx[entity, :], self.inv_zscore] = inv_zScoreII

        if self.inv_zscore is not None:
            msg = 'SpectralAnomalyScoreExt'
        else:
            msg = 'SpectralAnomalyScore'
        self.trace_append(msg)

        return (df_copy)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to analyze'))

        inputs.append(UISingle(name='windowsize', datatype=int,
                               description='Size of each sliding window in data points. Typically set to 12.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='output_item', datatype=float, description='Spectral anomaly score (z-score)'))
        return (inputs, outputs)


class SpectralAnomalyScoreExt(SpectralAnomalyScore):
    """
    An unsupervised anomaly detection function.
     Applies a spectral analysis clustering techniqueto extract features from time series data and to create z scores.
     Moves a sliding window across the data signal and applies the anomalymodelto each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item, inv_zscore):
        super().__init__(input_item, windowsize, output_item)
        logger.debug(input_item)

        self.inv_zscore = inv_zscore

    def execute(self, df):
        return super().execute(df)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to analyze'))

        inputs.append(UISingle(name='windowsize', datatype=int,
                               description='Size of each sliding window in data points. Typically set to 12.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='output_item', datatype=float, description='Spectral anomaly score (z-score)'))
        outputs.append(UIFunctionOutSingle(name='inv_zscore', datatype=float,
                                           description='z-score of inverted signal energy - detects unusually low activity'))
        return (inputs, outputs)


class KMeansAnomalyScore(BaseTransformer):
    """
    An unsupervised anomaly detection function.
     Applies a k-means analysis clustering technique to time series data.
     Moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
     Try several anomaly models on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item, expr=None):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, windowoverlap = set_window_size_and_overlap(windowsize)

        # step
        self.step = self.windowsize - windowoverlap

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.whoami = 'KMeans'

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, temperature

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return (df_copy)

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()

            # get rid of entityid part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta))

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug(
                'Module KMeans, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) + ', Windowsize: ' + str(
                    self.windowsize) + ', Output: ' + str(self.output_item) + ', Overlap: ' + str(
                    self.step) + ', Inputsize: ' + str(temperature.size))

            if temperature.size > self.windowsize:
                logger.debug(str(temperature.size) + ',' + str(self.windowsize))

                # Chop into overlapping windows
                #slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
                slices = view_as_windows(temperature, self.windowsize, self.step)

                if self.windowsize > 1:
                    n_cluster = 40
                else:
                    n_cluster = 20

                n_cluster = np.minimum(n_cluster, slices.shape[0] // 2)

                logger.debug('KMeans params, Clusters: ' + str(n_cluster) + ', Slices: ' + str(slices.shape))

                cblofwin = CBLOF(n_clusters=n_cluster, n_jobs=-1)
                try:
                    cblofwin.fit(slices)
                except Exception as e:
                    logger.info('KMeans failed with ' + str(e))
                    self.trace_append('KMeans failed with' + str(e))
                    continue

                pred_score = cblofwin.decision_scores_.copy() * KMeans_normalizer

                # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                #   extend it to cover the full original length
                diff = temperature.size - pred_score.size

                time_series_temperature = np.linspace(self.windowsize // 2, temperature.size - self.windowsize // 2 + 1,
                                                      temperature.size - diff)

                linear_interpolate_k = sp.interpolate.interp1d(time_series_temperature, pred_score, kind='linear',
                                                               fill_value='extrapolate')

                zScoreII = merge_score(dfe, dfe_orig, self.output_item,
                                       linear_interpolate_k(np.arange(0, temperature.size, 1)), mindelta)

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = 'KMeansAnomalyScore'
        self.trace_append(msg)
        return (df_copy)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to analyze'))

        inputs.append(UISingle(name='windowsize', datatype=int,
                               description='Size of each sliding window in data points. Typically set to 12.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='output_item', datatype=float, description='Anomaly score (kmeans)'))
        return (inputs, outputs)


class GeneralizedAnomalyScore(BaseTransformer):
    """
    An unsupervised anomaly detection function.
     Applies the Minimum Covariance Determinant (FastMCD) technique to detect outliers.
     Moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__()
        logger.debug(input_item)

        self.whoami = 'GAM'

        self.input_item = input_item

        # use 12 by default
        self.windowsize, windowoverlap = set_window_size_and_overlap(windowsize)

        # step
        self.step = self.windowsize - windowoverlap

        # assume 1 per sec for now
        self.frame_rate = 1

        self.dampening = 1  # dampening - dampen anomaly score

        self.output_item = output_item

        self.normalizer = Generalized_normalizer

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, temperature

    def feature_extract(self, temperature):

        logger.debug(self.whoami + ': feature extract')

        #slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
        slices = view_as_windows(temperature, self.windowsize, self.step)

        return slices

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return (df_copy)

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()

            # get rid of entityid part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug('Module GeneralizedAnomaly, Entity: ' + str(entity) + ', Input: ' + str(
                self.input_item) + ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(
                self.output_item) + ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

            if temperature.size > self.windowsize:
                logger.debug(str(temperature.size) + "," + str(self.windowsize))

                temperature -= np.mean(temperature, axis=0)
                mcd = MinCovDet()

                # Chop into overlapping windows (default) or run through FFT first
                slices = self.feature_extract(temperature)

                pred_score = None

                try:
                    mcd.fit(slices)
                    pred_score = mcd.mahalanobis(slices).copy() * self.normalizer

                except ValueError as ve:

                    logger.info(self.whoami + " GeneralizedAnomalyScore: Entity: " + str(entity) + ", Input: " + str(
                        self.input_item) + ", WindowSize: " + str(self.windowsize) + ", Output: " + str(
                        self.output_item) + ", Step: " + str(self.step) + ", InputSize: " + str(
                        slices.shape) + " failed in the fitting step with \"" + str(ve) + "\" - scoring zero")

                    dfe[self.output_item] = 0
                    #  this fails in the interpolation step
                    continue

                except Exception as e:

                    dfe[self.output_item] = 0
                    logger.error(self.whoami + " GeneralizedAnomalyScore: Entity: " + str(entity) + ", Input: " + str(
                        self.input_item) + ", WindowSize: " + str(self.windowsize) + ", Output: " + str(
                        self.output_item) + ", Step: " + str(self.step) + ", InputSize: " + str(
                        slices.shape) + " failed in the fitting step with " + str(e))
                    continue

                # will break if pred_score is None
                # length of timesTS, ETS and ets_zscore is smaller than half the original
                #   extend it to cover the full original length
                diff = temperature.size - pred_score.size

                time_series_temperature = np.linspace(self.windowsize // 2, temperature.size - self.windowsize // 2 + 1,
                                                      temperature.size - diff)

                logger.debug(self.whoami + '   Entity: ' + str(entity) + ', result shape: ' + str(
                    time_series_temperature.shape) + ' score shape: ' + str(pred_score.shape))

                linear_interpolate_k = sp.interpolate.interp1d(time_series_temperature, pred_score, kind="linear",
                                                               fill_value="extrapolate")

                gam_scoreI = linear_interpolate_k(np.arange(0, temperature.size, 1))

                dampen_anomaly_score(gam_scoreI, self.dampening)

                zScoreII = merge_score(dfe, dfe_orig, self.output_item, gam_scoreI, mindelta)

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = "GeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze", ))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12."))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(name="output_item", datatype=float, description="Anomaly score (GeneralizedAnomaly)", ))
        return (inputs, outputs)


class NoDataAnomalyScore(GeneralizedAnomalyScore):
    """
    An unsupervised anomaly detection function.
     Uses FastMCD to find gaps in data.
     The function moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'NoData'
        self.normalizer = 1

        logger.debug('NoData')

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        # count the timedelta in seconds between two events
        timeSeq = (dfEntity.index.values - dfEntity.index[0].to_datetime64()) / np.timedelta64(1, 's')

        dfe = dfEntity.copy()

        # one dimensional time series - named temperature for catchyness
        #   we look at the gradient of the time series timestamps for anomaly detection
        #   might throw an exception - we catch it in the super class !!
        try:
            temperature = np.gradient(timeSeq)
            dfe[[self.input_item]] = temperature
        except Exception as pe:
            logger.info("NoData Gradient failed with " + str(pe))
            dfe[[self.input_item]] = 0
            temperature = dfe[[self.input_item]].values
            temperature[0] = 10 ** 10

        return dfe, temperature

    def execute(self, df):
        df_copy = super().execute(df)

        msg = "NoDataAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to analyze'))

        inputs.append(UISingle(name='windowsize', datatype=int,
                               description='Size of each sliding window in data points. Typically set to 12.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='output_item', datatype=float, description='No data anomaly score'))
        return (inputs, outputs)


class FFTbasedGeneralizedAnomalyScore(GeneralizedAnomalyScore):
    """
    An unsupervised and robust anomaly detection function.
     Extracts temporal features from time series data using Fast Fourier Transforms.
     Applies the GeneralizedAnomalyScore to the features to detect outliers.
     Moves a sliding window across the data signal and applies the anomaly models to each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'FFT'
        self.normalizer = FFT_normalizer

        logger.debug('FFT')

    def feature_extract(self, temperature):
        logger.debug(self.whoami + ': feature extract')

        #slices_ = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
        slices_ = view_as_windows(temperature, self.windowsize, self.step)

        slicelist = []
        for slice in slices_:
            slicelist.append(fftpack.rfft(slice))

        return np.stack(slicelist, axis=0)

    def execute(self, df):
        df_copy = super().execute(df)

        msg = "FFTbasedGeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze", ))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12."))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="output_item", datatype=float,
                                           description="Anomaly score (FFTbasedGeneralizedAnomalyScore)", ))
        return (inputs, outputs)


class MatrixProfileAnomalyScore(BaseTransformer):
    """
    An unsupervised anomaly detection function.
     Applies matrix profile analysis on time series data.
     Moves a sliding window across the data signal to calculate the euclidean distance from one window to all others to build a distance profile.
     The window size is typically set to 12 data points.
     Try several anomaly models on your data and use the one that fits your data best.
    """
    DATAPOINTS_AFTER_LAST_WINDOW = 1e-15
    INIT_SCORES = 1e-20
    ERROR_SCORES = 1e-16

    def __init__(self, input_item, output_item, window_size):
        super().__init__()
        logger.debug(f'Input item: {input_item}')
        self.input_item = input_item
        self.window_size = window_size
        self.output_item = output_item
        self.whoami = 'MatrixProfile'

    def prepare_data(self, df_entity):

        logger.debug(self.whoami + ': prepare Data')

        # operate on simple timestamp index
        if len(df_entity.index.names) > 1:
            index_names = df_entity.index.names
            dfe = df_entity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = df_entity

        # interpolate gaps - data imputation
        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series
        analysis_input = dfe[[self.input_item]].fillna(0).to_numpy(dtype=np.float64).reshape(-1, )

        return dfe, analysis_input

    def execute(self, df):
        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(f'Entities: {str(entities)}')
        df_copy[self.output_item] = self.INIT_SCORES

        # check data type
        if df_copy[self.input_item].dtype != np.float64:
            return df_copy

        for entity in entities:
            # per entity - copy for later inplace operations
            dfe = df_copy.loc[[entity]].dropna(how='all')
            dfe_orig = df_copy.loc[[entity]].copy()
            logger.debug(f' Original df shape: {df_copy.shape} Entity df shape: {dfe.shape}')

            # get rid of entity_id part of the index
            # do it inplace as we copied the data before
            dfe.reset_index(level=[0], inplace=True)
            dfe.sort_index(inplace=True)
            dfe_orig.reset_index(level=[0], inplace=True)
            dfe_orig.sort_index(inplace=True)

            # minimal time delta for merging
            mindelta, dfe_orig = min_delta(dfe_orig)

            if dfe.size >= self.window_size:
                # interpolate gaps - data imputation by default
                dfe, matrix_profile_input = self.prepare_data(dfe)
                try:  # calculate scores
                    matrix_profile = stumpy.aamp(matrix_profile_input, m=self.window_size)[:, 0]
                    # fill in a small value for newer data points outside the last possible window
                    fillers = np.array([self.DATAPOINTS_AFTER_LAST_WINDOW] * (self.window_size - 1))
                    matrix_profile = np.append(matrix_profile, fillers)
                except Exception as er:
                    logger.warning(f' Error in calculating Matrix Profile Scores. {er}')
                    matrix_profile = np.array([self.ERROR_SCORES] * dfe.shape[0])
            else:
                logger.warning(f' Not enough data to calculate Matrix Profile for entity. {entity}')
                matrix_profile = np.array([self.ERROR_SCORES] * dfe.shape[0])

            anomaly_score = merge_score(dfe, dfe_orig, self.output_item, matrix_profile, mindelta)

            idx = pd.IndexSlice
            df_copy.loc[idx[entity, :], self.output_item] = anomaly_score

        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = [UISingleItem(name="input_item", datatype=float, description="Time series data item to analyze", ),
                  UISingle(name="window_size", datatype=int,
                           description="Size of each sliding window in data points. Typically set to 12.")]

        # define arguments that behave as function outputs
        outputs = [UIFunctionOutSingle(name="output_item", datatype=float,
                                       description="Anomaly score (MatrixProfileAnomalyScore)", )]
        return inputs, outputs


#####
#  experimental function with dampening factor
####
class FFTbasedGeneralizedAnomalyScore2(GeneralizedAnomalyScore):
    """
    An unsupervised and robust anomaly detection function.
     Extracts temporal features from time series data using Fast Fourier Transforms.
     Applies the GeneralizedAnomalyScore to the features to detect outliers.
     Moves a sliding window across the data signal and applies the anomaly models to each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, dampening, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'FFT dampen'
        self.dampening = dampening
        self.normalizer = FFT_normalizer / dampening

        logger.debug('FFT')

    def feature_extract(self, temperature):
        logger.debug(self.whoami + ': feature extract')

        #slices_ = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
        slices_ = view_as_windows(temperature, self.windowsize, self.step)

        slicelist = []
        for slice in slices_:
            slicelist.append(fftpack.rfft(slice))

        return np.stack(slicelist, axis=0)

    def execute(self, df):
        df_copy = super().execute(df)

        msg = "FFTbasedGeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze", ))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12."))

        inputs.append(UISingle(name="dampening", datatype=float,
                               description="Moderate the anomaly score. Use a value <=1. Typically set to 1."))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="output_item", datatype=float,
                                           description="Anomaly score (FFTbasedGeneralizedAnomalyScore)", ))
        return (inputs, outputs)


class SaliencybasedGeneralizedAnomalyScore(GeneralizedAnomalyScore):
    """
    An unsupervised anomaly detection function.
     Based on salient region detection models,
         it uses fast fourier transform to reconstruct a signal using the salient features of a the signal.
     It applies GeneralizedAnomalyScore to the reconstructed signal.
     The function moves a sliding window across the data signal and applies its analysis to each window.
     The window size is typically set to 12 data points.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'Saliency'
        self.saliency = Saliency(windowsize, 0, 0)
        self.normalizer = Saliency_normalizer

        logger.debug('Saliency')

    def feature_extract(self, temperature):
        logger.debug(self.whoami + ': feature extract')

        temperature_saliency = self.saliency.transform_spectral_residual(temperature)

        #slices = skiutil.view_as_windows(temperature_saliency, window_shape=(self.windowsize,), step=self.step)
        slices = view_as_windows(temperature, self.windowsize, self.step)

        return slices

    def execute(self, df):
        df_copy = super().execute(df)

        msg = "SaliencybasedGeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze"))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12.", ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="output_item", datatype=float,
                                           description="Anomaly score (SaliencybasedGeneralizedAnomalyScore)", ))
        return (inputs, outputs)


#######################################################################################
# Anomaly detectors with scaling
#######################################################################################
class KMeansAnomalyScoreV2(Standard_Scaler):
    """
    An unsupervised anomaly detection function.
     Applies a k-means analysis clustering technique to time series data.
     Moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
     The normalize switch allows to learn and apply a standard scaler prior to computing the anomaly score.
     Try several anomaly models on your data and use the one that fits your data best.
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def __init__(self, input_item, windowsize, normalize, output_item, expr=None):
        super().__init__(features=[input_item], targets=[output_item], predictions=None)

        logger.debug(input_item)
        # do not run score and call transform instead of predict

        self.input_item = input_item

        # use 12 by default
        self.windowsize, windowoverlap = set_window_size_and_overlap(windowsize)

        # step
        self.step = self.windowsize - windowoverlap

        self.normalize = normalize

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.whoami = 'KMeansV2'

    def kexecute(self, entity, df_copy):

        # per entity - copy for later inplace operations
        dfe = df_copy.loc[[entity]].dropna(how='all')
        dfe_orig = df_copy.loc[[entity]].copy()

        # get rid of entityid part of the index
        # do it inplace as we copied the data before
        dfe.reset_index(level=[0], inplace=True)
        dfe.sort_index(inplace=True)
        dfe_orig.reset_index(level=[0], inplace=True)
        dfe_orig.sort_index(inplace=True)

        # minimal time delta for merging
        mindelta, dfe_orig = min_delta(dfe_orig)

        logger.debug('Timedelta:' + str(mindelta))

        # interpolate gaps - data imputation by default
        #   for missing data detection we look at the timestamp gradient instead
        dfe, temperature = self.prepare_data(dfe)

        logger.debug('Module ' + self.whoami + ', Entity: ' + str(entity) + ', Input: ' + str(
            self.input_item) + ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(
            self.output_item) + ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

        if temperature.size > self.windowsize:
            logger.debug(str(temperature.size) + ',' + str(self.windowsize))

            # Chop into overlapping windows
            #slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
            slices = view_as_windows(temperature, self.windowsize, self.step)

            if self.windowsize > 1:
                n_cluster = 40
            else:
                n_cluster = 20

            n_cluster = np.minimum(n_cluster, slices.shape[0] // 2)

            logger.debug('KMeans parms, Clusters: ' + str(n_cluster) + ', Slices: ' + str(slices.shape))

            cblofwin = CBLOF(n_clusters=n_cluster, n_jobs=-1)
            try:
                cblofwin.fit(slices)
            except Exception as e:
                logger.info('KMeans failed with ' + str(e))
                self.trace_append('KMeans failed with' + str(e))
                return df_copy

            pred_score = cblofwin.decision_scores_.copy() * KMeans_normalizer

            # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
            #   extend it to cover the full original length
            diff = temperature.size - pred_score.size

            time_series_temperature = np.linspace(self.windowsize // 2, temperature.size - self.windowsize // 2 + 1,
                                                  temperature.size - diff)

            linear_interpolate_k = sp.interpolate.interp1d(time_series_temperature, pred_score, kind='linear',
                                                           fill_value='extrapolate')

            z_score_ii = merge_score(dfe, dfe_orig, self.output_item,
                                     linear_interpolate_k(np.arange(0, temperature.size, 1)), mindelta)

            idx = pd.IndexSlice
            df_copy.loc[idx[entity, :], self.output_item] = z_score_ii

            return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='input_item', datatype=float, description='Data item to analyze'))

        inputs.append(UISingle(name='windowsize', datatype=int,
                               description='Size of each sliding window in data points. Typically set to 12.'))

        inputs.append(UISingle(name='normalize', datatype=bool, description='Flag for normalizing data.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='output_item', datatype=float, description='Anomaly score (kmeans)'))
        return (inputs, outputs)


class GeneralizedAnomalyScoreV2(Standard_Scaler):
    """
    An unsupervised anomaly detection function.
     Applies the Minimum Covariance Determinant (FastMCD) technique to detect outliers.
     Moves a sliding window across the data signal and applies the anomaly model to each window.
     The window size is typically set to 12 data points.
     The normalize switch allows to learn and apply a standard scaler prior to computing the anomaly score.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """
    # class variables
    eval_metric = staticmethod(metrics.r2_score)

    train_if_no_model = True

    def __init__(self, input_item, windowsize, normalize, output_item, expr=None):
        super().__init__(features=[input_item], targets=[output_item], predictions=None)

        logger.debug(input_item)
        # do not run score and call transform instead of predict

        self.input_item = input_item

        # use 12 by default
        self.windowsize, windowoverlap = set_window_size_and_overlap(windowsize)

        # step
        self.step = self.windowsize - windowoverlap

        self.normalize = normalize

        # assume 1 per sec for now
        self.frame_rate = 1

        self.dampening = 1  # dampening - dampen anomaly score

        self.output_item = output_item

        self.normalizer = Generalized_normalizer

        self.whoami = 'GAMV2'

    def feature_extract(self, temperature):

        logger.debug(self.whoami + ': feature extract')

        #slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
        slices = view_as_windows(temperature, self.windowsize, self.step)

        return slices

    def kexecute(self, entity, df_copy):

        # per entity - copy for later inplace operations
        dfe = df_copy.loc[[entity]].dropna(how='all')
        dfe_orig = df_copy.loc[[entity]].copy()

        # get rid of entityid part of the index
        # do it inplace as we copied the data before
        dfe.reset_index(level=[0], inplace=True)
        dfe.sort_index(inplace=True)
        dfe_orig.reset_index(level=[0], inplace=True)
        dfe_orig.sort_index(inplace=True)

        # minimal time delta for merging
        mindelta, dfe_orig = min_delta(dfe_orig)

        logger.debug('Timedelta:' + str(mindelta))

        # interpolate gaps - data imputation by default
        #   for missing data detection we look at the timestamp gradient instead
        dfe, temperature = self.prepare_data(dfe)

        logger.debug('Module ' + self.whoami + ', Entity: ' + str(entity) + ', Input: ' + str(
            self.input_item) + ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(
            self.output_item) + ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

        if temperature.size > self.windowsize:
            logger.debug(str(temperature.size) + "," + str(self.windowsize))

            temperature -= np.mean(temperature, axis=0)
            mcd = MinCovDet()

            # Chop into overlapping windows (default) or run through FFT first
            slices = self.feature_extract(temperature)

            pred_score = None

            try:
                mcd.fit(slices)
                pred_score = mcd.mahalanobis(slices).copy() * self.normalizer

            except ValueError as ve:

                logger.info(self.whoami + " GeneralizedAnomalyScore: Entity: " + str(entity) + ", Input: " + str(
                    self.input_item) + ", WindowSize: " + str(self.windowsize) + ", Output: " + str(
                    self.output_item) + ", Step: " + str(self.step) + ", InputSize: " + str(
                    slices.shape) + " failed in the fitting step with \"" + str(ve) + "\" - scoring zero")

                dfe[self.output_item] = 0
                return df_copy

            except Exception as e:

                dfe[self.output_item] = 0
                logger.error(self.whoami + " GeneralizedAnomalyScore: Entity: " + str(entity) + ", Input: " + str(
                    self.input_item) + ", WindowSize: " + str(self.windowsize) + ", Output: " + str(
                    self.output_item) + ", Step: " + str(self.step) + ", InputSize: " + str(
                    slices.shape) + " failed in the fitting step with " + str(e))
                return df_copy

            # will break if pred_score is None
            # length of timesTS, ETS and ets_zscore is smaller than half the original
            #   extend it to cover the full original length
            diff = temperature.size - pred_score.size

            time_series_temperature = np.linspace(self.windowsize // 2, temperature.size - self.windowsize // 2 + 1,
                                                  temperature.size - diff)

            logger.debug(self.whoami + '   Entity: ' + str(entity) + ', result shape: ' + str(
                time_series_temperature.shape) + ' score shape: ' + str(pred_score.shape))

            linear_interpolate_k = sp.interpolate.interp1d(time_series_temperature, pred_score, kind="linear",
                                                           fill_value="extrapolate")

            gam_scoreI = linear_interpolate_k(np.arange(0, temperature.size, 1))

            dampen_anomaly_score(gam_scoreI, self.dampening)

            zScoreII = merge_score(dfe, dfe_orig, self.output_item, gam_scoreI, mindelta)

            idx = pd.IndexSlice
            df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = "GeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze", ))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12."))

        inputs.append(UISingle(name='normalize', datatype=bool, description='Flag for normalizing data.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(name="output_item", datatype=float, description="Anomaly score (GeneralizedAnomaly)", ))
        return (inputs, outputs)


class FFTbasedGeneralizedAnomalyScoreV2(GeneralizedAnomalyScoreV2):
    """
    An unsupervised and robust anomaly detection function.
     Extracts temporal features from time series data using Fast Fourier Transforms.
     Applies the GeneralizedAnomalyScore to the features to detect outliers.
     Moves a sliding window across the data signal and applies the anomaly models to each window.
     The window size is typically set to 12 data points.
     The normalize switch allows to learn and apply a standard scaler prior to computing the anomaly score.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, normalize, output_item):
        super().__init__(input_item, windowsize, normalize, output_item)

        self.whoami = 'FFTV2'
        self.normalizer = FFT_normalizer

        logger.debug('FFT')

    def feature_extract(self, temperature):
        logger.debug(self.whoami + ': feature extract')

        #slices_ = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)
        slices_ = view_as_windows(temperature, self.windowsize, self.step)

        slicelist = []
        for slice in slices_:
            slicelist.append(fftpack.rfft(slice))

        return np.stack(slicelist, axis=0)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze", ))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12."))

        inputs.append(UISingle(name='normalize', datatype=bool, description='Flag for normalizing data.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="output_item", datatype=float,
                                           description="Anomaly score (FFTbasedGeneralizedAnomalyScore)", ))
        return (inputs, outputs)


class SaliencybasedGeneralizedAnomalyScoreV2(GeneralizedAnomalyScoreV2):
    """
    An unsupervised anomaly detection function.
     Based on salient region detection models,
         it uses fast fourier transform to reconstruct a signal using the salient features of a the signal.
     It applies GeneralizedAnomalyScore to the reconstructed signal.
     The function moves a sliding window across the data signal and applies its analysis to each window.
     The window size is typically set to 12 data points.
     The normalize switch allows to learn and apply a standard scaler prior to computing the anomaly score.
     Try several anomaly detectors on your data and use the one that fits your data best.
    """

    def __init__(self, input_item, windowsize, normalize, output_item):
        super().__init__(input_item, windowsize, normalize, output_item)

        self.whoami = 'SaliencyV2'
        self.saliency = Saliency(windowsize, 0, 0)
        self.normalizer = Saliency_normalizer

        logger.debug('Saliency')

    def feature_extract(self, temperature):
        logger.debug(self.whoami + ': feature extract')

        temperature_saliency = self.saliency.transform_spectral_residual(temperature)

        #slices = skiutil.view_as_windows(temperature_saliency, window_shape=(self.windowsize,), step=self.step)
        slices = view_as_windows(temperature, self.windowsize, self.step)

        return slices

    def execute(self, df):
        df_copy = super().execute(df)

        msg = "SaliencybasedGeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze"))

        inputs.append(UISingle(name="windowsize", datatype=int,
                               description="Size of each sliding window in data points. Typically set to 12.", ))

        inputs.append(UISingle(name='normalize', datatype=bool, description='Flag for normalizing data.'))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="output_item", datatype=float,
                                           description="Anomaly score (SaliencybasedGeneralizedAnomalyScore)", ))
        return (inputs, outputs)


#######################################################################################
# Regressors
#######################################################################################

class BayesRidgeRegressor(BaseEstimatorFunction):
    """
    Linear regressor based on a probabilistic model as provided by sklearn
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True
    num_rounds_per_estimator = 3

    def BRidgePipeline(self):
        steps = [('scaler', StandardScaler()), ('bridge', linear_model.BayesianRidge(compute_score=True))]
        return Pipeline(steps)

    def set_estimators(self):
        params = {}
        self.estimators['bayesianridge'] = (self.BRidgePipeline, params)

        logger.info('Bayesian Ridge Regressor start searching for best model')

    def __init__(self, features, targets, predictions=None, deviations=None):
        super().__init__(features=features, targets=targets, predictions=predictions, stddev=True)
        if deviations is not None:
            self.pred_stddev = deviations

        self.experiments_per_execution = 1
        self.auto_train = True
        self.correlation_threshold = 0
        self.stop_auto_improve_at = -2

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities) + ' predicting ' + str(self.targets) + ' from ' + str(self.features) +\
                     ' to appear in ' + str(self.predictions) + ' with confidence interval ' + str(self.pred_stddev))

        missing_cols = [x for x in self.predictions + self.pred_stddev if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            try:
                #check_array(df_copy.loc[[entity]][self.features].values, allow_nd=2)
                logger.debug('check passed')
                dfe = super()._execute(df_copy.loc[[entity]], entity)
                print(df_copy.columns)

                print('BayesianRidge: Entity ', entity, ' Type of pred, stddev arrays ',
                      type(dfe[self.predictions]), type(dfe[self.pred_stddev].values))

                #print('BayesianRidge: Entity ', entity, ' stddev elements', dfe[self.pred_stddev].values)
                dfe.fillna(0, inplace=True)

                df_copy.loc[entity, self.predictions] = dfe[self.predictions]
                df_copy.loc[entity, self.pred_stddev] = dfe[self.pred_stddev]

                print(df_copy.columns)
            except Exception as e:
                logger.info('Bayesian Ridge regressor for entity ' + str(entity) + ' failed with: ' + str(e))
                df_copy.loc[entity, self.predictions] = 0
                df_copy.loc[entity, self.pred_stddev] = 0
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True, output_item='deviations',
                                  is_output_datatype_derived=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))

        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)


class GBMRegressor(BaseEstimatorFunction):
    """
    Regressor based on gradient boosting method as provided by lightGBM
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def GBMPipeline(self):
        steps = [('scaler', StandardScaler()), ('gbm', lightgbm.LGBMRegressor())]
        return Pipeline(steps=steps)

    def set_estimators(self):
        # gradient_boosted
        self.estimators['light_gradient_boosted_regressor'] = (self.GBMPipeline, self.params)
        logger.info('GBMRegressor start searching for best model')

    def __init__(self, features, targets, predictions=None, n_estimators=None, num_leaves=None, learning_rate=None,
                 max_depth=None):
        super().__init__(features=features, targets=targets, predictions=predictions, keep_current_models=True)
        self.experiments_per_execution = 1
        self.correlation_threshold = 0
        self.auto_train = True

        self.num_rounds_per_estimator = 1
        self.parameter_tuning_iterations = 1
        self.cv = 1

        if n_estimators is not None or num_leaves is not None or learning_rate is not None:
            self.params = {'gbm__n_estimators': [n_estimators], 'gbm__num_leaves': [num_leaves],
                           'gbm__learning_rate': [learning_rate], 'gbm__max_depth': [max_depth], 'gbm__verbosity': [2]}
        else:
            self.params = {'gbm__n_estimators': [500], 'gbm__num_leaves': [50], 'gbm__learning_rate': [0.001],
                           'gbm__verbosity': [2]}

        self.stop_auto_improve_at = -2

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            # per entity - copy for later inplace operations
            try:
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.predictions] = dfe[self.predictions]

        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        inputs.append(
            UISingle(name='n_estimators', datatype=int, required=False, description=('Max rounds of boosting')))
        inputs.append(
            UISingle(name='num_leaves', datatype=int, required=False, description=('Max leaves in a boosting tree')))
        inputs.append(UISingle(name='learning_rate', datatype=float, required=False, description=('Learning rate')))
        inputs.append(
            UISingle(name='max_depth', datatype=int, required=False, description=('Cut tree to prevent overfitting')))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)


class SimpleRegressor(BaseEstimatorFunction):
    """
    Regressor based on stochastic gradient descent and gradient boosting method as provided by sklearn
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True
    num_rounds_per_estimator = 3

    def GBRPipeline(self):
        steps = [('scaler', StandardScaler()), ('gbr', ensemble.GradientBoostingRegressor)]
        return Pipeline(steps)

    def SGDPipeline(self):
        steps = [('scaler', StandardScaler()), ('sgd', linear_model.SGDRegressor)]
        return Pipeline(steps)

    def set_estimators(self):
        # gradient_boosted
        params = {'n_estimators': [100, 250, 500, 1000], 'max_depth': [2, 4, 10], 'min_samples_split': [2, 5, 9],
                  'learning_rate': [0.01, 0.02, 0.05], 'loss': ['ls']}
        self.estimators['gradient_boosted_regressor'] = (ensemble.GradientBoostingRegressor, params)
        logger.info('SimpleRegressor start searching for best model')

    def __init__(self, features, targets, predictions=None, n_estimators=None, num_leaves=None, learning_rate=None,
                 max_depth=None):
        super().__init__(features=features, targets=targets, predictions=predictions)

        self.experiments_per_execution = 1
        self.auto_train = True
        self.correlation_threshold = 0

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            try:
                check_array(df_copy.loc[[entity]][self.features].values)
                dfe = super()._execute(df_copy.loc[[entity]], entity)
                df_copy.loc[entity, self.predictions] = dfe[self.predictions]

            except Exception as e:
                logger.info('GBMRegressor for entity ' + str(entity) + ' failed with: ' + str(e))
                df_copy.loc[entity, self.predictions] = 0

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


class SimpleAnomaly(BaseRegressor):
    """
    A supervised anomaly detection function.
     Uses a regression model to predict the value of target data items based on dependent data items or features.
     Then, it compares the actual value to the predicted valueand generates an alert when the difference falls outside of a threshold.
    """

    # class variables
    train_if_no_model = True
    num_rounds_per_estimator = 3

    def __init__(self, features, targets, threshold, predictions=None, alerts=None):
        super().__init__(features=features, targets=targets, predictions=predictions)
        if alerts is None:
            alerts = ['%s_alert' % x for x in self.targets]
        self.alerts = alerts
        self.threshold = threshold
        self.correlation_threshold = 0

    def execute(self, df):

        try:
            df_new = super().execute(df)
            df = df_new
            for i, t in enumerate(self.targets):
                prediction = self.predictions[i]
                df['_diff_'] = (df[t] - df[prediction]).abs()
                alert = AlertHighValue(input_item='_diff_', upper_threshold=self.threshold, alert_name=self.alerts[i])
                alert.set_entity_type(self.get_entity_type())
                df = alert.execute(df)
        except Exception as e:
            logger.info('Simple Anomaly failed with: ' + str(e))

        return df

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        inputs.append(UISingle(name='threshold', datatype=float,
                               description=('Threshold for firing an alert. Expressed as absolute value not percent.')))
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutMulti(name='alerts', datatype=bool, cardinality_from='targets', is_datatype_derived=False, ))

        return (inputs, outputs)

#######################################################################################
# Forecasting
#######################################################################################

class FeatureBuilder(BaseTransformer):

    def __init__(self, features, lag, method, lagged_features):
        super().__init__()

        self.features = features
        self.lagged_features = lagged_features

        self.lag = lag   # list of integers (days) to define lags

        self.method = method   #

        self.whoami = 'FeatureBuilder'

        print(self.whoami, self.features, self.lagged_features, self.lag, self.method)


    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.lagged_features if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            # per entity - copy for later inplace operations
            try:
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
                dfe = df_copy.loc[[entity]]
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            dfroll = dfe[self.features].rolling(window=self.lag, min_periods=0)
            if self.method == 'mean':
                dfe[self.lagged_features] = dfroll.mean().shift(1)
            elif self.method == 'stddev':
                dfe[self.lagged_features] = dfroll.std().shift(1)
            else:
                dfe[self.lagged_features] = dfe[self.features].shift(1)

            #dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.lagged_features] = dfe[self.lagged_features]

        return df_copy


    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True, output_item='lagged_features',
                                  is_output_datatype_derived=True))
        inputs.append(UISingle(name='lag', datatype=int, description='Lag for each input_item'))
        inputs.append(UISingle(name='method', datatype=str, description='Method: Plain, Mean, Stddev'))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)


class GBMForecaster(BaseEstimatorFunction):
    """
    Forecasting regressor based on gradient boosting method as provided by lightGBM
    """
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True

    def GBMPipeline(self):
        steps = [('scaler', StandardScaler()), ('gbm', lightgbm.LGBMRegressor())]
        return Pipeline(steps=steps)

    def set_estimators(self):
        # gradient_boosted
        self.estimators['light_gradient_boosted_regressor'] = (self.GBMPipeline, self.params)
        logger.info('GBMRegressor start searching for best model')

    #
    # return list of new columns for the lagged features and dataframe extended with these new columns
    #
    def lag_features(self, df=None, Train=True):
        print ('lags ' + str(self.lags) + '  lagged_features ' + str(self.lagged_features) + ' Train mode: ' + str(Train))
        create_feature_triplets = []
        new_features = []

        if self.lags is None or self.lagged_features is None:
            return (new_features, None)

        for lagged_feature in self.lagged_features:
            for lag in self.lags:
                # collect triple of new column, original column and lag
                if Train:
                    create_feature_triplets.append((lagged_feature + '_' + str(lag), lagged_feature, lag))
                else:
                    create_feature_triplets.append((lagged_feature + '_' + str(lag), lagged_feature, lag - self.forecast))

                new_features.append(lagged_feature + '_' + str(lag))

        # add day of week and month of year as two feature pairs
        new_features = np.concatenate((new_features, ['_DayOfWeekCos_', '_DayOfWeekSin_', '_DayOfYearCos_', '_DayOfYearSin_']))

        if df is not None:
            df_copy = df.copy()
            missing_cols = [x[0] for x in create_feature_triplets if x not in df_copy.columns]
            for m in missing_cols:
                df_copy[m] = None

            # I hope I can do that for all entities in one fell swoop
            for new_feature in create_feature_triplets:
                df_copy[new_feature[0]] = df[new_feature[1]].shift(new_feature[2])

            # get rid of NaN as result of shifting columns
            df_copy.dropna(inplace=True)

            # add day of week and month of year as two feature pairs
            # operate on simple timestamp index
            df_copy['_DayOfWeekCos_'] = np.cos(df_copy.index.get_level_values(1).dayofweek / 7)
            df_copy['_DayOfWeekSin_'] = np.sin(df_copy.index.get_level_values(1).dayofweek / 7)
            df_copy['_DayOfYearCos_'] = np.cos(df_copy.index.get_level_values(1).dayofyear / 365)
            df_copy['_DayOfYearSin_'] = np.sin(df_copy.index.get_level_values(1).dayofyear / 365)

            #df = df[df[df.columns.intersection(new_features)].notna()]   # drop NaNs
        else:
            df_copy = df

        return (new_features, df_copy)


    def __init__(self, features, targets, predictions=None, lags=None):
        #
        # from https://github.com/ashitole/Time-Series-Project/blob/main/Auto-Arima%20and%20LGBM.ipynb
        #   as taken from https://www.kaggle.com/rohanrao/ashrae-half-and-half
        #
        n_estimators = 500
        num_leaves = 40
        #learning_rate = 0.001
        learning_rate = 0.2   # default 0.001
        feature_fraction = 0.85  # default 1.0
        reg_lambda = 2  # default 0
        metric = "rmse"
        max_depth = -1
        self.lagged_features = features
        self.lags = lags

        self.forecast = min(lags)  # forecast = number to shift features back is the negative minimum lag

        newfeatures,_ = self.lag_features()

        super().__init__(features=newfeatures, targets=targets, predictions=predictions, keep_current_models=True)

        self.experiments_per_execution = 1
        self.correlation_threshold = 0
        self.auto_train = True

        self.num_rounds_per_estimator = 1
        self.parameter_tuning_iterations = 1
        self.cv = 1

        if n_estimators is not None or num_leaves is not None or learning_rate is not None:
            self.params = {'gbm__n_estimators': [n_estimators], 'gbm__num_leaves': [num_leaves],
                           'gbm__reg_lambda' : [reg_lambda], 'gbm__feature_fraction': [feature_fraction],
                           'gbm__learning_rate': [learning_rate], 'gbm__max_depth': [max_depth], 'gbm__verbosity': [2]}
        else:
            self.params = {'gbm__n_estimators': [500], 'gbm__num_leaves': [50], 'gbm__learning_rate': [0.001],
                           'gbm__verbosity': [2]}

        self.stop_auto_improve_at = -2

    def execute(self, df):

        #df_copy = df.copy()
        _, df_copy = self.lag_features(df=df, Train=True)

        print('Here 1', type(df_copy))

        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        # make sure to train a model
        for entity in entities:
            # per entity - copy for later inplace operations
            try:
                print (self.features)
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                #print(df_copy.loc[[entity]][self.features].head(20))
                continue

            dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.predictions] = dfe[self.predictions]

        # preserve predictions based on full lag
        df_pred = df_copy[self.predictions].rename(columns={self.predictions[0] : '__forecast__'})

        # use the model for inferencing - with less lag
        strip_features, df_copy = self.lag_features(df=df, Train=False)

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        for entity in entities:
            # per entity - copy for later inplace operations
            try:
                print (self.features)
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                #print(df_copy.loc[[entity]][self.features].head(20))
                continue

            dfe = super()._execute(df_copy.loc[[entity]], entity)
            df_copy.loc[entity, self.predictions] = dfe[self.predictions]

            #df_copy = pd.merge(df_copy, df_pred, left_index=True, right_index=True, how='outer')

        logger.debug('Drop artificial features ' + str(strip_features))
        df_copy.drop(columns = strip_features, inplace=True)

        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        inputs.append(UIMulti(name='lags', datatype=int, description='Comma separated list of lags'))

        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)


#
# following Jake Vanderplas Data Science Handbook
#   https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
#

class KDEAnomalyScore(BaseTransformer):
    """
    A supervised anomaly detection function.
     Uses kernel density estimate to assign an anomaly score
    """
    def __init__(self, threshold, features, targets, predictions=None):
        logger.debug("init KDE Estimator")
        super().__init__()

        self.threshold = threshold
        self.features = features
        self.targets = targets
        self.name = "KDEAnomalyScore"
        self.models = {}
        if predictions is None:
            predictions = ['predicted_%s' % x for x in self.targets]
        self.predictions = predictions

    def get_model_name(self, prefix='model', suffix=None):

        name = []
        if prefix is not None:
            name.append(prefix)

        name.extend([self._entity_type.name, self.name])
        name.extend(self.targets)
        if suffix is not None:
            name.append(suffix)
        name = '.'.join(name)
        return name

    def execute(self, df):

        df_copy = df.copy()
        db = self._entity_type.db

        print('Here 1', type(df_copy))

        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in self.predictions if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        # make sure to train a model
        for entity in entities:
            # check data okay
            try:
                print (self.features)
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            # per entity - copy for later inplace operations
            model_name = self.get_model_name(suffix=entity)
            kde_model = None
            try:
                kde_model = db.model_store.retrieve_model(model_name)
                logger.info('load model %s' % str(kde_model))
            except Exception as e:
                logger.error('Model retrieval failed with ' + str(e))
                pass

            xy = np.hstack([df_copy.loc[[entity]][self.features].values, df_copy.loc[[entity]][self.targets].values])

            # train new model
            if kde_model is None:

                # all variables should be continuous
                kde_model = KDEMultivariate(xy, var_type= "c" * (len(self.features) + len(self.targets)))
                logger.debug('Created KDE ' + str(kde_model))

                try:
                    db.model_store.store_model(model_name, kde_model)
                except Exception as e:
                    logger.error('Model store failed with ' + str(e))
                    pass

            self.models[entity] = kde_model

            predictions = kde_model.pdf(xy)
            #predictions[predictions < SmallEnergy] = SmallEnergy
            #predictions = self.threshold / predictions
            df_copy.loc[entity, self.predictions] = predictions

        return df_copy


    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingle(name="threshold", datatype=float,
                               description="Probability threshold for outliers. Typically set to 10e-6.", required=True))
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)

'''
    def fit(self, X, y):
        xy = np.vstack(X, y).T
        self.kde = KDEMultivariate(xy, var_type='cc')
        return self

    def predict_proba(self, X):
        logprobs = np.vstack([model.score_samples(X)
                              for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
'''

#######################################################################################
# Variational Autoencoder
#   to approximate probability distribution of targets with respect to features
#######################################################################################
# from https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/
#   usual ELBO with standard prior N(0,1), standard reparametrization

# helper function
def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2


class VI(nn.Module):
    def __init__(self, prior_mu=0.0, prior_sigma=1.0, beta=1.0):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.beta = beta
        self.onnx_session = None
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 50),    # more parameters for sigma
            nn.ReLU(),
            nn.Linear(50, 35),
            nn.ReLU(),
            nn.Linear(35, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    # draw from N(mu, sigma)
    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    # sample from the one-dimensional normal distribution N(mu, exp(log_var))
    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var

    def elbo(self, y_pred, y, mu, log_var):
        # likelihood of observing y given Variational mu and sigma
        likelihood = ll_gaussian(y, mu, log_var)

        # prior probability of y_pred N(0,1)
        log_prior = ll_gaussian(y_pred, self.prior_mu, torch.log(torch.tensor(self.prior_sigma)))

        # variational probability of y_pred
        log_p_q = ll_gaussian(y_pred, mu, log_var)

        # by taking the mean we approximate the expectation according to the law of large numbers
        #converge = np.sqrt(torch.var(likelihood + self.beta * (log_prior - log_p_q)) / likelihood.shape[0])
        #if converge > 0.1:
        #    logger.debug('Elbo: approximating expectation ' + str(converge))
        return (likelihood + self.beta * (log_prior - log_p_q)).mean()

    # Minimizing negative ELBO
    def det_loss(self, y_pred, y, mu, log_var):
        return -elbo(y_pred, y, mu, log_var)


class VIAnomalyScore(BaseTransformer):
    """
    A supervised anomaly detection function.
     Uses VAE based density approximation to assign an anomaly score
    """
    def __init__(self, features, targets, predictions=None, pred_stddev=None):
        logger.debug("init KDE Estimator")
        super().__init__()

        self.features = features
        self.targets = targets
        self.name = "VIAnomalyScore"

        self.models = {}
        self.Input = {}
        self.Output = {}
        self.mu = {}
        self.quantile095 = {}

        if predictions is None:
            predictions = ['predicted_%s' % x for x in self.targets]
        if pred_stddev is None:
            pred_stddev = ['pred_dev_%s' % x for x in self.targets]
        self.predictions = predictions
        self.pred_stddev = pred_stddev

        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        self.beta = 1.0

    def get_model_name(self, prefix='model', suffix=None):

        name = []
        if prefix is not None:
            name.append(prefix)

        name.extend([self._entity_type.name, self.name])
        name.extend(self.targets)
        if suffix is not None:
            name.append(suffix)
        name = '.'.join(name)
        return name

    def execute(self, df):

        df_copy = df.copy()
        db = self._entity_type.db

        print('Here 1', type(df_copy))

        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        missing_cols = [x for x in (self.predictions + self.pred_stddev) if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        # make sure to train a model
        for entity in entities:
            # check data okay
            try:
                print (self.features)
                check_array(df_copy.loc[[entity]][self.features].values, allow_nd=True)
            except Exception as e:
                logger.error(
                    'Found Nan or infinite value in feature columns for entity ' + str(entity) + ' error: ' + str(e))
                continue

            # per entity - copy for later inplace operations
            model_name = self.get_model_name(suffix=entity)
            vi_model = None
            try:
                vi_model = db.model_store.retrieve_model(model_name)
                logger.info('load model %s' % str(vi_model))
            except Exception as e:
                logger.error('Model retrieval failed with ' + str(e))
                pass

            xy = np.hstack([df_copy.loc[[entity]][self.features].values, df_copy.loc[[entity]][self.targets].values])

            # TODO: assumption is cardinality of One for features and targets !!!
            ind = np.lexsort((xy[:,1], xy[:,0]))
            ind_r = np.argsort(ind)

            self.Input[entity] = xy[ind][:,0]

            X = torch.tensor(xy[ind][:,0].reshape(-1,1), dtype=torch.float)
            Y = torch.tensor(xy[ind][:,1].reshape(-1,1), dtype=torch.float)

            # train new model
            if vi_model is None:

                # all variables should be continuous
                epochs = 1500
                learning_rate = 0.005

                vi_model = VI(self.prior_mu, self.prior_sigma, self.beta)   # default: beta 1, prior N(0,1)

                optim = torch.optim.Adam(vi_model.parameters(), lr=learning_rate)

                for epoch in range(epochs):
                    optim.zero_grad()
                    y_pred, mu, log_var = vi_model(X)
                    #loss = det_loss(y_pred, Y, mu, log_var)
                    loss = -vi_model.elbo(y_pred, Y, mu, log_var)
                    if epoch % 10 == 0:
                        logger.debug('Epoch: ' + str(epoch) + ', Loss: ' + str(loss.item()))
                    loss.backward()
                    grad_norm = 0
                    optim.step()

                logger.debug('Created VAE ' + str(vi_model))

                try:
                    db.model_store.store_model(model_name, vi_model)
                except Exception as e:
                    logger.error('Model store failed with ' + str(e))
                    pass

            self.models[entity] = vi_model

            mu = None
            q1 = None
            with torch.no_grad():
                mu_and_log_sigma = vi_model(X)
                mue = mu_and_log_sigma[1]
                sigma = torch.exp(0.5 * mu_and_log_sigma[2]) + 1e-5
                mu = sp.stats.norm.ppf(0.5, loc=mue, scale=sigma).reshape(-1,)
                q1 = sp.stats.norm.ppf(0.95, loc=mue, scale=sigma).reshape(-1,)
                self.mu[entity] = mu
                self.quantile095[entity] = q1

            df_copy.loc[entity, self.predictions] = mu[ind_r]
            df_copy.loc[entity, self.pred_stddev] = q1[ind_r]

        return df_copy


    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True,
                                  output_item='pred_stddev', is_output_datatype_derived=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True,
                                  output_item='predictions', is_output_datatype_derived=True))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)

#######################################################################################
# Crude change point detection
#######################################################################################

def make_histogram(t, bins):
    rv = ''
    if t is None:
        logger.warning('make_histogram encountered None')
        return rv
    logger.info('make_histogram ' + str(type(t)) + ' ' + str(t.shape))
    if np.isnan(t).any():
        logger.warning('make_histogram encountered NaN')
        return rv
    try:
        tv = minmax_scale(t.values)
        hist = np.histogram(tv, bins=bins, density=True)
        logger.info('make_histogram returns ' + str(hist))
        rv = str(hist[0])
    except Exception as e:
        logger.warning('make_histogram np.hist failed with ' + str(e))
    return rv


class HistogramAggregator(BaseSimpleAggregator):
    """
    The docstring of the function will show as the function description in the UI.
    """

    def __init__(self, source=None, bins=None):

        self.input_item = source
        if bins is None:
            self.bins = 15
        else:
            self.bins = int(bins)

    def execute(self, group):
        #
        # group is a series
        #   when calling agg(<aggregator functions>) for each element of the group dictionary
        #   df_input.groupby([pd.Grouper(freq='1H', level='timestamp'), pd.Grouper(level='deviceid')])
        #
        return make_histogram(group, self.bins)

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(name='source', datatype=float,
                                   description='Choose the data items that you would like to aggregate'))
        # output_item='name', is_output_datatype_derived=True))
        inputs.append(UISingle(name='bins', datatype=int, description='Histogram bins - 15 by default'))

        outputs = []
        outputs.append(UIFunctionOutSingle(name='name', datatype=str, description='Histogram encoded as string'))
        return (inputs, outputs)
