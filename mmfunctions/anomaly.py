# *****************************************************************************
# © Copyright IBM Corp. 2018-2020.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

'''
The Built In Functions module contains preinstalled functions
'''

import re
import datetime as dt
import numpy as np
import scipy as sp

#  for Spectral Analysis
from scipy import signal, fftpack
# from scipy.stats import energy_distance
from sklearn import metrics
from sklearn.covariance import MinCovDet

#   for KMeans
#  import skimage as ski
from skimage import util as skiutil  # for nifty windowing
from pyod.models.cblof import CBLOF

# for gradient boosting
import lightgbm

# import re
import pandas as pd
import logging
# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEvent, BaseEstimatorFunction)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti,
                             UIExpression)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'
_IS_PREINSTALLED = False

FrequencySplit = 0.3
DefaultWindowSize = 12
SmallEnergy = 1e-20

# KMeans_normalizer = 100 / 1.3
KMeans_normalizer = 1
Spectral_normalizer = 100 / 2.8
FFT_normalizer = 1
Saliency_normalizer = 1
Generalized_normalizer = 1 / 300


def custom_resampler(array_like):
    # initialize
    if 'gap' not in dir():
        gap = 0

    if (array_like.values.size > 0):
        gap = 0
        return 0
    else:
        gap += 1
        return gap


def min_delta(df):
    # minimal time delta for merging

    if len(df.index.names) > 1:
        df2 = df.copy()
        print(df.index.size)
        df2.index = df2.index.droplevel(list(range(1, df.index.size-1)))
    else:
        df2 = df

    try:
        mindelta = df2.index.to_series().diff().min()
    except Exception as e:
        logger.debug('Min Delta error: ' + str(e))
        mindelta = pd.Timedelta('5 seconds')

    if mindelta == dt.timedelta(seconds=0) or pd.isnull(mindelta):
        mindelta = pd.Timedelta('5 seconds')
    return mindelta


def set_window_size_and_overlap(windowsize, trim_value=2*DefaultWindowSize):
    # make sure it is positive and not too large
    trimmed_ws = np.minimum(np.maximum(windowsize, 1), trim_value)

    # overlap
    if trimmed_ws == 1:
        ws_overlap = 0
    else:
        ws_overlap = trimmed_ws - np.maximum(trimmed_ws // DefaultWindowSize, 1)

    return trimmed_ws, ws_overlap


def dampen_anomaly_score(array, dampening):

    if dampening is None:
        dampening = 0.9  # gradient dampening

    if dampening >= 1:
        return array

    if dampening < 0.01:
        return array

    # TODO error testing for arrays of size <= 1
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


class SpectralAnomalyScore2(BaseTransformer):
    '''
    Employs spectral analysis to extract features from the time series data and to compute zscore from it
    '''
    def __init__(self, input_item, windowsize, output_item, signal_energy, inv_zscore):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, self.windowoverlap = set_window_size_and_overlap(windowsize)

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item
        self.signal_energy = signal_energy
        self.inv_zscore = inv_zscore

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0
        # df_copy.sort_index()   # NoOp

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
            mindelta = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta))

            #  interpolate gaps - data imputation
            # Size = dfe[[self.input_item]].fillna(0).to_numpy().size
            dfe = dfe.interpolate(method='time')

            # one dimensional time series - named temperature for catchyness
            temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1,)

            logger.debug('Module Spectral, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) +
                         ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(self.output_item) +
                         ', Overlap: ' + str(self.windowoverlap) + ', Inputsize: ' + str(temperature.size))

            if temperature.size <= self.windowsize:
                logger.debug(str(temperature.size) + ' <= ' + str(self.windowsize))
                dfe[self.output_item] = 0.0001
            else:
                logger.debug(str(temperature.size) + str(self.windowsize))

                dfe[self.output_item] = 0.00001
                dfe[self.signal_energy] = 0.00001
                dfe[self.inv_zscore] = 0.00001

                try:
                    # Fourier transform:
                    #   frequency, time, spectral density
                    frequency_temperature, time_series_temperature, spectral_density_temperature = signal.spectrogram(
                        temperature, fs=self.frame_rate, window='hanning',
                        nperseg=self.windowsize, noverlap=self.windowoverlap,
                        detrend='l', scaling='spectrum')

                    # cut off freqencies too low to fit into the window
                    frequency_temperatureb = (frequency_temperature > 2/self.windowsize).astype(int)
                    frequency_temperature = frequency_temperature * frequency_temperatureb
                    frequency_temperature[frequency_temperature == 0] = 1 / self.windowsize

                    signal_energy = np.dot(spectral_density_temperature.T, frequency_temperature)

                    # np.linalg.norm(spectral_density_temperature, axis=0)

                    signal_energy[signal_energy < SmallEnergy] = SmallEnergy
                    inv_signal_energy = np.divide(np.ones(signal_energy.size), signal_energy)

                    dfe[self.output_item] = 0.0002

                    ets_zscore = abs(sp.stats.zscore(signal_energy)) * Spectral_normalizer
                    inv_zscore = abs(sp.stats.zscore(inv_signal_energy))

                    logger.debug('Spectral z-score max: ' + str(ets_zscore.max()) +
                                 ',   Spectral inv z-score max: ' + str(inv_zscore.max()))

                    # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                    #   extend it to cover the full original length
                    dfe[self.output_item] = 0.0005
                    linear_interpolate = sp.interpolate.interp1d(
                        time_series_temperature, ets_zscore, kind='linear', fill_value='extrapolate')

                    dfe[self.output_item] = abs(linear_interpolate(np.arange(0, temperature.size, 1)))

                    linear_interpol_signal = sp.interpolate.interp1d(
                        time_series_temperature, signal_energy, kind='linear', fill_value='extrapolate')

                    dfe[self.signal_energy] = abs(linear_interpol_signal(np.arange(0, temperature.size, 1)))

                    linear_interpol_inv_zscore = sp.interpolate.interp1d(
                        time_series_temperature, inv_zscore, kind='linear', fill_value='extrapolate')

                    dfe[self.inv_zscore] = abs(linear_interpol_inv_zscore(np.arange(0, temperature.size, 1)))

                except Exception as e:
                    logger.error('Spectral failed with ' + str(e))

                # absolute zscore > 3 ---> anomaly
                dfe_orig = pd.merge_asof(dfe_orig, dfe[[self.output_item, self.signal_energy, self.inv_zscore]],
                                         left_index=True, right_index=True, direction='nearest', tolerance=mindelta)

                print(dfe_orig.head(3))

                if self.output_item+'_y' in dfe_orig:
                    zScoreII = dfe_orig[self.output_item+'_y'].to_numpy()
                elif self.output_item in dfe_orig:
                    zScoreII = dfe_orig[self.output_item].to_numpy()
                else:
                    zScoreII = dfe_orig[self.input_item].to_numpy()

                if self.signal_energy+'_y' in dfe_orig:
                    signalII = dfe_orig[self.signal_energy+'_y'].to_numpy()
                else:
                    signalII = dfe_orig[self.signal_energy].to_numpy()

                if self.inv_zscore+'_y' in dfe_orig:
                    inv_zscoreII = dfe_orig[self.inv_zscore+'_y'].to_numpy()
                else:
                    inv_zscoreII = dfe_orig[self.inv_zscore].to_numpy()

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII
                df_copy.loc[idx[entity, :], self.signal_energy] = signalII
                df_copy.loc[idx[entity, :], self.inv_zscore] = inv_zscoreII

        msg = 'SpectralAnomalyScore2'
        self.trace_append(msg)

        return (df_copy)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Column for feature extraction'
                                              ))

        inputs.append(UISingle(
                name='windowsize',
                datatype=int,
                description='Window size for spectral analysis - default 12'
                                              ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Spectral anomaly score (z-Score)'
                ))
        outputs.append(UIFunctionOutSingle(
                name='signal_energy',
                datatype=float,
                description='Signal energy'
                ))
        outputs.append(UIFunctionOutSingle(
                name='inv_zscore',
                datatype=float,
                description='zScore of inverted signal energy'
                ))
        return (inputs, outputs)


class SpectralAnomalyScore(BaseTransformer):
    '''
    Employs spectral analysis to extract features from the time series data and to compute zscore from it
    '''
    def __init__(self, input_item, windowsize, output_item):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, self.windowoverlap = set_window_size_and_overlap(windowsize)

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.whoami = 'Spectral'

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0
        # df_copy.sort_index()   # NoOp

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
            mindelta = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta))

            #  interpolate gaps - data imputation
            # Size = dfe[[self.input_item]].fillna(0).to_numpy().size
            dfe = dfe.interpolate(method='time')

            # one dimensional time series - named temperature for catchyness
            temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1,)

            logger.debug('Module Spectral, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) +
                         ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(self.output_item) +
                         ', Overlap: ' + str(self.windowoverlap) + ', Inputsize: ' + str(temperature.size))

            if temperature.size <= self.windowsize:
                logger.debug(str(temperature.size) + ' <= ' + str(self.windowsize))
                # df_copy.loc[[entity]] = 0.0001
                dfe[self.output_item] = 0.0001
            else:
                logger.debug(str(temperature.size) + str(self.windowsize))

                dfe[self.output_item] = 0.0007
                try:
                    # Fourier transform:
                    #   frequency, time, spectral density
                    frequency_temperature, time_series_temperature, spectral_density_temperature = signal.spectrogram(
                        temperature, fs=self.frame_rate, window='hanning',
                        nperseg=self.windowsize, noverlap=self.windowoverlap,
                        detrend=False, scaling='spectrum')

                    # cut off freqencies too low to fit into the window
                    frequency_temperatureb = (frequency_temperature > 2/self.windowsize).astype(int)
                    frequency_temperature = frequency_temperature * frequency_temperatureb
                    frequency_temperature[frequency_temperature == 0] = 1 / self.windowsize

                    highfrequency_temperature = frequency_temperature.copy()
                    lowfrequency_temperature = frequency_temperature.copy()
                    highfrequency_temperature[highfrequency_temperature <= FrequencySplit] = 0
                    lowfrequency_temperature[lowfrequency_temperature > FrequencySplit] = 0

                    # Compute energy = frequency * spectral density over time in decibel
                    # lowsignal_energy = np.log10(np.maximum(SmallEnergy, np.dot(spectral_density_temperature.T,
                    #                             lowfrequency_temperature)) + SmallEnergy)
                    # highsignal_energy = np.log10(np.maximum(SmallEnergy, np.dot(spectral_density_temperature.T,
                    #                              highfrequency_temperature)) + SmallEnergy)

                    signal_energy = np.dot(spectral_density_temperature.T, frequency_temperature)
                    lowsignal_energy = np.dot(spectral_density_temperature.T, lowfrequency_temperature)
                    highsignal_energy = np.dot(spectral_density_temperature.T, highfrequency_temperature)

                    signal_energy[signal_energy < 0] = 0
                    lowsignal_energy[lowsignal_energy < 0] = 0
                    highsignal_energy[highsignal_energy < 0] = 0

                    # compute the elliptic envelope to exploit Minimum Covariance Determinant estimates
                    #    standardizing

                    signal_energy = (signal_energy - signal_energy.mean())
                    lowsignal_energy = (lowsignal_energy - lowsignal_energy.mean())
                    highsignal_energy = (highsignal_energy - highsignal_energy.mean())

                    twoDimsignal_energy = np.vstack((lowsignal_energy, highsignal_energy)).T
                    logger.debug('lowsignal_energy: ' + str(lowsignal_energy.shape) + ', highsignal_energy:' +
                                 str(highsignal_energy.shape) + 'input' + str(twoDimsignal_energy.shape))

                    # inliers have a score of 1, outliers -1, and 0 indicates an issue with the data
                    dfe[self.output_item] = 0.0002
                    # ellEnv = EllipticEnvelope(random_state=0)

                    # dfe[self.output_item] = 0.0003
                    # ellEnv.fit(twoDimsignal_energy)

                    # compute distance to elliptic envelope
                    # dfe[self.output_item] = 0.0004

                    # ets_zscore = np.maximum(ellEnv.decision_function(twoDimsignal_energy).copy(), -0.1)
                    # ets_zscore = ellEnv.decision_function(twoDimsignal_energy).copy()
                    # ets_zscore = ellEnv.score_samples(twoDimsignal_energy).copy() - ellEnv.offset_
                    ets_zscore = abs(sp.stats.zscore(signal_energy)) * Spectral_normalizer

                    # ets_zscore = (-ellEnv.offset_) ** 0.33 - (-ets_zscore) ** 0.33

                    logger.debug('Spectral z-score max: ' + str(ets_zscore.max()))

                    # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                    #   extend it to cover the full original length
                    dfe[self.output_item] = 0.0005
                    linear_interpolate = sp.interpolate.interp1d(
                        time_series_temperature, ets_zscore, kind='linear', fill_value='extrapolate')

                    dfe[self.output_item] = 0.0006
                    zscoreI = linear_interpolate(np.arange(0, temperature.size, 1))

                    dfe[self.output_item] = zscoreI

                except Exception as e:
                    logger.error('Spectral failed with ' + str(e))

                # absolute zscore > 3 ---> anomaly
                dfe_orig = pd.merge_asof(dfe_orig, dfe[self.output_item],
                                         left_index=True, right_index=True, direction='nearest', tolerance=mindelta)

                if self.output_item+'_y' in dfe_orig:
                    zScoreII = dfe_orig[self.output_item+'_y'].to_numpy()
                elif self.output_item in dfe_orig:
                    zScoreII = dfe_orig[self.output_item].to_numpy()
                else:
                    zScoreII = dfe_orig[self.input_item].to_numpy()

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = 'SpectralAnomalyScore'
        self.trace_append(msg)

        return (df_copy)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Column for feature extraction'
                                              ))

        inputs.append(UISingle(
                name='windowsize',
                datatype=int,
                description='Window size for spectral analysis - default 12'
                                              ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Spectral anomaly score (elliptic envelope)'
                ))
        return (inputs, outputs)


class KMeansAnomalyScore(BaseTransformer):
    '''
    Employs kmeans on windowed time series data and to compute
     an anomaly score from proximity to centroid's center points
    '''
    def __init__(self, input_item, windowsize, output_item):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # use 12 by default
        self.windowsize, _ = set_window_size_and_overlap(windowsize)

        # step
        self.step = 1

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

        self.whoami = 'KMeans'

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # interpolate gaps - data imputation
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1,)

        if index_names is not None:
            dfe = dfe.reset_index().set_index(index_names)

        return dfe, temperature

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0
        # df_copy.sort_index() - NoOP

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
            mindelta = min_delta(dfe_orig)

            logger.debug('Timedelta:' + str(mindelta))

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug('Module KMeans, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) +
                         ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(self.output_item) +
                         ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

            if temperature.size > self.windowsize:
                logger.debug(str(temperature.size) + ',' + str(self.windowsize))

                # Chop into overlapping windows
                slices = skiutil.view_as_windows(temperature, window_shape=(self.windowsize,), step=self.step)

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
                    continue

                pred_score = cblofwin.decision_scores_.copy() * KMeans_normalizer

                # length of time_series_temperature, signal_energy and ets_zscore is smaller than half the original
                #   extend it to cover the full original length
                time_series_temperature = np.linspace(
                     self.windowsize//2, temperature.size - self.windowsize//2 + 1,
                     temperature.size - self.windowsize + 1)

                linear_interpolateK = sp.interpolate.interp1d(
                    time_series_temperature, pred_score, kind='linear', fill_value='extrapolate')

                kmeans_scoreI = linear_interpolateK(np.arange(0, temperature.size, 1))

                dfe[self.output_item] = kmeans_scoreI

                # absolute kmeans_score > 1000 ---> anomaly
                dfe_orig = pd.merge_asof(dfe_orig, dfe[self.output_item],
                                         left_index=True, right_index=True, direction='nearest', tolerance=mindelta)

                if self.output_item+'_y' in dfe_orig:
                    zScoreII = dfe_orig[self.output_item+'_y'].to_numpy()
                elif self.output_item in dfe_orig:
                    zScoreII = dfe_orig[self.output_item].to_numpy()
                else:
                    zScoreII = dfe_orig[self.input_item].to_numpy()

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = 'KMeansAnomalyScore'
        self.trace_append(msg)
        return (df_copy)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Column for feature extraction'
                                              ))

        inputs.append(UISingle(
                name='windowsize',
                datatype=int,
                description='Window size for spectral analysis - default 12'
                                              ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Anomaly score (kmeans)'
                ))
        return (inputs, outputs)


class GeneralizedAnomalyScore(BaseTransformer):
    """
    Employs GAM on windowed time series data to compute an anomaly score from the covariance matrix
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__()
        logger.debug(input_item)

        self.whoami = 'GAM'

        self.input_item = input_item

        # use 12 by default
        self.windowsize, self.windowoverlap = set_window_size_and_overlap(windowsize)

        # step
        self.step = 1

        # assume 1 per sec for now
        self.frame_rate = 1

        self.dampening = 1  # dampening - dampen anomaly score

        self.output_item = output_item

        self.normalizer = Generalized_normalizer

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

        # interpolate gaps - data imputation
        if len(dfEntity.index.names) > 1:
            index_names = dfEntity.index.names
            dfe = dfEntity.reset_index().set_index(index_names[0])
        else:
            index_names = None
            dfe = dfEntity

        try:
            dfe = dfe.interpolate(method="time")
        except Exception as e:
            logger.error('Prepare data error: ' + str(e))

        # one dimensional time series - named temperature for catchyness
        temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1,)

        if index_names is not None:
            dfe = dfe.reset_index().set_index(index_names)

        return dfe, temperature

    def feature_extract(self, temperature):

        logger.debug(self.whoami + ': feature extract')

        slices = skiutil.view_as_windows(
            temperature, window_shape=(self.windowsize,), step=self.step
        )
        return slices

    def execute(self, df):

        df_copy = df.copy()
        entities = np.unique(df_copy.index.levels[0])
        logger.debug(str(entities))

        df_copy[self.output_item] = 0
        # df_copy.sort_index()

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
            mindelta = min_delta(dfe_orig)

            # interpolate gaps - data imputation by default
            #   for missing data detection we look at the timestamp gradient instead
            dfe, temperature = self.prepare_data(dfe)

            logger.debug('Module GeneralizedAnomaly, Entity: ' + str(entity) + ', Input: ' + str(self.input_item) +
                         ', Windowsize: ' + str(self.windowsize) + ', Output: ' + str(self.output_item) +
                         ', Overlap: ' + str(self.step) + ', Inputsize: ' + str(temperature.size))

            if temperature.size > self.windowsize:
                logger.debug(str(temperature.size) + "," + str(self.windowsize))

                # NN = GeneralizedAnomalyModel( base_learner=MinCovDet(), fit_function="fit",
                #        predict_function="mahalanobis", score_sign=1,)
                temperature -= np.mean(temperature, axis=0)
                mcd = MinCovDet()

                # Chop into overlapping windows (default) or run through FFT first
                slices = self.feature_extract(temperature)

                pred_score = None

                try:
                    # NN.fit(slices)
                    mcd.fit(slices)

                    # pred_score = NN.decision_function(slices).copy()
                    pred_score = mcd.mahalanobis(slices).copy() * self.normalizer

                except ValueError as ve:

                    logger.info(
                        self.whoami + " GeneralizedAnomalyScore: Entity: "
                        + str(entity) + ", Input: " + str(self.input_item) + ", WindowSize: "
                        + str(self.windowsize) + ", Output: " + str(self.output_item) + ", Step: "
                        + str(self.step) + ", InputSize: " + str(slices.shape)
                        + " failed in the fitting step with \"" + str(ve) + "\" - scoring zero")

                    pred_score = np.zeros(slices.shape[0])
                    pass

                except Exception as e:

                    dfe[self.output_item] = 0
                    logger.error(
                        self.whoami + " GeneralizedAnomalyScore: Entity: "
                        + str(entity) + ", Input: " + str(self.input_item) + ", WindowSize: "
                        + str(self.windowsize) + ", Output: " + str(self.output_item) + ", Step: "
                        + str(self.step) + ", InputSize: " + str(slices.shape)
                        + " failed in the fitting step with " + str(e))
                    continue

                # will break if pred_score is None
                # length of timesTS, ETS and ets_zscore is smaller than half the original
                #   extend it to cover the full original length
                timesTS = np.linspace(
                    self.windowsize // 2,
                    temperature.size - self.windowsize // 2 + 1,
                    temperature.size - self.windowsize + 1,
                )

                logger.debug(self.whoami + '   Entity: ' + str(entity) + ', result shape: ' + str(timesTS.shape) +
                             ' score shape: ' + str(pred_score.shape))

                # timesI = np.linspace(0, Size - 1, Size)
                linear_interpolateK = sp.interpolate.interp1d(
                    timesTS, pred_score, kind="linear", fill_value="extrapolate"
                )

                # kmeans_scoreI = np.interp(timesI, timesTS, pred_score)
                gam_scoreI = linear_interpolateK(np.arange(0, temperature.size, 1))

                dampen_anomaly_score(gam_scoreI, self.dampening)

                dfe[self.output_item] = gam_scoreI

                # absolute kmeans_score > 1000 ---> anomaly

                dfe_orig = pd.merge_asof(
                    dfe_orig,
                    dfe[self.output_item],
                    left_index=True,
                    right_index=True,
                    direction="nearest",
                    tolerance=mindelta,
                )

                if self.output_item + "_y" in dfe_orig:
                    zScoreII = dfe_orig[self.output_item + "_y"].to_numpy()
                elif self.output_item in dfe_orig:
                    zScoreII = dfe_orig[self.output_item].to_numpy()
                else:
                    print(dfe_orig.head(2))
                    zScoreII = dfe_orig[self.input_item].to_numpy()

                idx = pd.IndexSlice
                df_copy.loc[idx[entity, :], self.output_item] = zScoreII

        msg = "GeneralizedAnomalyScore"
        self.trace_append(msg)
        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Column for feature extraction",
            )
        )

        inputs.append(
            UISingle(
                name="windowsize",
                datatype=int,
                description="Window size for Generalized Anomaly analysis - default 12",
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item",
                datatype=float,
                description="Anomaly score (GeneralizedAnomaly)",
            )
        )
        return (inputs, outputs)


class NoDataAnomalyScore(GeneralizedAnomalyScore):
    '''
    Employs generalized anomaly analysis to extract features from the
      gaps in time series data and to compute the elliptic envelope from it
    '''
    def __init__(self, input_item, windowsize, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'NoData'
        self.normalizer = 1

        logger.debug('NoData')

    def prepare_data(self, dfEntity):

        logger.debug(self.whoami + ': prepare Data')

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
            temperature[0] = 10**10
            pass

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
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Column for feature extraction'
                                              ))

        inputs.append(UISingle(
                name='windowsize',
                datatype=int,
                description='Window size for no data spectral analysis - default 12'
                                              ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='No data anomaly score'
                ))
        return (inputs, outputs)


class FFTbasedGeneralizedAnomalyScore(GeneralizedAnomalyScore):
    """
    Employs FFT and GAM on windowed time series data to compute an anomaly score from the covariance matrix
    """

    def __init__(self, input_item, windowsize, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'FFT'
        self.normalizer = FFT_normalizer

        logger.debug('FFT')

    def feature_extract(self, temperature):

        logger.debug(self.whoami + ': feature extract')

        slices_ = skiutil.view_as_windows(
            temperature, window_shape=(self.windowsize,), step=self.step
        )
        slicelist = []
        for slice in slices_:
            slicelist.append(fftpack.rfft(slice))

        # return np.array(slicelist)
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
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Column for feature extraction",
            )
        )

        inputs.append(
            UISingle(
                name="windowsize",
                datatype=int,
                description="Window size for FFT feature based Generalized Anomaly analysis - default 12",
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item",
                datatype=float,
                description="Anomaly score (FFTbasedGeneralizedAnomalyScore)",
            )
        )
        return (inputs, outputs)


class FFTbasedGeneralizedAnomalyScore2(GeneralizedAnomalyScore):
    """
    Employs FFT and GAM on windowed time series data to compute an anomaly score from the covariance matrix
    """

    def __init__(self, input_item, windowsize, dampening, output_item):
        super().__init__(input_item, windowsize, output_item)

        self.whoami = 'FFT dampen'
        self.dampening = dampening
        self.normalizer = FFT_normalizer / dampening

        logger.debug('FFT')

    def feature_extract(self, temperature):

        logger.debug(self.whoami + ': feature extract')

        slices_ = skiutil.view_as_windows(
            temperature, window_shape=(self.windowsize,), step=self.step
        )
        slicelist = []
        for slice in slices_:
            slicelist.append(fftpack.rfft(slice))

        # return np.array(slicelist)
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
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Column for feature extraction",
            )
        )

        inputs.append(
            UISingle(
                name="windowsize",
                datatype=int,
                description="Window size for FFT feature based Generalized Anomaly analysis - default 12",
            )
        )

        inputs.append(
            UISingle(
                name="dampening",
                datatype=float,
                description="Moderate anomaly scores (value <= 1, default 1)",
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item",
                datatype=float,
                description="Anomaly score (FFTbasedGeneralizedAnomalyScore)",
            )
        )
        return (inputs, outputs)


class SaliencybasedGeneralizedAnomalyScore(GeneralizedAnomalyScore):
    """
    Employs Saliency and GAM on windowed time series data to compute an anomaly score from the covariance matrix
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

        slices = skiutil.view_as_windows(
            temperature_saliency, window_shape=(self.windowsize,), step=self.step
        )
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
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Column for feature extraction",
            )
        )

        inputs.append(
            UISingle(
                name="windowsize",
                datatype=int,
                description="Window size for Saliency feature based Generalized Anomaly analysis - default 12",
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item",
                datatype=float,
                description="Anomaly score (FFTbasedGeneralizedAnomalyScore)",
            )
        )
        return (inputs, outputs)

#######################################################################################


class AlertExpressionWithFilter(BaseEvent):
    '''
    Create alerts that are triggered when data values the expression is True
    '''

    def __init__(self, expression, dimension_name, dimension_value, alert_name, **kwargs):
        self.dimension_name = dimension_name
        self.dimension_value = dimension_value
        self.expression = expression
        self.alert_name = alert_name
        logger.info('AlertExpressionWithFilter  dim: ' + dimension_name + '  exp: ' + expression + '  alert: ' +
                    alert_name)
        super().__init__()

    def _calc(self, df):
        '''
        unused
        '''
        return df

    def execute(self, df):
        # c = self._entity_type.get_attributes_dict()
        df = df.copy()
        logger.info('AlertExpressionWithFilter  exp: ' + self.expression + '  input: ' + str(df.columns))

        expr = self.expression

        # if '${}' in expr:
        #    expr = expr.replace("${}", "df['" + self.dimension_name + "']")

        if '${' in expr:
            expr = re.sub(r"\$\{(\w+)\}", r"df['\1']", expr)
            msg = 'Expression converted to %s. ' % expr
        else:
            msg = 'Expression (%s). ' % expr

        self.trace_append(msg)

        expr = str(expr)
        logger.info('AlertExpressionWithFilter  - after regexp: ' + expr)

        try:
            evl = eval(expr)
            n1 = np.where(evl, True, False)
            n2 = np.where(df[self.dimension_name] == self.dimension_value, True, False)
            np_res = np.logical_and(n1, n2)
            logger.info('AlertExpressionWithFilter  shapes ' + str(n1.shape) + ' ' + str(n2.shape) + ' ' +
                        str(np_res.shape) + '  results\n - ' + str(n1) + '\n - ' + str(n2) + '\n - ' + str(np_res))
            df[self.alert_name] = np_res

        except Exception as e:
            logger.info('AlertExpressionWithFilter  eval for ' + expr + ' failed with ' + str(e))
            df[self.alert_name] = None
            pass

        return df

    def get_input_items(self):
        items = set(self.dimension_name)
        items = items | self.get_expression_items(self.expression)
        return items

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name='dimension_name', datatype=str))
        inputs.append(UISingle(name='dimension_value', datatype=str,
                               description='Dimension Filter Value'))
        inputs.append(UIExpression(name='expression',
                                   description="Define alert expression using pandas systax. \
                                                Example: df['inlet_temperature']>50. ${pressure} will be substituted \
                                                with df['pressure'] before evaluation, ${} with df[<dimension_name>]"))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='alert_name', datatype=bool, description='Output of alert function'))
        return (inputs, outputs)

#######################################################################################


class GBMRegressor(BaseEstimatorFunction):

    '''
    Regressor based on gradient boosting method as provided by lightGBM
    '''
    eval_metric = staticmethod(metrics.r2_score)

    # class variables
    train_if_no_model = True
    estimators_per_execution = 1
    num_rounds_per_estimator = 1

    def set_estimators(self):
        # gradient_boosted
        self.estimators['light_gradient_boosted_regressor'] = (lightgbm.LGBMRegressor, self.params)
        logger.info('GBMRegressor start search for best model')

    def __init__(self, features, targets, predictions=None,
                 n_estimators=None, num_leaves=None, learning_rate=None, max_depth=None):
        super().__init__(features=features, targets=targets, predictions=predictions)
        self.experiments_per_execution = 1
        self.auto_train = False
        # if n_estimators is not None or num_leaves is not None or learning_rate is not None or max_depth is not None:
        if n_estimators is not None or num_leaves is not None or learning_rate is not None:
            self.params = {'n_estimators': [n_estimators],
                           'num_leaves': [num_leaves],
                           'learning_rate': [learning_rate],
                           'max_depth': [max_depth],
                           'verbosity': [2]}
        else:
            self.params = {'n_estimators': [500],
                           'num_leaves': [50],
                           'learning_rate': [0.001],
                           'verbosity': [2]}
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
        inputs.append(UISingle(name='n_estimators', datatype=int, required=False,
                               description=('Max rounds of boosting')))
        inputs.append(UISingle(name='num_leaves', datatype=int, required=False,
                               description=('Max leaves in a boosting tree')))
        inputs.append(UISingle(name='learning_rate', datatype=float, required=False,
                               description=('Learning rate')))
        inputs.append(UISingle(name='max_depth', datatype=int, required=False,
                               description=('Cut tree to prevent overfitting')))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)

    @classmethod
    def get_input_items(cls):
        return ['features', 'targets']


class SimpleAnomaly(BaseRegressor):
    '''
    Sample function uses a regression model to predict the value of one or more output
    variables. It compares the actual value to the prediction and generates an alert
    when the difference between the actual and predicted value is outside of a threshold.
    '''

    # class variables
    train_if_no_model = True
    estimators_per_execution = 3
    num_rounds_per_estimator = 3

    def __init__(self, features, targets, threshold, predictions=None, alerts=None):
        super().__init__(features=features, targets=targets, predictions=predictions)
        if alerts is None:
            alerts = ['%s_alert' % x for x in self.targets]
        self.alerts = alerts
        self.threshold = threshold

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
            pass

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


class SimpleRegressor(BaseRegressor):
    '''
    Sample function that predicts the value of a continuous target variable using the selected list of features.
    This function is intended to demonstrate the basic workflow of training, evaluating, deploying
    using a model.
    '''
    # class variables
    train_if_no_model = True
    estimators_per_execution = 3
    num_rounds_per_estimator = 3

    def __init__(self, features, targets, predictions=None):
        super().__init__(features=features, targets=targets, predictions=predictions)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        return (inputs, [])


class AnomalyGeneratorExtremeValue(BaseTransformer):
    '''
    This function generates extreme anomaly.
    '''

    def __init__(self, input_item, factor, size, output_item):
        self.input_item = input_item
        self.output_item = output_item
        self.factor = int(factor)
        self.size = int(size)
        super().__init__()

    def execute(self, df):

        logger.debug('Dataframe shape {}'.format(df.shape))

        entity_type = self.get_entity_type()
        derived_metric_table_name = 'DM_' + entity_type.logical_name
        schema = entity_type._db_schema

        # store and initialize the counts by entity id
        # db = self.get_db()
        db = self._entity_type.db

        raw_dataframe = None
        try:
            query, table = db.query(derived_metric_table_name, schema, column_names='KEY', filters={'KEY': self.output_item})
            raw_dataframe = db.get_query_data(query)
            key = '_'.join([derived_metric_table_name, self.output_item])
            logger.debug('Check for key {} in derived metric table {}'.format(self.output_item, raw_dataframe.shape))
        except Exception as e:
            logger.error('Checking for derived metric table %s failed with %s.' % (str(self.output_item), str(e)))
            key = str(derived_metric_table_name) + str(self.output_item)
            pass

        if raw_dataframe is not None and raw_dataframe.empty:
            # delete old counts if present
            db.model_store.delete_model(key)
            logger.debug('Intialize count for first run')

        counts_by_entity_id = None
        try:
            counts_by_entity_id = db.model_store.retrieve_model(key)
        except Exception as e2:
            logger.error('Counts by entity id not yet initialized - error: ' + str(e2))
            pass

        if counts_by_entity_id is None:
            counts_by_entity_id = {}
        logger.debug('Initial Grp Counts {}'.format(counts_by_entity_id))

        # mark Anomalies
        entity_id = df.index.names[0]
        print(entity_id)
        timeseries = df.reset_index()
        timeseries[self.output_item] = timeseries[self.input_item]

        df_copy = df.copy()
        df_copy[self.output_item] = df_copy[self.input_item]
        entities = np.unique(df_copy.index.levels[0])
        for entity in entities:
            dfe = df_copy.loc[[entity]]
            a = dfe[self.output_item].values  # reference to make life easier
            if a.size < self.factor:
                logger.info('Entity ' + entity + ' has not enough values to inject an extreme value anomaly')
                continue

            a1 = a[:(a.size - a.size % self.factor)]
            a1 = np.reshape(a1, (-1, self.factor)).T
            b = np.random.choice([-1, 1], a1.shape[1])
            print(self.factor, '\n', dfe[self.output_item].values.shape, '\n',
                  a1.shape, '\n', a1[0].shape, '\n', b.shape)

            # use 'local' standard deviation if it exceeds 1 to make sure we're generating an anomaly
            stdvec = np.maximum(np.std(a1, axis=0), np.ones(a1[0].size)) * self.size
            a1[0] = np.multiply(a1[0], np.multiply(b, stdvec))

            # np.copyto(a,a1)
            a[:(a.size - a.size % self.factor)] = a1.T.flatten()
            idx = pd.IndexSlice
            df_copy.loc[idx[entity, :], self.output_item] = a

        # df_grpby = timeseries.groupby('id')
        df_grpby = timeseries.groupby(entity_id)
        for grp in df_grpby.__iter__():

            entity_grp_id = grp[0]
            df_entity_grp = grp[1]
            logger.debug('Group {} Indexes {}'.format(grp[0], df_entity_grp.index))

            count = 0
            # local_std = df_entity_grp.iloc[:10][self.input_item].std()
            if entity_grp_id in counts_by_entity_id:
                count = counts_by_entity_id[entity_grp_id]

            counts_by_entity_id[entity_grp_id] = count + len(df_entity_grp.index) % self.factor

        logger.debug('Final Grp Counts {}'.format(counts_by_entity_id))

        # save the group counts to db
        try:
            db.model_store.store_model(key, counts_by_entity_id)
        except Exception as e3:
            logger.error('Counts by entity id cannot be stored - error: ' + str(e3))
            pass

        # timeseries.set_index(df.index.names, inplace=True)
        # return timeseries
        return df_copy

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Item to base anomaly on'
                                              ))

        inputs.append(UISingle(
                name='factor',
                datatype=int,
                description='Frequency of anomaly e.g. A value of 3 will create anomaly every 3 datapoints',
                default=5
                                              ))

        inputs.append(UISingle(
                name='size',
                datatype=int,
                description='Size of extreme anomalies to be created. e.g. 10 will create 10x size extreme \
                             anomaly compared to the normal variance', default=10
                                              ))

        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Generated Item With Extreme anomalies'
                ))
        return (inputs, outputs)


class AnomalyGeneratorNoData(BaseTransformer):
    '''
    This function generates nodata anomaly.
    '''

    def __init__(self, input_item, width, factor, output_item):
        self.input_item = input_item
        self.output_item = output_item
        self.width = int(width)
        self.factor = int(factor)
        super().__init__()

    def execute(self, df):

        logger.debug('Dataframe shape {}'.format(df.shape))

        entity_type = self.get_entity_type()
        derived_metric_table_name = 'DM_'+entity_type.logical_name
        schema = entity_type._db_schema

        # store and initialize the counts by entity id
        # db = self.get_db()
        db = self._entity_type.db
        query, table = db.query(derived_metric_table_name, schema, column_names='KEY', filters={'KEY': self.output_item})
        raw_dataframe = db.get_query_data(query)
        logger.debug('Check for key {} in derived metric table {}'.format(self.output_item, raw_dataframe.shape))
        key = '_'.join([derived_metric_table_name, self.output_item])

        if raw_dataframe is not None and raw_dataframe.empty:
            # delete old counts if present
            db.model_store.delete_model(key)
            logger.debug('Intialize count for first run')

        counts_by_entity_id = db.model_store.retrieve_model(key)
        if counts_by_entity_id is None:
            counts_by_entity_id = {}
        logger.debug('Initial Grp Counts {}'.format(counts_by_entity_id))

        # mark Anomalies
        timeseries = df.reset_index()
        timeseries[self.output_item] = timeseries[self.input_item]
        df_grpby = timeseries.groupby('id')
        for grp in df_grpby.__iter__():

            entity_grp_id = grp[0]
            df_entity_grp = grp[1]
            logger.debug('Group {} Indexes {}'.format(grp[0], df_entity_grp.index))

            count = 0
            width = self.width
            if entity_grp_id in counts_by_entity_id:
                count = counts_by_entity_id[entity_grp_id][0]
                width = counts_by_entity_id[entity_grp_id][1]

            mark_anomaly = False
            for grp_row_index in df_entity_grp.index:
                count += 1

                if width != self.width or count % self.factor == 0:
                    # start marking points
                    mark_anomaly = True

                if mark_anomaly:
                    timeseries[self.output_item].iloc[grp_row_index] = np.NaN
                    width -= 1
                    logger.debug('Anomaly Index Value{}'.format(grp_row_index))

                if width == 0:
                    # end marking points
                    mark_anomaly = False
                    # update values
                    width = self.width
                    count = 0

            counts_by_entity_id[entity_grp_id] = (count, width)

        logger.debug('Final Grp Counts {}'.format(counts_by_entity_id))

        # save the group counts to db
        db.model_store.store_model(key, counts_by_entity_id)

        timeseries.set_index(df.index.names, inplace=True)
        return timeseries

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Item to base anomaly on'
                                              ))

        inputs.append(UISingle(
                name='factor',
                datatype=int,
                description='Frequency of anomaly e.g. A value of 3 will create anomaly every 3 datapoints',
                default=10
                                              ))

        inputs.append(UISingle(
                name='width',
                datatype=int,
                description='Width of the anomaly created',
                default=5
                                              ))

        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Generated Item With NoData anomalies'
                ))
        return (inputs, outputs)


class AnomalyGeneratorFlatline(BaseTransformer):
    '''
    This function generates flatline anomaly.
    '''

    def __init__(self, input_item, width, factor, output_item):
        self.input_item = input_item
        self.output_item = output_item
        self.width = int(width)
        self.factor = int(factor)
        super().__init__()

    def execute(self, df):

        logger.debug('Dataframe shape {}'.format(df.shape))

        entity_type = self.get_entity_type()
        derived_metric_table_name = 'DM_'+entity_type.logical_name
        schema = entity_type._db_schema

        # store and initialize the counts by entity id
        # db = self.get_db()
        db = self._entity_type.db
        query, table = db.query(derived_metric_table_name, schema, column_names='KEY', filters={'KEY': self.output_item})
        raw_dataframe = db.get_query_data(query)
        logger.debug('Check for key column {} in derived metric table {}'.format(self.output_item, raw_dataframe.shape))
        key = '_'.join([derived_metric_table_name, self.output_item])

        if raw_dataframe is not None and raw_dataframe.empty:
            # delete old counts if present
            db.model_store.delete_model(key)
            logger.debug('Intialize count for first run')

        counts_by_entity_id = db.model_store.retrieve_model(key)
        if counts_by_entity_id is None:
            counts_by_entity_id = {}
        logger.debug('Initial Grp Counts {}'.format(counts_by_entity_id))

        # mark Anomalies
        timeseries = df.reset_index()
        timeseries[self.output_item] = timeseries[self.input_item]
        df_grpby = timeseries.groupby('id')
        for grp in df_grpby.__iter__():

            entity_grp_id = grp[0]
            df_entity_grp = grp[1]
            logger.debug('Group {} Indexes {}'.format(grp[0], df_entity_grp.index))

            count = 0
            width = self.width
            local_mean = df_entity_grp.iloc[:10][self.input_item].mean()
            if entity_grp_id in counts_by_entity_id:
                count = counts_by_entity_id[entity_grp_id][0]
                width = counts_by_entity_id[entity_grp_id][1]
                if count != 0:
                    local_mean = counts_by_entity_id[entity_grp_id][2]

            mark_anomaly = False
            for grp_row_index in df_entity_grp.index:
                count += 1

                if width != self.width or count % self.factor == 0:
                    # start marking points
                    mark_anomaly = True

                if mark_anomaly:
                    timeseries[self.output_item].iloc[grp_row_index] = local_mean
                    width -= 1
                    logger.debug('Anomaly Index Value{}'.format(grp_row_index))

                if width == 0:
                    # end marking points
                    mark_anomaly = False
                    # update values
                    width = self.width
                    count = 0
                    local_mean = df_entity_grp.iloc[:10][self.input_item].mean()

            counts_by_entity_id[entity_grp_id] = (count, width, local_mean)

        logger.debug('Final Grp Counts {}'.format(counts_by_entity_id))

        # save the group counts to db
        db.model_store.store_model(key, counts_by_entity_id)

        timeseries.set_index(df.index.names, inplace=True)
        return timeseries

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=float,
                description='Item to base anomaly on'
                                              ))

        inputs.append(UISingle(
                name='factor',
                datatype=int,
                description='Frequency of anomaly e.g. A value of 3 will create anomaly every 3 datapoints',
                default=10
                                              ))

        inputs.append(UISingle(
                name='width',
                datatype=int,
                description='Width of the anomaly created',
                default=5
                                              ))

        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='Generated Item With Flatline anomalies'
                ))
        return (inputs, outputs)
