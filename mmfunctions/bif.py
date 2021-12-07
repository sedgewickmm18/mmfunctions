# *****************************************************************************
# Â© Copyright IBM Corp. 2018.  All Rights Reserved.
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

import datetime as dt
import logging
import re
import time
import warnings
from collections import OrderedDict

import numpy as np
import scipy as sp
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles

#from sqlalchemy import String

from iotfunctions.base import (BaseTransformer, BaseEvent, BaseSCDLookup, BaseSCDLookupWithDefault, BaseMetadataProvider,
                               BasePreload, BaseDatabaseLookup, BaseDataSource, BaseDBActivityMerge, BaseSimpleAggregator)
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti, UIMulti, UIExpression,
                             UIText, UIParameters)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False


class AggregateWithExpression(BaseSimpleAggregator):
    """
    Create aggregation using expression. The calculation is evaluated for
    each data_item selected. The data item will be made available as a
    Pandas Series. Refer to the Pandas series using the local variable named
    "x". The expression must return a scalar value.

    Example:

    x.max() - x.min()

    """

    def __init__(self, source=None, expression=None, name=None):
        super().__init__()
        logger.info('AggregateWithExpression _init')

        self.source = source
        self.expression = expression
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UIMultiItem(name='source', datatype=None,
                                  description=('Choose the data items that you would like to aggregate'),
                                  output_item='name', is_output_datatype_derived=True))

        inputs.append(UIExpression(name='expression', description='Paste in or type an AS expression'))

        return (inputs, [])

    def aggregate(self, x):
        return eval(self.expression)

    def execute(self, x):
        logger.info('Execute AggregateWithExpression')
        print('Source ', self.source, 'Expression ', self.expression, 'Name ', self.name)
        y = eval(self.expression)
        logger.info('AggregateWithExpression returns ' + str(y))
        return y


class AggregateTimeInState(BaseSimpleAggregator):
    """
    Creates aggregation from the output of StateTimePreparation, a string
    encoded pair of a state change variable (-1 for leaving the state,
    0 for no change, 1 for entering the state) together with a unix epoch
    timestamp.
    It computes the overall number of seconds spent in a particular state.
    """

    def __init__(self, source=None, name=None):
        super().__init__()
        logger.info('AggregateTimeInState _init')

        self.source = source
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(name='source', datatype=None,
                                  description='Output of StateTimePreparation to aggregate over'))

        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='name', datatype=float,
                                description='Overall amount of seconds spent in a particular state'))

        return (inputs, outputs)

    def execute(self, group):
        logger.info('Execute AggregateTimeInState')
        #print('Source ', self.source,  'Name ', self.name, ' Index ', group.index)

        lg = group.size
        if lg == 0:
            return 0

        # group_exp[0] = change array, group_exp[1] = timestamps
        try:
            group_exp = group.str.split(pat=',', n=1, expand=True).astype(int)
        except Exception as esplit:
            logger.info('AggregateTimeInState returns 0 due to NaNs')
            return 0

        g0 = group_exp[0].values
        g1 = group_exp[1].values
        logger.debug(str(g0) + ' ' + str(g1))

        logger.debug(str(np.all(g1[:-1] <= g1[1:]) ))

        # adjust for intervals cut in half by aggregation
        '''
        +---------------------------- Interval ------------------------+
        0            1           -1           1           -1           0
          negative     positive     negative     positive     negative
          (ignore)       ADD        (ignore)      ADD         (ignore)

        0            1           -1           1                        0
          (ignore)       ADD        (ignore)      ADD


        0           -1            1          -1            1           0
           ADD         ignore         ADD        ignore       ADD

        0           -1            1          -1                        0
           ADD         ignore         ADD        (ignore)
        '''

        # first non zero index
        nonzeroMin = 0
        nonzeroMax = 0
        try:
            nonzeroMin = np.min(np.nonzero(g0 != 0))
            nonzeroMax = np.max(np.nonzero(g0 != 0))
        except Exception:
            logger.info('AggregateTimeInState all elements zero - returns ' + str(0) + ' seconds, from ' + str(g0.size))
            return 0
            pass

        if nonzeroMin > 0:
            #print('YES1', nonzeroMin, g0[nonzeroMin])
            if g0[nonzeroMin] < 0:
                g0[0] = 1
        else:
            #print('NO 1', nonzeroMin, g0[nonzeroMin])
            if g0[0] < 0:
                g0[0] = 0

        if nonzeroMax > 0:
            #print('YES2', nonzeroMax, g0[nonzeroMax], g0.size)
            if g0[nonzeroMax] > 0:
                g0[-1] = -1
                # if nonzeroMax is last, ignore
                if g0[nonzeroMax] < 0:
                    g0[-1] = 0

        #y = abs((g0 * g1).sum())
        y = g1.sum()
        if y < 0: y = 0
        logger.info('AggregateTimeInState returns ' + str(y) + ' seconds, computed from ' + str(g0.size))
        return y

class StateTimePreparation(BaseTransformer):
    '''
    Together with AggregateTimeInState StateTimePreparation
    calculates the amount of time a selected metric has been in a
    particular state.
    StateTimePreparation outputs an encoded pair of a state change
    variable (-1 for leaving the state, 0 for no change,
     1 for entering the state) together with a unix epoch
    timestamp.
    The condition for the state change is given as binary operator
    together with the second argument, for example
    ">= 37"  ( for fever) or "=='running'" (for process states)
    '''
    def __init__(self, source=None, state_name=None, name=None):
        super().__init__()
        logger.info('StateTimePrep _init')

        self.source = source
        self.state_name = state_name
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UISingleItem(name='source', datatype=float,
                                  description='Data item to compute the state change array from'))
        inputs.append(UISingle(name='state_name', datatype=str,
                              description='Condition for the state change array computation'))

        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='name', datatype=str, description='State change array output'))

        return (inputs, outputs)

    def _calc(self, df):
        logger.info('Execute StateTimePrep per entity')

        index_names = df.index.names
        ts_name = df.index.names[1]  # TODO: deal with non-standard dataframes (no timestamp)

        logger.info('Source ', self.source, 'state_name ', self.state_name, 'Name ', self.name)
        #df[self.name] = (df[self.source] == self.state_name).astype(int).diff().fillna(1).astype(int)
        df_copy = df.reset_index()

        # pair of +- seconds and regular timestamp
        v1 = eval("df_copy[self.source] " + self.state_name).astype(int).diff().values.astype(int)
        #v1 = (df_copy[self.source] > 50).astype(int).diff().values.astype(int)

        logger.info('HERE')

        # first element is NaN - pretend a state change
        if v1.size > 0:
            v1[0] = 0
            try:
                nonzero = np.min(np.nonzero(v1))
                if v1[nonzero] > 0:
                    v1[0] = -1
                else:
                    v1[0] = 1
            except Exception:
                logger.debug('No Non-Zero')
                # no non zero element
                pass

        logger.info('HERE2')
        # if last element is 0 - pretend a state change
        if v1.size > 0:
            if v1[-1] == 0:
                try:
                    # last nonzero element
                    nonzero = np.max(np.nonzero(v1))
                    if v1[nonzero] > 0:
                        v1[-1] = -1
                    else:
                        v1[-1] = 1
                except Exception:
                    logger.debug('No Non-Zero 2')
                    # no non zero element
                    pass

        logger.info('HERE3')
        logger.info(str(v1))

        df_copy['__intermediate1__'] = v1
        df_copy['__intermediate2__'] = (df_copy[ts_name].astype(int)// 1000000000) * v1

        df_copy[self.name] = df_copy['__intermediate1__'].map(str) + ',' + df_copy['__intermediate2__'].map(str)

        df_copy.drop(columns=['__intermediate1__','__intermediate2__'], inplace=True)
        #df_copy[self.name] = (v1 * df_copy[ts_name].astype(int)// 1000000000)

        return df_copy.set_index(index_names)


# NaNs and STL are not on good terms so base this class on the Interpolator
class SeasonalDecompose(BaseTransformer):
    """
    Create aggregation using expression. The calculation is evaluated for
    each data_item selected. The data item will be made available as a
    Pandas Series. Refer to the Pandas series using the local variable named
    "x". The expression must return a scalar value.

    Example:

    x.max() - x.min()

    """

    def __init__(self, input_item, windowsize, missing, output_item):
        super().__init__()
        self.input_item = input_item
        self.windowsize = windowsize
        self.missing = missing
        self.output_item = output_item
        logger.info('SeasonalDecompose _init')

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

    def _calc(self, df):
        logger.info('kexecute SeasonalDecompose')

        print(df.index)
        df.to_csv('/tmp/testtest')

        index_names = df.index.names
        ts_name = df.index.names[1]  # TODO: deal with non-standard dataframes (no timestamp)

        df_copy = df.copy()
        df_rst = df.reset_index().set_index(ts_name)

        # deal with string timestamp indices
        if not isinstance(df_rst.index, pd.core.indexes.datetimes.DatetimeIndex):
            df_rst.index = pd.to_datetime(df_rst.index, format="%Y-%m-%d-%H.%M.%S.%f")
            #print(df_copy.index)

        # minimal frequency supported by STL
        df_sample = df_rst[[self.input_item]].resample('H').mean().ffill()
        res = STL(df_sample, robust=True).fit()

        df_new = pd.DataFrame(index=df_rst.index)
        df_new['power'] = np.interp(df_rst.index, res.trend.index, res.trend.values)
        print('Power trend', df_new['power'][0:3])

        df_copy[self.output_item] = df_new['power'].values
        print('Power trend', df_copy[self.output_item][0:3])

        logger.info('Exit SeasonalDecompose')
        return df_copy

class AggregateKDEDensity1d(BaseSimpleAggregator):
    """
    Create aggregation using expression. The calculation is evaluated for
    each data_item selected. The data item will be made available as a
    Pandas Series. Refer to the Pandas series using the local variable named
    "x". The expression must return a scalar value.

    Example:

    x.max() - x.min()

    """

    def __init__(self, source=None, alpha=0.995, name=None):
        super().__init__()
        logger.info('AggregateKDEDensity1d _init')

        self.source = source
        self.alpha = alpha
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UIMultiItem(name='source', datatype=None,
                                  description=('Choose the data items that you would like to aggregate'),
                                  output_item='name', is_output_datatype_derived=True))

        inputs.append(UIExpression(name='alpha', description='Quantile level - default 0.995'))

        return (inputs, [])

    def execute(self, group):
        logger.info('Execute AggregateKDEDensity1d')

        if group.size == 0:
            return 0

        X = group.values.reshape(-1,1)

        # set up kernel density
        kde = KernelDensity(kernel='gaussian')
        kde.fit(X)

        # apply it and compute the log density for the observed data
        kde_X = kde.score_samples(X)

        # cut point
        tau_kde = mquantiles(kde_X, 1. - self.alpha)

        # locate inliers and outliers
        outliers = np.nonzero(kde_X < tau_kde)
        #outliers = outliers.flatten()
        inliers = np.nonzero(kde_X >= tau_kde)
        #inliers = inliers.flatten()

        logger.info('AggregateKDEDensity1d: size: ' + str(len(X)) + ' inliers: ' + str(len(inliers)) + ' outliers: ' + str(len(outliers)))

        # inliers provides a lower bound
        lower_bound = 0
        try:
            lower_bound = np.max(X[inliers])
        except Exception as e:
            logger.info('Establishing lower bound failed with ' + str(e))

        print(lower_bound)

        raw_threshold = 0
        try:
            high_outliers = np.nonzero(X[outliers] > lower_bound)
            raw_threshold = np.min(X[high_outliers])
        except Exception as ee:
            logger.info('Establishing threshold failed with ' + str(ee))

        #print('Source ', self.source, 'Expression ', self.expression, 'Name ', self.name)
        logger.info('AggregateKDEDensity1d returns ' + str(raw_threshold))

        return raw_threshold

