# *****************************************************************************
# © Copyright IBM Corp. 2018.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0 license
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
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles

#from sqlalchemy import String

from iotfunctions.base import (BaseTransformer, BaseEvent, BaseSCDLookup, BaseSCDLookupWithDefault, BaseMetadataProvider,
                               BasePreload, BaseDatabaseLookup, BaseDataSource, BaseDBActivityMerge, BaseSimpleAggregator)
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti, UIMulti, UIExpression,
                             UIText, UIParameters, UIStatusFlag)

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

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UIMultiItem(name='source', datatype=None,
                                  description=('Choose the data items that you would like to aggregate'),
                                  output_item='name', is_output_datatype_derived=True))
        inputs.append(UIExpression(name='expression', description='Paste in or type an AS expression'))
        return (inputs, [])

    def execute(self, x):
        logger.info('Execute AggregateWithExpression')
        logger.debug('Source ' + str(self.source) +  'Expression ' +  str(self.expression) + 'Name ' + str(self.name))
        y = eval(self.expression)
        self.log_df_info(y, 'AggregateWithExpression evaluation')
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

        lg = group.size
        if lg == 0:
            logger.info('AggregateTimeInState no elements - returns 0 seconds, from 0')
            return 0.0

        # debug stuff
        #pd.set_option("display.max_rows", 50)
        #logger.info(str(group))

        df_group_exp = group.str.split(pat=',', n=3, expand=True)
        #logger.info(str(df_group_exp))

        gchange = None
        gstate = None
        gtime = None
        try:
            gchange = df_group_exp[0].values.astype(int).copy()
            gstate = df_group_exp[1].values.astype(int).copy()
            gtime = df_group_exp[2].values.astype(int).copy()
        except Exception as esplit:
            logger.info('AggregateTimeInState elements with NaN- returns 0 seconds, from ' + str(gchange.size))
            return 0.0

        # look for state change 2 - start interval for StateTimePrep
        #
        # |  previous run  -1 ...  |2 ..  1 next run    |
        flag = 0
        index = 0
        with np.nditer(gchange, op_flags=['readwrite']) as it:
            for x in it:
                # apparently a StateTimePrep interval start, adjust
                # apparently a StateTimePrep interval start, adjust
                if x == 2:
                    # we haven't seen a statechange yet, so state == statechange
                    if flag == 0:
                        x[...] = gstate[index]
                        x = gstate[index]
                    # we have seen a statechange before, check whether our state is different
                    elif gstate[index] != flag:
                        x[...] = -flag
                        x = -flag
                    # same state as before, just set to zero
                    else:
                        x[...] = 0
                        x = 0
                # no interval start but state change, so change the flag accordingly
                #   if x had been 2 before it is now corrected
                if x != 0:
                    flag = x
                index += 1

        # now reduce false statechange sequences like -1, 0, 0, -1, 0, 1
        #logger.info('HERE1: ' + str(gchange[0:400]))
        flag = 0
        with np.nditer(gchange, op_flags=['readwrite']) as it:
            for x in it:
                if flag == 0 and x != 0:
                    flag = x
                elif flag == x:
                    x[...] = 0
                elif flag == -x:
                    flag = x

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
            nonzeroMin = np.min(np.nonzero(gchange))
            nonzeroMax = np.max(np.nonzero(gchange))
        except Exception:
            logger.info('AggregateTimeInState all elements zero - returns 0 seconds, from ' + str(gchange.size))
            return 0.0
            pass

        if nonzeroMin > 0:
            #logger.info('YES1 ' + str(nonzeroMin) + ' ' + str(gchange[nonzeroMin]))
            if gchange[nonzeroMin] < 0:
                gchange[0] = 1
        else:
            #logger.info('NO 1 ' + str(nonzeroMin) + ' ' + str(gchange[nonzeroMin]))
            if gchange[0] < 0:
                gchange[0] = 0

        if nonzeroMax > 0:
            #logger.info('YES2 ' + str(nonzeroMax) + ' ' + str(gchange[nonzeroMax]))
            if gchange[nonzeroMax] > 0:
                gchange[-1] = -1
                # if nonzeroMax is last, ignore
                if gchange[nonzeroMax] < 0:
                    gchange[-1] = 0

        # we have odd
        #   -1     1    -1      -> gchange[0] = 0
        #    1    -1     1      -> gchange[-1] = 0
        #         even
        #   -1     1    -1     1   -> gchange[0] = 0 & gchange[-1] = 0
        #    1    -1     1    -1
        # small
        #   -1     1
        #    1    -1
        # smallest
        #   -1           -> gchange[0] = 0
        #    1           -> gchange[0] = 0

        siz = 0
        try:
            siz = np.count_nonzero(gchange)
            if siz == 1:
                gchange[0] = 0
            elif siz == 2 or siz == 0:
                print(2)
            elif siz % 2 != 0:
                # odd
                if gchange[0] == -1: gchange[0] = 0
                else: gchange[-1] = 0
            else:
                # even
                if gchange[0] == -1:
                    gchange[0] = 0
                    gchange[-1] = 0
        except Exception:
            logger.debug('AggregateTimeInState: no state change')
            pass

        #logger.debug('HERE2: ' + str(gchange[0:400]))
        logger.debug('AggregateTimeInState:  state changes ' + str(np.count_nonzero(gchange == 1)) +\
                     ' ' + str(np.count_nonzero(gchange == -1)))

        y = -(gchange * gtime).sum()
        #y = gtime.sum()
        logger.info(str(y))
        if y < 0:
            y = 0.0
        logger.info('AggregateTimeInState returns ' + str(y) + ' seconds, computed from ' + str(gchange.size))
        return y

class StateTimePreparation(BaseTransformer):
    '''
    Together with AggregateTimeInState StateTimePreparation
    calculates the amount of time a selected metric has been in a
    particular state.
    StateTimePreparation outputs an encoded triple of a state variable,
    a state change variable (-1 for leaving the state, 0 for no change,
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

        logger.info('Source: ' + self.source +  ', state_name ' +  self.state_name +  ', Name: ' + self.name +
                    ', Entity: ' + df.index[0][0])

        df_copy = df.reset_index()

        # pair of +- seconds and regular timestamp
        vstate = eval("df_copy[self.source] " + self.state_name).astype(int).values.astype(int)
        vchange = eval("df_copy[self.source] " + self.state_name).astype(int).diff().values.astype(int)

        logger.info(str(vstate))
        logger.info(str(vchange))

        #v1 = np.roll(v1_, -1)  # push the first element, NaN, to the end
        # v1[-1] = 0

        # first value is a NaN, replace it with special value for Aggregator
        vchange[0] = 2

        #logger.debug('HERE: ' + str(v1[0:600]))

        df_copy['__intermediate1__'] = vchange
        df_copy['__intermediate2__'] = vstate
        df_copy['__intermediate3__'] = (df_copy[ts_name].astype(int)// 1000000000)

        df_copy[self.name] = df_copy['__intermediate1__'].map(str) + ',' +\
                             df_copy['__intermediate2__'].map(str) + ',' +\
                             df_copy['__intermediate3__'].map(str) + ',' + df.index[0][0]

        df_copy.drop(columns=['__intermediate1__','__intermediate2__','__intermediate3__'], inplace=True)


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

class DBPreload(BaseTransformer):
    """
    Do a DB request as a preload activity. Load results of the get into the Entity Type time series table.
    """

    out_table_name = None

    def __init__(self, table, timestamp_column, time_offset=True, output_item='db_preload_done'):

        super().__init__()

        # create an instance variable with the same name as each arg
        self.table = table
        self.timestamp_column = timestamp_column
        self.time_offset = time_offset
        self.output_item = output_item

    def execute(self, df, start_ts=None, end_ts=None, entities=None):

        if df is None:
            return df

        logger.info('Execute DBPreload')

        entity_type = self.get_entity_type()
        db = entity_type.db
        table = entity_type.name

        # preserve index
        index_names = df.index.names
        ts_name = df.index.names[1]  # TODO: deal with non-standard dataframes (no timestamp)
        logger.info('DBPreload: Index is ' + str(index_names))

        df = df.reset_index().set_index(ts_name)  # copy

        # read data
        schema = entity_type._db_schema

        #start_ts = df.index.min()    # make that an extra argument - honor_time
        #end_ts = df.index.max()
        start_ts = None
        end_ts = None


        df_input = db.read_table(self.table, None, None, None, self.timestamp_column, start_ts, end_ts)
        logger.info('DBPreload: load columns ' + str(df_input.columns) + ', index ' + str(df_input.index))


        #df_input = df_input.reset_index().rename(columns={self.timestamp_column: ts_name, \
        #    entity_type._entity_id: entity_type._df_index_entity_id}).set_index(ts_name)
        df_input[self.timestamp_column] = pd.to_datetime(df_input[self.timestamp_column])

        if self.time_offset:
            offset = df.index.max() - df_input[self.timestamp_column].max()
            df_input[self.timestamp_column] += offset

        df_input = df_input.rename(columns={'deviceid':'id'}).set_index(self.timestamp_column).sort_index()


        # align dataframe with data received
        db_columns = df_input.columns
        logger.info('DBPreload: columns loaded: ' + str(db_columns))
        new_columns = list(set(db_columns) - set(df.columns) - set(index_names))
        logger.info('DBPreload: new columns: ' + str(new_columns))
        old_columns = list(set(db_columns) - set(new_columns) - set(index_names))
        logger.info('DBPreload: old columns: ' + str(old_columns))

        # ditch old columns - no overwriting
        if len(old_columns) > 0:
            logger.info('DBPreload: Dropping columns: ' + str(old_columns))
            df_input = df_input.drop(columns=old_columns)

        output_column = new_columns.pop()

        if len(new_columns) > 1:
            logger.info('DBPreload: Dropping superfluous columns: ' + str(new_columns))
            df_input.drop(columns=new_columns, inplace=True)

        # rename output column
        logger.info('DBPreload: Rename output column: ' + str(output_column))
        df_input = df_input.rename(columns={output_column: self.output_item})

        # merge data - merge_ordered is not supported
        df = pd.merge_ordered(df, df_input, on=index_names, how='outer')
        logger.info('DBPreload: Merged columns: ' + str(df.columns))

        # write the dataframe to the database table
        #self.write_frame(df=df, table_name=table)
        #kwargs = {'table_name': table, 'schema': schema, 'row_count': len(df.index)}
        #entity_type.trace_append(created_by=self, msg='Wrote data to table', log_method=logger.debug, **kwargs)

        return df.reset_index().set_index(index_names)

    @classmethod
    def build_ui(cls):
        """
        Registration metadata
        """
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingle(name='table', datatype=str, description='db table name'))
        inputs.append(UISingle(name='timestamp_column', datatype=str, description='name of the timestamp column'))
        # define arguments that behave as function outputs
        inputs.append(UISingle(name='time_offset', datatype=bool,
                               description='If true interpret add an time offset to the timestamp column.'))
        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='output_item', datatype=float, description='Data from database'))

        return (inputs, outputs)

