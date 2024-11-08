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
import json
from collections import OrderedDict

import numpy as np
import scipy as sp
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles

# from sqlalchemy import String
from ibm_watson_machine_learning import APIClient
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import empirical_covariance, shrunk_covariance, graphical_lasso

from iotfunctions.base import (
    BaseTransformer,
    BaseEvent,
    BaseSCDLookup,
    BaseSCDLookupWithDefault,
    BaseMetadataProvider,
    BasePreload,
    BaseDatabaseLookup,
    BaseDataSource,
    BaseDBActivityMerge,
    BaseSimpleAggregator,
)
from iotfunctions.ui import (
    UISingle,
    UIMultiItem,
    UIFunctionOutSingle,
    UISingleItem,
    UIFunctionOutMulti,
    UIMulti,
    UIExpression,
    UIText,
    UIParameters,
    UIStatusFlag,
)
from iotfunctions.util import log_data_frame

logger = logging.getLogger(__name__)
PACKAGE_URL = "git+https://github.com/sedgewickmm18/mmfunctions.git"
_IS_PREINSTALLED = False

"""
class KLByWindow:
    __init__(self, alpha):
        self.alpha = alpha

    fit(self, X):
        emp_cov = empirical_covariance(X)
        result = graphical_lasso(emp_cov, alpha=self.alpha, return_n_iter=True)
"""


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
        logger.info("AggregateWithExpression _init")

        self.source = source
        self.expression = expression
        self.name = name

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(
            UIMultiItem(
                name="source",
                datatype=None,
                description=("Choose the data items that you would like to aggregate"),
                output_item="name",
                is_output_datatype_derived=True,
            )
        )
        inputs.append(
            UIExpression(
                name="expression", description="Paste in or type an AS expression"
            )
        )
        return (inputs, [])

    def execute(self, x):
        logger.info("Execute AggregateWithExpression")
        logger.debug(
            "Source "
            + str(self.source)
            + "Expression "
            + str(self.expression)
            + "Name "
            + str(self.name)
        )
        y = eval(self.expression)
        self.log_df_info(y, "AggregateWithExpression evaluation")
        return y


class AggregateTimeInStateX(BaseSimpleAggregator):
    """
    Creates aggregation from the output of StateTimePreparation, a string
    encoded pair of a state change variable (-1 for leaving the state,
    0 for no change, 1 for entering the state) together with a unix epoch
    timestamp.
    It computes the overall number of seconds spent in a particular state.
    """

    def __init__(self, source=None, name=None):
        super().__init__()
        logger.info("AggregateTimeInState _init")

        self.source = source
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(
            UISingleItem(
                name="source",
                datatype=None,
                description="Output of StateTimePreparation to aggregate over",
            )
        )

        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="name",
                datatype=float,
                description="Overall amount of seconds spent in a particular state",
            )
        )

        return (inputs, outputs)

    def execute(self, group):
        logger.info("Execute AggregateTimeInState")

        lg = group.size
        if lg < 2:
            # We need at least two data points for function sp.interpolate.interp1d()
            logger.info(
                f"AggregateTimeInState no elements - returns 0 seconds, from {lg}"
            )
            return 0.0

        # debug stuff
        # pd.set_option("display.max_rows", 50)
        # logger.info(str(group))

        df_group_exp = group.str.split(pat=",", n=3, expand=True)
        # logger.info(str(df_group_exp))

        gchange = None
        gstate = None
        gtime = None
        try:
            gchange = np.append(df_group_exp[0].values.astype(int), 0)
            gstate = np.append(df_group_exp[1].values.astype(int), 0)
            gtime = df_group_exp[2].values.astype(int)

        except Exception as esplit:
            logger.info(
                "AggregateTimeInState elements with NaN- returns 0 seconds, from "
                + str(gchange.size)
            )
            return 0.0

        logger.info("AggregateTimeInState: gchange " + str(gchange))
        logger.info("AggregateTimeInState: gstate " + str(gstate))

        linear_interpolate = sp.interpolate.interp1d(
            np.arange(0, len(gtime)), gtime, kind="linear", fill_value="extrapolate"
        )
        gtime = np.append(gtime, linear_interpolate(len(gtime)))

        sum = 0
        old_state = None
        old_time = None
        for state, time in zip(gstate, gtime):
            if old_time is not None:
                sum += time - old_time
            old_state = state
            old_time = time

        logger.info(
            "AggregateTimeInState returns "
            + str(sum)
            + " seconds, computed from "
            + str(gtime.size)
        )
        return sum


class AggregateTimeInStateXOrig(BaseSimpleAggregator):
    """
    Creates aggregation from the output of StateTimePreparation, a string
    encoded pair of a state change variable (-1 for leaving the state,
    0 for no change, 1 for entering the state) together with a unix epoch
    timestamp.
    It computes the overall number of seconds spent in a particular state.
    """

    def __init__(self, source=None, name=None):
        super().__init__()
        logger.info("AggregateTimeInState _init")

        self.source = source
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(
            UISingleItem(
                name="source",
                datatype=None,
                description="Output of StateTimePreparation to aggregate over",
            )
        )

        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="name",
                datatype=float,
                description="Overall amount of seconds spent in a particular state",
            )
        )

        return (inputs, outputs)

    def execute(self, group):
        logger.info("Execute AggregateTimeInState")

        lg = group.size
        if lg < 2:
            # We need at least two data points for function sp.interpolate.interp1d()
            logger.info(
                f"AggregateTimeInState no elements - returns 0 seconds, from {lg}"
            )
            return 0.0

        # debug stuff
        # pd.set_option("display.max_rows", 50)
        # logger.info(str(group))

        df_group_exp = group.str.split(pat=",", n=3, expand=True)
        # logger.info(str(df_group_exp))

        gchange = None
        gstate = None
        gtime = None
        try:
            gchange = np.append(df_group_exp[0].values.astype(int), 0)
            gstate = np.append(df_group_exp[1].values.astype(int), 0)
            gtime = df_group_exp[2].values.astype(int)

        except Exception as esplit:
            logger.info(
                "AggregateTimeInState elements with NaN- returns 0 seconds, from "
                + str(gchange.size)
            )
            return 0.0

        logger.info("AggregateTimeInState: gchange " + str(gchange))
        logger.info("AggregateTimeInState: gstate " + str(gstate))

        linear_interpolate = sp.interpolate.interp1d(
            np.arange(0, len(gtime)), gtime, kind="linear", fill_value="extrapolate"
        )
        gtime = np.append(gtime, linear_interpolate(len(gtime)))

        # no statechange at all
        if not np.any(gchange):
            logger.debug(
                "AggregateTimeInState: no state change at all in this aggregation, inject it"
            )
            gchange[0] = gstate[0]
            gchange[-1] = -gstate[0]

        # look for state change 2 - start interval for StateTimePrep
        #
        # |  previous run  -1 ...  |2 ..  1 next run    |
        flag = 0
        index = 0
        with np.nditer(gchange, op_flags=["readwrite"]) as it:
            for x in it:
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
        # logger.info('HERE1: ' + str(gchange[0:400]))
        flag = 0
        with np.nditer(gchange, op_flags=["readwrite"]) as it:
            for x in it:
                if flag == 0 and x != 0:
                    flag = x
                elif flag == x:
                    x[...] = 0
                elif flag == -x:
                    flag = x

        # adjust for intervals cut in half by aggregation
        """
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
        """

        # first non zero index
        nonzeroMin = 0
        nonzeroMax = 0
        try:
            nonzeroMin = np.min(np.nonzero(gchange))
            nonzeroMax = np.max(np.nonzero(gchange))
        except Exception:
            logger.info(
                "AggregateTimeInState all elements zero - returns 0 seconds, from "
                + str(gchange.size)
            )
            return 0.0
            pass

        if nonzeroMin > 0:
            logger.info("YES1 " + str(nonzeroMin) + " " + str(gchange[nonzeroMin]))
            if gchange[nonzeroMin] < 0:
                gchange[0] = 1
        else:
            logger.info("NO 1 " + str(nonzeroMin) + " " + str(gchange[nonzeroMin]))
            if gchange[0] < 0:
                gchange[0] = 0

        if nonzeroMax > 0:
            logger.info("YES2 " + str(nonzeroMax) + " " + str(gchange[nonzeroMax]))
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
                if gchange[0] == -1:
                    gchange[0] = 0
                else:
                    gchange[-1] = 0
            else:
                # even
                if gchange[0] == -1:
                    gchange[0] = 0
                    gchange[-1] = 0
        except Exception:
            logger.debug("AggregateTimeInState: no state change")
            pass

        # logger.debug('HERE2: ' + str(gchange[0:400]))
        logger.info(
            "AggregateTimeInState:  state changes "
            + str(np.count_nonzero(gchange == 1))
            + " "
            + str(np.count_nonzero(gchange == -1))
        )

        y = -(gchange * gtime).sum()
        # y = gtime.sum()
        logger.info(str(y))
        if y < 0:
            y = 0.0
        logger.info(
            "AggregateTimeInState returns "
            + str(y)
            + " seconds, computed from "
            + str(gchange.size)
        )
        trigger_error
        return y


class StateTimePreparationX(BaseTransformer):
    """
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
    """

    def __init__(self, source=None, state_name=None, name=None):
        super().__init__()
        logger.info("StateTimePrep _init")

        self.source = source
        self.state_name = state_name
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(
            UISingleItem(
                name="source",
                datatype=float,
                description="Data item to compute the state change array from",
            )
        )
        inputs.append(
            UISingle(
                name="state_name",
                datatype=str,
                description="Condition for the state change array computation",
            )
        )
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="name", datatype=str, description="State change array output"
            )
        )

        return (inputs, outputs)

    def execute(self, df):
        logger.debug("Execute StateTimePrep")
        df_copy = df  # no copy

        # set output columns to zero
        df_copy[self.state_name] = 0

        # group over entities
        group_base = [pd.Grouper(axis=0, level=0)]

        if not df_copy.empty:
            df_copy = df_copy.groupby(group_base).apply(self._calc)

        logger.debug("StateTimePrep done")
        return df_copy

    def _calc(self, df):
        logger.info("Execute StateTimePrep per entity")

        index_names = df.index.names
        ts_name = df.index.names[
            1
        ]  # TODO: deal with non-standard dataframes (no timestamp)

        logger.info(
            "Source: "
            + self.source
            + ", ts_name "
            + ts_name
            + ", state_name "
            + self.state_name
            + ", Name: "
            + self.name
            + ", Entity: "
            + df.index[0][0]
        )

        df_copy = df.reset_index()

        # pair of +- seconds and regular timestamp
        vstate = (
            eval("df_copy[self.source] " + self.state_name)
            .astype(int)
            .values.astype(int)
        )
        vchange = np.diff(vstate, prepend=2)

        logger.info("vstate: " + str(vstate))
        logger.info("vchange: " + str(vchange))

        # v1 = np.roll(v1_, -1)  # push the first element, NaN, to the end
        # v1[-1] = 0

        # first value is a NaN, replace it with special value for Aggregator
        vchange[0] = 2

        # logger.debug('HERE: ' + str(v1[0:600]))

        df_copy["__intermediate1__"] = vchange
        df_copy["__intermediate2__"] = vstate
        df_copy["__intermediate3__"] = df_copy[ts_name].astype(int) // 1000000000

        df_copy[self.name] = (
            df_copy["__intermediate1__"].map(str)
            + ","
            + df_copy["__intermediate2__"].map(str)
            + ","
            + df_copy["__intermediate3__"].map(str)
            + ","
            + df.index[0][0]
        )

        df_copy.drop(
            columns=["__intermediate1__", "__intermediate2__", "__intermediate3__"],
            inplace=True,
        )

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
        logger.info("SeasonalDecompose _init")

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Data item to interpolate",
            )
        )
        inputs.append(
            UISingle(
                name="windowsize",
                datatype=int,
                description="Minimal size of the window for interpolating data.",
            )
        )
        inputs.append(
            UISingle(
                name="missing",
                datatype=int,
                description="Data to be interpreted as not-a-number.",
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item", datatype=float, description="Interpolated data"
            )
        )
        return (inputs, outputs)

    def _calc(self, df):
        logger.info("kexecute SeasonalDecompose")

        print(df.index)
        df.to_csv("/tmp/testtest")

        index_names = df.index.names
        ts_name = df.index.names[
            1
        ]  # TODO: deal with non-standard dataframes (no timestamp)

        df_copy = df.copy()
        df_rst = df.reset_index().set_index(ts_name)

        # deal with string timestamp indices
        if not isinstance(df_rst.index, pd.core.indexes.datetimes.DatetimeIndex):
            df_rst.index = pd.to_datetime(df_rst.index, format="%Y-%m-%d-%H.%M.%S.%f")
            # print(df_copy.index)

        # minimal frequency supported by STL
        df_sample = df_rst[[self.input_item]].resample("H").mean().ffill()
        res = STL(df_sample, robust=True).fit()

        df_new = pd.DataFrame(index=df_rst.index)
        df_new["power"] = np.interp(df_rst.index, res.trend.index, res.trend.values)
        print("Power trend", df_new["power"][0:3])

        df_copy[self.output_item] = df_new["power"].values
        print("Power trend", df_copy[self.output_item][0:3])

        logger.info("Exit SeasonalDecompose")
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
        logger.info("AggregateKDEDensity1d _init")

        self.source = source
        self.alpha = alpha
        self.name = name
        print(dir(self))

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(
            UIMultiItem(
                name="source",
                datatype=None,
                description=("Choose the data items that you would like to aggregate"),
                output_item="name",
                is_output_datatype_derived=True,
            )
        )

        inputs.append(
            UIExpression(name="alpha", description="Quantile level - default 0.995")
        )

        return (inputs, [])

    def execute(self, group):
        logger.info("Execute AggregateKDEDensity1d")

        if group.size == 0:
            return 0

        X = group.values.reshape(-1, 1)

        # set up kernel density
        kde = KernelDensity(kernel="gaussian")
        kde.fit(X)

        # apply it and compute the log density for the observed data
        kde_X = kde.score_samples(X)

        # cut point
        tau_kde = mquantiles(kde_X, 1.0 - self.alpha)

        # locate inliers and outliers
        outliers = np.nonzero(kde_X < tau_kde)
        # outliers = outliers.flatten()
        inliers = np.nonzero(kde_X >= tau_kde)
        # inliers = inliers.flatten()

        logger.info(
            "AggregateKDEDensity1d: size: "
            + str(len(X))
            + " inliers: "
            + str(len(inliers))
            + " outliers: "
            + str(len(outliers))
        )

        # inliers provides a lower bound
        lower_bound = 0
        try:
            lower_bound = np.max(X[inliers])
        except Exception as e:
            logger.info("Establishing lower bound failed with " + str(e))

        print(lower_bound)

        raw_threshold = 0
        try:
            high_outliers = np.nonzero(X[outliers] > lower_bound)
            raw_threshold = np.min(X[high_outliers])
        except Exception as ee:
            logger.info("Establishing threshold failed with " + str(ee))

        # print('Source ', self.source, 'Expression ', self.expression, 'Name ', self.name)
        logger.info("AggregateKDEDensity1d returns " + str(raw_threshold))

        return raw_threshold


class DBPreload(BaseTransformer):
    """
    Do a DB request as a preload activity. Load results of the get into the Entity Type time series table.
    """

    out_table_name = None

    def __init__(
        self, table, timestamp_column, time_offset=True, output_item="db_preload_done"
    ):
        super().__init__()

        # create an instance variable with the same name as each arg
        self.table = table
        self.timestamp_column = timestamp_column
        self.time_offset = time_offset
        self.output_item = output_item

    def execute(self, df, start_ts=None, end_ts=None, entities=None):
        if df is None:
            return df

        logger.info("Execute DBPreload")

        entity_type = self.get_entity_type()
        db = entity_type.db
        table = entity_type.name

        # preserve index
        index_names = df.index.names
        ts_name = df.index.names[
            1
        ]  # TODO: deal with non-standard dataframes (no timestamp)
        logger.info("DBPreload: Index is " + str(index_names))

        df = df.reset_index().set_index(ts_name)  # copy

        # read data
        schema = entity_type._db_schema

        # start_ts = df.index.min()    # make that an extra argument - honor_time
        # end_ts = df.index.max()
        start_ts = None
        end_ts = None

        df_input = db.read_table(
            self.table, None, None, None, self.timestamp_column, start_ts, end_ts
        )
        logger.info(
            "DBPreload: load columns "
            + str(df_input.columns)
            + ", index "
            + str(df_input.index)
        )

        # df_input = df_input.reset_index().rename(columns={self.timestamp_column: ts_name, \
        #    entity_type._entity_id: entity_type._df_index_entity_id}).set_index(ts_name)
        df_input[self.timestamp_column] = pd.to_datetime(
            df_input[self.timestamp_column]
        )

        if self.time_offset:
            offset = df.index.max() - df_input[self.timestamp_column].max()
            df_input[self.timestamp_column] += offset

        df_input = (
            df_input.rename(columns={"deviceid": "id"})
            .set_index(self.timestamp_column)
            .sort_index()
        )

        # align dataframe with data received
        db_columns = df_input.columns
        logger.info("DBPreload: columns loaded: " + str(db_columns))
        new_columns = list(set(db_columns) - set(df.columns) - set(index_names))
        logger.info("DBPreload: new columns: " + str(new_columns))
        old_columns = list(set(db_columns) - set(new_columns) - set(index_names))
        logger.info("DBPreload: old columns: " + str(old_columns))

        # ditch old columns - no overwriting
        if len(old_columns) > 0:
            logger.info("DBPreload: Dropping columns: " + str(old_columns))
            df_input = df_input.drop(columns=old_columns)

        output_column = new_columns.pop()

        if len(new_columns) > 1:
            logger.info("DBPreload: Dropping superfluous columns: " + str(new_columns))
            df_input.drop(columns=new_columns, inplace=True)

        # rename output column
        logger.info("DBPreload: Rename output column: " + str(output_column))
        df_input = df_input.rename(columns={output_column: self.output_item})

        # merge data - merge_ordered is not supported
        df = pd.merge_ordered(df, df_input, on=index_names, how="outer")
        logger.info("DBPreload: Merged columns: " + str(df.columns))

        # write the dataframe to the database table
        # self.write_frame(df=df, table_name=table)
        # kwargs = {'table_name': table, 'schema': schema, 'row_count': len(df.index)}
        # entity_type.trace_append(created_by=self, msg='Wrote data to table', log_method=logger.debug, **kwargs)

        return df.reset_index().set_index(index_names)

    @classmethod
    def build_ui(cls):
        """
        Registration metadata
        """
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingle(name="table", datatype=str, description="db table name"))
        inputs.append(
            UISingle(
                name="timestamp_column",
                datatype=str,
                description="name of the timestamp column",
            )
        )
        # define arguments that behave as function outputs
        inputs.append(
            UISingle(
                name="time_offset",
                datatype=bool,
                description="If true interpret add an time offset to the timestamp column.",
            )
        )
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item", datatype=float, description="Data from database"
            )
        )

        return (inputs, outputs)


class InvokeWMLModelBase(BaseTransformer):
    """
    Pass multivariate data in input_items to a regression function deployed to
    Watson Machine Learning. The results are passed back to the univariate
    output_items column.
    Credentials for the WML endpoint representing the deployed function are stored
    as pipeline constants, a name to lookup the WML credentials as JSON document.
    Example: 'my_deployed_endpoint_wml_credentials' referring to
    {
            "apikey": "<my api key",
            "url": "https://us-south.ml.cloud.ibm.com",
            "space_id": "<my space id>",
            "deployment_id": "<my deployment id">
    }
    This name is passed to InvokeWMLModel in wml_auth.
    """

    def __init__(self, input_items, wml_auth, output_items):
        super().__init__()

        logger.debug(input_items)

        self.whoami = "InvokeWMLModel"

        self.input_items = input_items

        if isinstance(output_items, str):
            self.output_items = [output_items]  # regression
        else:
            self.output_items = output_items  # classification

        self.wml_auth = wml_auth

        self.deployment_id = None
        self.apikey = None
        self.wml_endpoint = None
        self.space_id = None

        self.client = None

        self.time_column = None

        self.logged_on = False

    def __str__(self):
        out = self.__class__.__name__
        try:
            out = out + "Input: " + str(self.input_items) + "\n"
            out = out + "Output: " + str(self.output_items) + "\n"

            if self.wml_auth is not None:
                out = out + "WML auth: " + str(self.wml_auth) + "\n"
            else:
                # out = out + 'APIKey: ' + str(self.apikey) + '\n'
                out = out + "WML endpoint: " + str(self.wml_endpoint) + "\n"
                out = out + "WML space id: " + str(self.space_id) + "\n"
                out = out + "WML deployment id: " + str(self.deployment_id) + "\n"
        except Exception:
            pass
        return out

    def login(self):
        # only do it once
        if self.logged_on:
            return

        # retrieve WML credentials as constant
        #    {"apikey": api_key, "url": 'https://' + location + '.ml.cloud.ibm.com'}
        c = None
        if isinstance(self.wml_auth, str):
            try:
                wml_auth = json.loads(self.wml_auth)
                self.wml_auth = wml_auth
            except Exception:
                logger.debug('WML parm is not a JSON document')

        if isinstance(self.wml_auth, dict):
            logger.info("WML AUTH " + str(self.wml_auth) + "  is dict")
            wml_credentials = self.wml_auth
        elif self.wml_auth is not None:
            logger.info("WML AUTH " + str(self.wml_auth) + " - get from pipeline")
            try:
                c = self._entity_type.get_attributes_dict()
            except Exception:
                c = None

            try:
                wml_credentials = c[self.wml_auth]
            except Exception as ae:
                wml_credentials = None

            if wml_credentials is None:
                logger.info("Pipeline constant does not exist")
                wml_credentials = json.loads(self.wml_auth)

            logger.info("WE GOT " + str(wml_credentials))

            # except Exception as ae:
            #    raise RuntimeError("No WML credentials specified")
        else:
            wml_credentials = {
                "apikey": self.apikey,
                "url": self.wml_endpoint,
                "space_id": self.space_id,
            }

        try:
            self.deployment_id = wml_credentials["deployment_id"]
            # self.space_id = wml_credentials['space_id']
            logger.info("Found credentials for WML")
        except Exception as ae:
            raise RuntimeError("No valid WML credentials specified")

        # get client and check credentials
        self.client = APIClient(wml_credentials)
        if self.client is None:
            # logger.error('WML API Key invalid')
            raise RuntimeError("WML API Key invalid")

        # get space
        if 'space_id' in wml_credentials:
            self.space_id = wml_credentials["space_id"]
        else:
            json_d = self.client.spaces._get_resources(
                wml_credentials["url"] + "/v2/spaces", "spaces", {"limit": 1}
            )
            self.space_id = json_d["resources"][0]["metadata"]["id"]

        # set space
        self.client.set.default_space(self.space_id)

        # check deployment
        deployment_details = self.client.deployments.get_details(self.deployment_id, 1)
        # ToDo - test return and error msg
        logger.debug("Deployment Details check results in " + str(deployment_details))

        self.logged_on = True

    def execute(self, df):
        logger.info("InvokeWML exec")

        # Create missing columns before doing group-apply
        df = df.copy().fillna("")
        missing_cols = [x for x in (self.output_items) if x not in df.columns]
        for m in missing_cols:
            df[m] = None

        self.login()

        return super().execute(df)

    def _calc(self, df):
        if len(self.input_items) >= 1:
            idx_names = df.index.names

            if self.time_column is not None:
                df = df.reset_index()
                df["__timestamp__"] = df[idx_names[1]].dt.strftime("%Y-%m-%dT%H:%M")
                # df[idx_names[1]].values.astype(str)
                df = df.drop_duplicates(subset=["__timestamp__"])
                df = df.set_index(idx_names)
                input_items = ["__timestamp__"]
                field_names = [self.time_column]
                input_items.extend(self.input_items)
                field_names.extend(self.input_items)
            else:
                input_items = self.input_items
                field_names = self.input_items

            index_nans = df[df[input_items].isna().any(axis=1)].index
            rows = df.loc[~df.index.isin(index_nans), input_items].values.tolist()

            scoring_payload = {
                self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                    {"fields": field_names, "values": rows}
                ]
            }
            print("SCORING PAYLOAD", scoring_payload)
        else:
            logging.error("no input columns provided, forwarding all")
            return df

        results = self.client.deployments.score(self.deployment_id, scoring_payload)

        if results is not None:
            # Check if results is already a dictionary
            if isinstance(results, dict):
                result_json = results
                logger.info("Result is in a string")
            else:
                # Assume results is a JSON string and try to load it
                try:
                    result_json = json.loads(results)
                except JSONDecodeError as jse:
                    logger.info("Apparently not a JSON string")
                    result_json = (
                        None  # If it's not JSON, set to None or handle appropriately
                    )

            # Regression
            if len(self.output_items) == 1:
                #logger.info(str(results["predictions"]) + ", " + str(results["predictions"][0]) + ", " + 
                #    str(results["predictions"][0]["values"]))
                #logger.info(str(np.array(results["predictions"][0]["values"])))

                df.loc[~df.index.isin(index_nans), self.output_items] = np.array(
                    results["predictions"][0]["values"]
                ).flatten()
            # Classification
            else:
                arr = np.array(results["predictions"][0]["values"])
                df.loc[~df.index.isin(index_nans), self.output_items[0]] = arr[
                    :, 0
                ].astype(int)
                arr2 = np.array(arr[:, 1].tolist())
                df.loc[~df.index.isin(index_nans), self.output_items[1]] = arr2.T[0]

            logger.info(str(df[self.output_items]))

        else:
            logging.error("error invoking external model")

        return df

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UIMultiItem(
                name="input_items",
                datatype=float,
                description="Data items adjust",
                is_output_datatype_derived=True,
            )
        )
        inputs.append(
            UISingle(
                name="wml_auth",
                datatype=str,
                description="Endpoint to WML service where model is hosted",
                tags=["TEXT"],
                required=True,
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UISingle(name="output_items", datatype=float))
        return (inputs, outputs)


class InvokeWMLModelX(InvokeWMLModelBase):
    """
    Pass multivariate data in input_items to a regression function deployed to
    Watson Machine Learning. The results are passed back to the univariate
    output_items column.
    Credentials for the WML endpoint representing the deployed function are stored
    as pipeline constants, a name to lookup the WML credentials as JSON document.
    Example: 'my_deployed_endpoint_wml_credentials' referring to
    {
            "apikey": "<my api key",
            "url": "https://us-south.ml.cloud.ibm.com",
            "space_id": "<my space id>",
            "deployment_id": "<my deployment id">
    }
    This name is passed to InvokeWMLModel in wml_auth.
    """

    def __init__(self, input_items, wml_auth, output_items):
        super().__init__(input_items, wml_auth, output_items)

        logger.debug(input_items)

        self.whoami = "InvokeWMLModelX"

        self.time_column = "TIMESTAMP"

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UIMultiItem(
                name="input_items",
                datatype=float,
                description="Data items adjust",
                is_output_datatype_derived=True,
            )
        )
        inputs.append(
            UISingle(
                name="wml_auth",
                datatype=str,
                description="Endpoint to WML service where model is hosted",
                tags=["TEXT"],
                required=True,
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UISingle(name="output_items", datatype=float))
        return (inputs, outputs)


class InvokeWMLModelWithTime(InvokeWMLModelBase):
    """
    Pass multivariate data in input_items to a regression function deployed to
    Watson Machine Learning. The results are passed back to the univariate
    output_items column.
    Credentials for the WML endpoint representing the deployed function are stored
    as pipeline constants, a name to lookup the WML credentials as JSON document.
    Example: 'my_deployed_endpoint_wml_credentials' referring to
    {
            "apikey": "<my api key",
            "url": "https://us-south.ml.cloud.ibm.com",
            "space_id": "<my space id>",
            "deployment_id": "<my deployment id">
    }
    This name is passed to InvokeWMLModel in wml_auth.
    """
    def __init__(self, input_items, wml_auth, output_items, time_column):
        super().__init__(input_items, wml_auth, output_items)

        logger.debug(input_items)

        self.whoami = "InvokeWMLModelWithTime"

        self.time_column = time_column

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UIMultiItem(
                name="input_items",
                datatype=float,
                description="Data items adjust",
                is_output_datatype_derived=True,
            )
        )
        inputs.append(
            UISingle(
                name="wml_auth",
                datatype=str,
                description="Endpoint to WML service where model is hosted",
                tags=["TEXT"],
                required=True,
            )
        )

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UISingle(name="output_items", datatype=float))
        outputs.append(UISingle(name="time_column", datatype=dt.datetime))
        return (inputs, outputs)


LASTROWS = 500


class InvokeWMLModelMulti(BaseTransformer):
    """
    Pass multivariate data in input_items to a regression function deployed to
    Watson Machine Learning. The results are passed back to the univariate
    output_items column.
    Credentials for the WML endpoint representing the deployed function are stored
    as pipeline constants, a name to lookup the WML credentials as JSON document.
    Example: 'my_deployed_endpoint_wml_credentials' referring to
    {
            "apikey": "<my api key",
            "url": "https://us-south.ml.cloud.ibm.com",
            "space_id": "<my space id>",
            "deployment_id": "<my deployment id">
    }
    This name is passed to InvokeWMLModel in wml_auth.
    """

    def __init__(self, input_items, wml_auth, ignore_output, output_items):
        super().__init__()

        logger.debug(str(input_items))
        logger.debug(str(output_items))

        self.whoami = "InvokeWMLModel"

        self.input_items = input_items
        self.output_items = output_items

        if isinstance(output_items, str):
            # self.output_items = [output_items]    # regression
            self.output_items = [f"{output_items}{i}" for i in range(0, 34)]
        else:
            self.output_items = output_items  # classification

        self.db = None
        self.wml_auth = wml_auth
        self.ignore_output = ignore_output

        self.deployment_id = None
        self.apikey = None
        self.wml_endpoint = None
        self.space_id = None

        self.client = None

        self.logged_on = False

    def __str__(self):
        out = self.__class__.__name__
        try:
            out = out + "Input: " + str(self.input_items) + "\n"
            out = out + "Output: " + str(self.output_items) + "\n"

            if self.wml_auth is not None:
                out = out + "WML auth: " + str(self.wml_auth) + "\n"
            else:
                # out = out + 'APIKey: ' + str(self.apikey) + '\n'
                out = out + "WML endpoint: " + str(self.wml_endpoint) + "\n"
                out = out + "WML space id: " + str(self.space_id) + "\n"
                out = out + "WML deployment id: " + str(self.deployment_id) + "\n"
        except Exception:
            pass
        return out

    def login(self):
        # only do it once
        if self.logged_on:
            return

        # retrieve WML credentials as constant
        #    {"apikey": api_key, "url": 'https://' + location + '.ml.cloud.ibm.com'}
        c = None
        if isinstance(self.wml_auth, dict):
            wml_credentials = self.wml_auth
        elif self.wml_auth is not None:
            try:
                c = self._entity_type.get_attributes_dict()
            except Exception:
                c = None

            logger.info("Constants " + str(c))

            try:
                wml_credentials = c[self.wml_auth]
            except Exception as ae:
                raise RuntimeError("No WML credentials specified")
        else:
            wml_credentials = {
                "apikey": self.apikey,
                "url": self.wml_endpoint,
                "space_id": self.space_id,
            }

        logger.info("Creds " + str(wml_credentials))

        try:
            # if hasattr(wml_credentials, 'deployment_id'):
            self.deployment_id = wml_credentials["deployment_id"]
            if hasattr(wml_credentials, "space_id"):
                self.space_id = wml_credentials["space_id"]
            # if hasattr(wml_credentials, 'wml_deployment_space_name'):
            self.space_id = wml_credentials["wml_deployment_space_name"]

            logger.info("Found credentials for WML")
        except Exception as ae:
            logger.info("No deployment or space id, but we'll try anyway")
            pass
        """
            raise RuntimeError("No valid WML credentials specified")
        """

        # get client and check credentials
        self.client = APIClient(wml_credentials)
        if self.client is None:
            # logger.error('WML API Key invalid')
            raise RuntimeError("WML API Key invalid")

        # set space
        if self.space_id is not None:
            logger.error("Setting space id to " + str(self.space_id))
            self.client.set.default_space(self.space_id)

        # check deployment
        deployment_details = None
        if self.deployment_id is not None:
            deployment_details = self.client.deployments.get_details(
                self.deployment_id, 1
            )

        # ToDo - test return and error msg
        if deployment_details is not None:
            logger.debug(
                "Deployment Details check results in " + str(deployment_details)
            )

        self.logged_on = True

    def execute(self, df):
        logger.info("InvokeWML exec")

        self.db = self._entity_type.db
        logger.info("db is " + str(self.db))
        if self.db is None:
            self.db = self._get_dms().db

        # Create missing columns before doing group-apply
        # df = df.copy().fillna('')
        df = df.copy().fillna(value=0.0)
        missing_cols = [x for x in (self.output_items) if x not in df.columns]
        for m in missing_cols:
            df[m] = 0.0

        self.login()

        return super().execute(df)

    def _calc(self, df):
        df_orig = df

        idx_names = df.index.names
        df = df.reset_index()

        input_items = [idx_names[1]]
        input_items.extend(self.input_items)

        index_nans = df[df[input_items].isna().any(axis=1)].index
        df_ = df.replace(r"^\s*$", 0.0, regex=True)

        shape = df_.loc[~df.index.isin(index_nans), input_items].values.shape
        orig_arr = df_.loc[~df.index.isin(index_nans), input_items].values

        full_arr = np.zeros(shape)

        ranges = range(0, shape[0], LASTROWS - 100)
        for start in ranges:
            segment_size = LASTROWS
            if start + LASTROWS > shape[0]:
                arr = orig_arr[start:, :]
                segment_size = shape[0] - start
            else:
                arr = orig_arr[start : start + segment_size, :]
            # self.db.model_store.store_model('Invoker', arr)

            rows = arr.tolist()
            scoring_payload = {
                "input_data": [{"fields": self.input_items, "values": rows}]
            }
            logger.info(
                "Field: "
                + str(self.input_items)
                + ", Payload length: "
                + str(len(rows))
                + ", Start: "
                + str(start)
                + ", SegSize: "
                + str(segment_size)
                + ", Shape: "
                + str(shape)
            )

            results = self.client.deployments.score(self.deployment_id, scoring_payload)

            if results:
                # logger.info(results['predictions'][0]['values'][1])
                arr = np.array(
                    results["predictions"][0]["values"][self.ignore_output :]
                )[:, 0, :]
                # self.db.model_store.store_model('Result', arr)

                logger.info("Result shape: " + str(arr.shape))

                logger.info("Stretch " + str(arr.shape[1]) + " columns")
                for i in range(arr.shape[1]):
                    current_x_axis = np.linspace(0, arr.shape[0], arr.shape[0])
                    linear_interpolate = sp.interpolate.interp1d(
                        current_x_axis,
                        arr[:, i],
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    full_arr[start : start + segment_size, i] = linear_interpolate(
                        np.arange(0, segment_size, 1)
                    )

                logger.info(
                    "Assigning " + str(full_arr.shape) + " to " + str(self.output_items)
                )

                # df.loc[~df.index.isin(index_nans), self.output_items] = full_arr
                logger.info("Frame columns are " + str(df.columns))
                df[self.output_items] = full_arr
                logger.info("Now frame columns are " + str(df.columns))
            else:
                logging.error("error invoking external model")

        df = df.set_index(idx_names)
        logging.info("Evaluation loop ended")
        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        outputs = []

        # define arguments that behave as function inputs
        inputs.append(
            UIMultiItem(
                name="input_items",
                datatype=float,
                required=True,
                output_item="output_items",
                description="Data items adjust",
                is_output_datatype_derived=True,
            )
        )
        inputs.append(
            UISingle(
                name="wml_auth",
                datatype=str,
                description="Endpoint to WML service where model is hosted",
                tags=["TEXT"],
                required=True,
            )
        )
        inputs.append(
            UISingle(
                name="ignore_output",
                datatype=int,
                description="Disregard the first output columns from the model",
            )
        )

        # define arguments that behave as function outputs
        return (inputs, outputs)


class AlertExpressionPulsed(BaseEvent):
    """
    Create alerts that are triggered when data values the expression is True
    """

    def __init__(self, expression, alert_name, **kwargs):
        self.expression = expression
        self.alert_name = alert_name
        self.pulse_trigger = True
        super().__init__()
        logger.info(
            "AlertExpressionPulsed exp: "
            + str(expression)
            + "  alert: "
            + str(alert_name)
        )

    # evaluate alerts by entity
    def _calc(self, df):
        try:
            c = self._entity_type.get_attributes_dict()
        except Exception:
            c = None

        df = df.copy()
        print(df.index)
        logger.info(
            "AlertExpressionPulsed exp: "
            + self.expression
            + "  input: "
            + str(df.columns)
        )

        expr = self.expression

        if "${" in expr:
            expr = re.sub(r"\$\{(\w+)\}", r"df['\1']", expr)
            msg = "Expression converted to %s. " % expr
        else:
            msg = "Expression (%s). " % expr

        self.trace_append(msg)

        expr = str(expr)
        logger.info("AlertExpressionWithFilter  - after regexp: " + expr)

        df[self.alert_name] = False
        try:
            evl = eval(expr)
            np_res = np.where(evl, 1, 0)

        except Exception as e:
            logger.info(
                "AlertExpressionPulsed eval for " + expr + " failed with " + str(e)
            )
            df[self.alert_name] = False
            return df

        try:
            # get time index
            ts_ind = df.index.get_level_values(self._entity_type._timestamp)

            if self.pulse_trigger:
                # walk through all subsequences starting with the longest
                # and replace all True with True, False, False, ...
                for i in range(np_res.size, 2, -1):
                    for j in range(0, i - 1):
                        if np.all(np_res[j:i]):
                            logger.debug("Found subsequence " + str(j) + ":" + str(i))
                            np_res[j + 1 : i] = np.zeros(i - j - 1, dtype=int)
                            np_res[j] = i - j  # keep track of sequence length

            logger.info("AlertExpressionPulsed shapes " + str(np_res.shape))
            df[self.alert_name] = np_res

        except Exception as e:
            logger.info(
                "AlertExpressionPulsed eval for " + expr + " failed with2 " + str(e)
            )
            df[self.alert_name] = None
            pass

        return df

    # def execute(self, df):
    #    df = super().execute(df)
    #    logger.info('AlertExpressionPulsed generated columns: ' + str(df.columns))
    #    return df

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(
            UIExpression(
                name="expression",
                description="Define alert expression using pandas systax. Example: df['inlet_temperature']>50",
            )
        )
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="alert_name", datatype=bool, description="Output of alert function"
            )
        )
        return (inputs, outputs)


class CumulativeCount(BaseTransformer):
    def __init__(self, input_item, state_name, output_item):
        """
        We plan to report the anomaly only for the recent time points
        """
        # Do away with numba logs
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.INFO)

        super().__init__()
        self.input_item = input_item
        self.state_name = state_name
        self.output_item = output_item
        logger.debug(f"Input item: {input_item}")
        self.whoami = "CumulativeCount"

    def get_dm_from_db(self, start_ts):
        """ """
        self.db = self._entity_type.db
        if self.db is None:
            self.db = self._get_dms().db

        entity_type = self.get_entity_type()
        logger.info("Retrieving dm data from " + str(start_ts) + " to ")

        schema = entity_type._db_schema
        raw_dataframe = None

        try:
            source_metadata = self._get_dms().data_items.get(self.output_item)
            derived_metric_table_name = source_metadata.get(
                md.DATA_ITEM_SOURCETABLE_KEY
            )
            query, table = db.query(
                derived_metric_table_name,
                schema,
                column_names=["KEY", "VALUE_N", entity_type._timestamp],
            )
            query = query.filter(db.get_column_object(table, "KEY") == self.output_item)

            raw_dataframe = db.read_sql_query(query.statement)
            raw_dataframe.astype({"KEY": "string", "TIMESTAMP": "string"}, copy=False)
            log_data_frame("Data frame after fetching from DB", raw_dataframe)
        except Exception as e:
            logger.info("Error when retrieving derived metrics. Error " + str(e))
            pass

        return raw_dataframe

    def _calc(self, df):
        """
        Entry point per entity cumsum
        """
        entity = df.index[0][0]

        index_names = df.index.names

        df_copy = df.reset_index()

        entity_type = self.get_entity_type()

        last_ts = df_copy[entity_type._timestamp].values[0]
        last_val = 0
        # get last timestamp and value for entity
        if self.right_db is not None:
            last_row = self.right_db.loc[self.right_db["DEVICEID"] == entity, :].head(1)

            if not last_row.empty:
                last_ts = last_row["TIMESTAMP"]
                last_val = last_row["value_n"]

        # condition and regular timestamp diff
        vstate = 1 - eval("df_copy[self.input_item] " + self.state_name).astype(
            int
        ).values.astype(int)
        print(vstate, last_ts)
        try:
            t_diff = (
                np.diff(np.insert(df_copy[entity_type._timestamp].values, 0, last_ts))
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
        except Exception as ee:
            print(ee)

        marr = np.ma.array(t_diff, mask=vstate).filled(0)
        print(marr)
        c_sum = marr.cumsum()  # .filled(np.nan)
        print(c_sum)
        df_copy[self.output_item] = c_sum + last_val
        # df_copy[self.output_item] = df_copy[self.output_item].fillna(method="bfill") + last_val

        print(self.output_item, df_copy[self.output_item])

        return df_copy.set_index(index_names)

    def execute(self, df):
        """
        Entry point
        """

        self.right_db = None
        start_ts = df.index[0:][0][1]
        last_ts = None
        last_val = 0
        try:
            self.right_db = self.get_dm_from_db(start_ts)
        except Exception as e:
            logger.info("Could not get old score. " + str(e))
            pass

        # delegate to _calc
        logger.debug("Execute " + self.whoami + " enter per entity execution")

        df[self.output_item] = 0

        # group over entities
        group_base = [pd.Grouper(axis=0, level=0)]

        if not df.empty:
            df = df.groupby(group_base).apply(self._calc)

        logger.debug("Scoring done")

        log_data_frame("Result after fetching from DB", df)

        if self.right_db is None:
            return df

        try:
            self.right_db = self.right_db.set_index("TIMESTAMP")
            left_db = df.droplevel(0)
            merged_df = left_db.join(self.right_db)
            scores_ = merged_df[[self.output_item, "value_n"]].values
            result = scores_[:, 1]
            loc = [
                i
                for i in range(len(scores_[:, 1]))
                if np.isnan(scores_[:, 1][i]) or scores_[:, 1][i] is None
            ]
            result[loc] = scores_[loc, 0]
            df[:][self.output_item] = result
        except Exception as e:
            logger.info("Merge of old and new score failed because of " + str(e))
            pass

        return df

    def build_ui(cls):
        inputs = []
        inputs.append(
            UISingleItem(
                name="input_item",
                datatype=float,
                description="Data item to compute the state change array from",
            )
        )
        inputs.append(
            UISingle(
                name="state_name",
                datatype=str,
                description="Condition for the state change array computation",
            )
        )
        outputs = []
        outputs.append(
            UIFunctionOutSingle(
                name="output_item",
                datatype=float,
                description="State change array output",
            )
        )

        return (inputs, outputs)


class PythonExpressionX(BaseTransformer):
    """
    Create a new item from an expression involving other items
    """

    def __init__(self, expression, output_name):
        self.output_name = output_name
        super().__init__()
        # convert single quotes to double
        self.expression = self.parse_expression(expression)
        # registration
        self.constants = ['expression']
        self.outputs = ['output_name']

    def execute(self, df):
        try:
            c = self._entity_type.get_attributes_dict()
        except Exception:
            c = None
        df = df.copy()
        df[self.output_name] = None
        requested = list(self.get_input_items())
        msg = self.expression + ' .'
        self.trace_append(msg)
        msg = 'Function requested items: %s . ' % ','.join(requested)
        self.trace_append(msg)
        return super().execute(df)

    def _calc(self, df):
        entity = df.index[0][0]
        logger.debug("PythonExpressionX for " + str(entity) + " expr " + self.expression)
        df[self.output_name] = eval(self.expression)
        return df

    def get_input_items(self):
        items = self.get_expression_items(self.expression)
        return items

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIExpression(name='expression',
                                   description="Define alert expression using pandas systax. Example: df['inlet_temperature']>50"))
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='output_name', datatype=float, description='Output of expression'))

        return (inputs, outputs)

class PythonFunctionX(BaseTransformer):
    """
    Execute a paste-in function. A paste-in function is python function declaration
    code block. The function must be called 'f' and accept two inputs:
    df (a pandas DataFrame) and parameters (a dict that you can use
    to externalize the configuration of the function).

    The function is supposed to return the modified DataFrame it has received as input argument !
    """

    function_name = 'f'

    def __init__(self, function_code, input_items, output_item, parameters=None):

        self.function_code = function_code
        self.input_items = input_items
        self.output_item = output_item
        super().__init__()
        if parameters is None:
            parameters = {}

        function_name = parameters.get('function_name', None)
        if function_name is not None:
            self.function_name = function_name

        self.parameters = parameters

    def execute(self, df):
        try:
            c = self._entity_type.get_attributes_dict()
        except Exception:
            c = None
        df = df.copy()
        df[self.output_item] = None
        requested = list(self.get_input_items())
        #msg = self.expression + ' .'
        #self.trace_append(msg)
        msg = 'Function requested items: %s . ' % ','.join(requested)
        self.trace_append(msg)
        return super().execute(df)

    def _calc(self, df):
        entity = df.index[0][0]
        logger.debug("PythonFunctionX for " + str(entity)) # + " expr " + self.expression)
        #df[self.output_item] = eval(self.expression)
        
        # function may have already been serialized to cos

        kw = {}

        if not self.function_code.startswith('def '):
            bucket = self.get_bucket_name()
            fn = self._entity_type.db.model_store.retrieve_model(self.function_code)
            kw['source'] = 'cos'
            kw['filename'] = self.function_code
            if fn is None:
                msg = (' Function text does not start with "def ". '
                       ' Function is assumed to located in COS'
                       ' Cant locate function %s in cos. Make sure this '
                       ' function exists in the %s bucket' % (self.function_code, bucket))
                raise RuntimeError(msg)

        else:
            fn = self._entity_type.db.make_function(function_name=self.function_name, function_code=self.function_code)
            kw['source'] = 'paste-in code'
            kw['filename'] = None

        try:
            c = self._entity_type.get_attributes_dict()
        except Exception:
            c = None
        kw['input_items'] = self.input_items
        kw['output_item'] = self.output_item
        kw['entity_type'] = self._entity_type
        kw['db'] = self._entity_type.db
        kw['c'] = c
        kw['logger'] = logger
        self.trace_append(msg=self.function_code, log_method=logger.debug, **kw)

        result = fn(df=df, parameters={**kw, **self.parameters})

        return result

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem('input_items'))
        inputs.append(UIText(name='function_code', description='Paste in your function definition'))
        inputs.append(UISingle(name='parameters', datatype=dict, required=False,
                               description='optional parameters specified in json format'))
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle('output_item', datatype=float))

        return (inputs, outputs)

