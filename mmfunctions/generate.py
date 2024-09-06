# *****************************************************************************
# © Copyright IBM Corp. 2024.  All Rights Reserved.
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

NEGATIVE = -1000.0


class GenerateSmoothTSData(BaseTransformer):
    """
    This GenerateSmoothTSData generate smooth test data with anomalies
    """

    def __init__(self, input_item, lag, anomaly_prob, output_item):
        """
        We plan to report the anomaly only for the recent time points
        """
        # Do away with numba logs
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.INFO)

        super().__init__()

        self.anomaly_prob = anomaly_prob
        self.output_item = output_item

        self.lag = lag
        self.average_vol = 0.30  # 30 percent

        logger.debug(f'Input item: {input_item}')
        self.whoami = 'GenerateSmoothTSData'


    def get_ad_score_from_db(self, start_ts):
        """
        """
        entity_type = self.get_entity_type()
        logger.info('Retrieving old generated data from ' + str(start_ts) + ' to ')
        source_metadata = self.dms.data_items.get(self.output_item)
        derived_metric_table_name = source_metadata.get(md.DATA_ITEM_SOURCETABLE_KEY)
        schema = entity_type._db_schema
        db = None
        try:
            db = self._get_dms().db
        except Exception as le:
            logger.info('Running without db')
            return None

        raw_dataframe = None
        try:
            query, table = db.query(derived_metric_table_name, schema, column_names=['KEY', 'VALUE_N', entity_type._timestamp])
            query = query.filter(#db.get_column_object(table, entity_type._timestamp) >= start_ts,
                                 #db.get_column_object(table, entity_type._timestamp) <= end_ts,
                                 db.get_column_object(table, 'KEY') == self.output_item)

            raw_dataframe = db.read_sql_query(query.statement)
            raw_dataframe.astype({'KEY':'string', 'TIMESTAMP': 'string'}, copy=False)
            log_data_frame('Data frame after fetching from DB', raw_dataframe)
        except Exception as e:
            logger.info('Some old generated data retrieval. Error ' + str(e))
            pass

        return raw_dataframe

    def execute(self, df):
        """
        Entry point
        """

        #ans = super().execute(df)
        logger.info("Execute GenerateSmoothTSData")
        ans = df.copy()
        ans[self.output_item] = NEGATIVE - 0.01

        #log_data_frame('Result after fetching from DB', ans)
        try:
            right_db = self.get_ad_score_from_db(df.index[0:][0][1])
            if right_db is None:
                raise Exception('No old data available')
            right_db = right_db.set_index('TIMESTAMP')
            left_db = ans.droplevel(0)
            merged_df = left_db.join(right_db)
            scores_ = merged_df[[self.output_item,'value_n']].values
            result = scores_[:,1]
            loc = [i for i in range(len(scores_[:,1])) if np.isnan(scores_[:,1][i]) or scores_[:,1][i] is None]
            result[loc] = scores_[loc,0]
            ans[:][self.output_item] = result
        except Exception as e:
            logger.info('Merge of old generated data with new data did not work out. ' + str(e))
            pass

        # delegate the per instance computations to _calc()
        # group over entities
        group_base = [pd.Grouper(axis=0, level=0)]
        # and call calc
        ans = ans.groupby(group_base).apply(self._calc)
        return ans

    def _calc(self, df):
        entity = df.index[0][0]

        # get rid of entity id as part of the index
        df = df.droplevel(0)

        logger.info("Execute GenerateSmoothTSData for entity " + str(entity))
        first = None
        try:
            first = np.argwhere(df[self.output_item].values < NEGATIVE).flatten()[0]
        except Exception as e:
            # no new element, do nothing
            return df
        
        arr = df[self.output_item].values
        random_normals = np.random.randn(arr[first:].shape[0])
        #random_normals = np.where(random_normals < NEGATIVE, NEGATIVE, random_normals)

        volatility = np.sqrt(self.average_vol) * np.sqrt(1.0 / self.lag)
        random_daily_log_returns = random_normals * volatility 

        arr[first:] = random_daily_log_returns.cumsum().reshape(-1,1)
        arr = sp.signal.savgol_filter(arr, self.lag, 0, axis=0)

        # merge again
        arr_orig = df[self.output_item].values
        arr_orig[first:] = arr[first:]
        df[self.output_item] = arr_orig
        print(type(arr_orig[0]))

        return df

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
                name="lag",
                datatype=int,
                description="Minimal size of the window for interpolating data.",
            )
        )
        inputs.append(
            UISingle(
                name="anomaly_prob",
                datatype=int,
                description="Minimal size of the window for interpolating data.",
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

