# *****************************************************************************
# Â© Copyright IBM Corp. 2018.  All Rights Reserved.
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

# import datetime as dt
# import time
# from collections import OrderedDict
import numpy as np
# import re
# import pandas as pd
import logging
# import warnings
# from sqlalchemy import String

from iotfunctions.base import (BaseTransformer)
from iotfunctions.ui import (UISingle, UIFunctionOutSingle, UISingleItem)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'
_IS_PREINSTALLED = False


def injectAnomaly(input_array, factor = None, size = None, width = None):

    # Create NaN padding for reshaping
    # nan_arr = np.repeat(np.nan, factor - input_array.size % factor)

    # Prepare numpy array to reshape
    # a_reshape_arr = np.append(input_array, nan_arr)

    lim_size = input_array.size - input_array.size % factor
    a_reshape_arr = input_array[:lim_size]

    # Final numpy array to be transformed into 2d array
    a1 = np.reshape(a_reshape_arr, (-1, factor)).T

    out_width = 0
    if width is None:
        # Calculate 'local' standard deviation if it exceeds 1 to generate anomalies
        std = np.std(a1, axis=0)
        stdvec = np.maximum(np.where(np.isnan(std), 1, std), np.ones(a1[0].size))

        # Mark Extreme anomalies
        a1[0] = np.multiply(a1[0], np.multiply(np.random.choice([-1, 1], a1.shape[1]), stdvec * size))
    else:
        out_width = width
        for i in range(1, width):
            a1[i] = np.nan

    # Flattening back to 1D array
    output_array = input_array.copy()
    output_array[0:lim_size] = a1.T.flatten()

    # Removing NaN padding
    # output_array = output_array[~np.isnan(output_array)]

    return out_width, output_array


class AnomalyGeneratorExtremeValue(BaseTransformer):
    '''
    This function generates extreme anomaly.
    '''

    def __init__(self, input_item, factor, size, output_item):
        self.input_item = input_item
        self.output_item = output_item
        self.factor = int(factor)
        self.size = int(size)
        self.count = None   # allow to set count != 0 for unit testing
        super().__init__()

    def execute(self, df):

        logger.debug('Dataframe shape {}'.format(df.shape))

        entity_type = self.get_entity_type()
        derived_metric_table_name = 'DM_' + entity_type.logical_name
        schema = entity_type._db_schema

        # store and initialize the counts by entity id
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
            logger.debug('Reintialize count')

        counts_by_entity_id = None
        try:
            counts_by_entity_id = db.model_store.retrieve_model(key)
        except Exception as e2:
            counts_by_entity_id = self.count
            logger.error('Counts by entity id not yet initialized - error: ' + str(e2))
            pass

        if counts_by_entity_id is None:
            counts_by_entity_id = {}
        logger.debug('Initial Grp Counts {}'.format(counts_by_entity_id))

        timeseries = df.reset_index()
        timeseries[self.output_item] = timeseries[self.input_item]
        df_grpby = timeseries.groupby('id')
        for grp in df_grpby.__iter__():

            entity_grp_id = grp[0]
            df_entity_grp = grp[1]

            # Initialize group counts
            count = 0
            if entity_grp_id in counts_by_entity_id:
                count = counts_by_entity_id[entity_grp_id]

            # Start index based on counts and factor
            if count == 0 or count % self.factor == 0:
                strt_idx = 0
            else:
                strt_idx = self.factor - count % self.factor

            # Prepare numpy array for marking anomalies
            actual = df_entity_grp[self.output_item].values
            a = actual[strt_idx:]

            if a.size < self.factor:
                logger.info('Not enough new data points to generate more anomalies')
                continue   # try next time with more data points

            # Update group counts for storage
            count += actual.size
            counts_by_entity_id[entity_grp_id] = count

            _, a2 = injectAnomaly(a, factor=self.factor, size=self.size)

            # Adding the missing elements to create final array
            final = np.append(actual[:strt_idx], a2)
            # Set values in the original dataframe
            try:
                timeseries.loc[df_entity_grp.index, self.output_item] = final
            except Exception as ee:
                logger.error('Could not set anomaly because of ' + str(ee) + '\nSizes are ' + str(final.shape) + ',' + str(actual.shape))
                pass

        logger.debug('Final Grp Counts {}'.format(counts_by_entity_id))

        # save the group counts to db
        try:
            db.model_store.store_model(key, counts_by_entity_id)
        except Exception as e3:
            logger.error('Counts by entity id cannot be stored - error: ' + str(e3))
            pass

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

            # Start index based on counts and factor
            if width == self.width:
                width = 0
                count += 1

            if count == 0 or count % self.factor == 0:
                strt_idx = 0
            else:
                strt_idx = self.factor - count % self.factor

            # Prepare numpy array for marking anomalies
            actual = df_entity_grp[self.output_item].values
            a = actual[strt_idx:]

            if a.size < self.factor:
                logger.info('Not enough new data points to generate more anomalies')
                continue   # try next time with more data points

            # Update group counts for storage
            count += actual.size
            counts_by_entity_id[entity_grp_id] = count

            width, a2 = injectAnomaly(a, factor=self.factor, width=self.width)

            if False:
                mark_anomaly = False
                for grp_row_index in df_entity_grp.index:
                    count += 1

                    if width != self.width or count % self.factor == 0:
                        # start marking points
                        mark_anomaly = True

                    if mark_anomaly:
                        timeseries[self.output_item].iloc[grp_row_index] = np.NaN
                        width -= 1
                        # logger.debug('Anomaly Index Value{}'.format(grp_row_index))

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
                    # logger.debug('Anomaly Index Value{}'.format(grp_row_index))

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
