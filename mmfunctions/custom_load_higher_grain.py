# *****************************************************************************
# © Copyright IBM Corp. 2021.  All Rights Reserved.
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

import logging

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from iotfunctions.loader import _generate_metadata, BaseLoader

from iotfunctions.util import log_data_frame
from iotfunctions.metadata import DATA_ITEM_KPI_FUNCTION_DTO_KEY

logger = logging.getLogger(__name__)

DATA_ITEM_COLUMN_NAME_KEY = 'columnName'
DATA_ITEM_SOURCETABLE_KEY = 'sourceTableName'
KPI_FUNCTION_GRANULARITY_KEY = 'granularityName'

DATA_ITEM_DATATYPE_BOOLEAN = 'BOOLEAN'
DATA_ITEM_DATATYPE_NUMBER = 'NUMBER'
DATA_ITEM_DATATYPE_LITERAL = 'LITERAL'
DATA_ITEM_DATATYPE_TIMESTAMP = 'TIMESTAMP'

OUTPUT_COLUMN_NAME_ENTITY_ID = 'entity_id'
OUTPUT_COLUMN_NAME_TIMESTAMP = 'timestamp'
OUTPUT_COLUMN_NAME_KEY = 'key'
OUTPUT_COLUMN_NAME_VALUE_NUMERIC = 'value_n'
OUTPUT_COLUMN_NAME_VALUE_STRING = 'value_s'
OUTPUT_COLUMN_NAME_VALUE_BOOLEAN = 'value_b'
OUTPUT_COLUMN_NAME_VALUE_TIMESTAMP = 'value_t'

NP_DATETIME64_NS = 'datetime64[ns]'

TYPE_OUTPUT_COLUMN_MAP = {DATA_ITEM_DATATYPE_BOOLEAN: OUTPUT_COLUMN_NAME_VALUE_BOOLEAN,
                          DATA_ITEM_DATATYPE_NUMBER: OUTPUT_COLUMN_NAME_VALUE_NUMERIC,
                          DATA_ITEM_DATATYPE_LITERAL: OUTPUT_COLUMN_NAME_VALUE_STRING,
                          DATA_ITEM_DATATYPE_TIMESTAMP: OUTPUT_COLUMN_NAME_VALUE_TIMESTAMP}


class LoadColumnsFromHigherGrain(BaseLoader):

    NEW_TIMESTAMP_COLUMN = "###IBM###_TIMESTAMPS_FROM_INDEX"

    @classmethod
    def metadata(cls):
        return _generate_metadata(cls, {'description': 'Create new data items by joining SQL query result.', 'input': [
            {'name': 'data_item_names', 'description': 'List of data items of a higher grain which are loaded',
             'type': 'CONSTANT', 'required': True, 'dataType': 'ARRAY',
             'jsonSchema': {"$schema": "http://json-schema.org/draft-07/schema#", "title": "data_item_names",
                            "type": "array", "minItems": 1, "items": {"type": "string"}}}], 'output': [
            {'name': 'names', 'description': 'The names of the new data items.'}], 'tags': ['JUPYTER']})

    def __init__(self, names, data_item_names):
        super().__init__()

        self.logger = logging.getLogger('%s.%s' % (self.__module__, self.__class__.__name__))

        if data_item_names is not None and isinstance(data_item_names, str):
            data_item_names = [n.strip() for n in data_item_names.split(',') if len(n.strip()) > 0]

        if names is not None and isinstance(names, str):
            names = [n.strip() for n in names.split(',') if len(n.strip()) > 0]

        if data_item_names is None or not isinstance(data_item_names, list):
            raise RuntimeError(f"Input parameter data_item_names of catalog function {self.__class__.__name__} must "
                               f"be a list: {str(data_item_names)}")

        if names is None or not isinstance(names, list):
            raise RuntimeError(f"Input parameter names of catalog function {self.__class__.__name__} must "
                               f"be a list: {str(names)}")

        if len(data_item_names) != len(names):
            raise RuntimeError(f"Input parameters data_item_names and output_item_names of catalog function "
                               f"{self.__class__.__name__} must contain the same number of elements: "
                               f"data_item_names={str(data_item_names)}, output_item_names={str(names)}")

        self.data_item_names = data_item_names
        self.output_item_names = names
        self.delta = None

    def execute(self, df, start_ts, end_ts, entities):

        # Find metadata for data_item_names and create mapping between output_item_names and column names of data_items
        output_item_name_to_column_name = {}
        table_name = None
        frequency = None
        for data_item_name, output_item_name in zip(self.data_item_names, self.output_item_names):
            required_data_item = self.dms.data_items.get(data_item_name)
            required_column_name = required_data_item.get(DATA_ITEM_COLUMN_NAME_KEY)
            output_item_name_to_column_name[output_item_name] = required_column_name

            tmp_table_name = required_data_item.get(DATA_ITEM_SOURCETABLE_KEY)
            grain = required_data_item.get(DATA_ITEM_KPI_FUNCTION_DTO_KEY).get(KPI_FUNCTION_GRANULARITY_KEY)
            if grain is not None:
                tmp_frequency = grain[0]
                if tmp_frequency is None or len(tmp_frequency) == 0:
                    raise RuntimeError(f"Catalog function {self.__class__.__name__} cannot handle data items with "
                                       f"grains without frequency.")
            else:
                raise RuntimeError(f"Catalog function {self.__class__.__name__} cannot handle data items without "
                                   f"grain.")

            if table_name is None:
                table_name = tmp_table_name
                frequency = tmp_frequency
            else:
                if table_name != tmp_table_name:
                    raise RuntimeError(f"Catalog function {self.__class__.__name__} can only load data items from the "
                                       f"same table but data items {str(self.data_item_names)} have origin from at "
                                       f"least two tables {self.dms.schema}.{table_name} and "
                                       f"{self.dms.schema}.{tmp_table_name}.")

        # Get event timestamps of index
        if df.index.names[0] == self.dms.entityIdName and df.index.names[1] == self.dms.eventTimestampName:
            time_index = df.index.get_level_values(1)
        else:
            raise RuntimeError(f"Catalog function {self.__class__.__name__} currently supports data frames with index"
                               f"[{self.dms.entityIdName}, {self.dms.eventTimestampName}] only but the current index "
                               f"is {str(df.index.names)}")

        if pd.notna(time_index.min()):
            # There is at least one event timestamp in index

            # Create new column with event timestamps of index
            df[self.NEW_TIMESTAMP_COLUMN] = time_index

            # Modify event timestamps in new column because we want to match the beginning of a grain interval of the
            # higher grain: First move event timestamp backwards by one unit to hit the previous interval; then set
            # trailing units to zero to hit the beginning of that interval
            if frequency == "H":
                df[self.NEW_TIMESTAMP_COLUMN] = (df[self.NEW_TIMESTAMP_COLUMN] - pd.Timedelta(hours=1)).map(
                    self._truncate_to_hour, na_action='ignore')
            elif frequency == "D":
                df[self.NEW_TIMESTAMP_COLUMN] = (df[self.NEW_TIMESTAMP_COLUMN] - pd.Timedelta(days=1)).map(
                    self._truncate_to_day, na_action='ignore')
            elif frequency == "MS":
                df[self.NEW_TIMESTAMP_COLUMN] = (df[self.NEW_TIMESTAMP_COLUMN] - MonthEnd()).map(
                    self._truncate_to_month, na_action='ignore')
            else:
                raise RuntimeError(f"Frequency {str(frequency)} is currently not supported in catalog function "
                                   f"{self.__class__.__name__}")

            # Get earliest and latest timestamp to retrieve from data base
            earliest_timestamp = df[self.NEW_TIMESTAMP_COLUMN].min()
            latest_timestamp = df[self.NEW_TIMESTAMP_COLUMN].max()

            # load required data items from table
            loaded_df = self._get_calc_metric_data({DATA_ITEM_DATATYPE_NUMBER: output_item_name_to_column_name},
                                                   self.dms.schema, table_name, earliest_timestamp, latest_timestamp,
                                                   None)

            # Merge loaded_df into df by matching entity ids and timestamps of loaded df with modified event
            # timestamps of df.
            df_index_names = df.index.names
            df = df.reset_index()

            loaded_df.rename(columns={self.dms.eventTimestampName: self.NEW_TIMESTAMP_COLUMN}, inplace=True)

            df = df.merge(loaded_df, left_on=[self.dms.entityIdName, self.NEW_TIMESTAMP_COLUMN],
                          right_on=[self.dms.entityIdName, self.NEW_TIMESTAMP_COLUMN], how='left')

            df.set_index(keys=df_index_names, inplace=True)
            log_data_frame('df after merge of higher grain ',df.head(30))   # kohlmann remove
            df.drop(columns=[self.NEW_TIMESTAMP_COLUMN], inplace=True)
        else:
            # Empty data frame, just add columns for consistency
            for output_item_name in self.output_item_names:
                df[output_item_name] = np.nan

        return df

    @staticmethod
    def _truncate_to_hour(input_ts):
        return pd.Timestamp(year=input_ts.year, month=input_ts.month, day=input_ts.day, hour=input_ts.hour)

    @staticmethod
    def _truncate_to_day(input_ts):
        return pd.Timestamp(year=input_ts.year, month=input_ts.month, day=input_ts.day)

    @staticmethod
    def _truncate_to_month(input_ts):
        return pd.Timestamp(year=input_ts.year, month=input_ts.month, day=1)

    def _get_calc_metric_data(self, data_items_per_type, schema_name, table_name, start_ts, end_ts, entities=None):

        dfs = []
        for data_type, data_item_col_mapping in data_items_per_type.items():

            if data_item_col_mapping is None or len(data_item_col_mapping) == 0:
                # No data items requested for this data type
                continue

            # Retrieve the data items for this data type from the corresponding output table
            value_col = TYPE_OUTPUT_COLUMN_MAP[data_type]
            column_names = [OUTPUT_COLUMN_NAME_ENTITY_ID, OUTPUT_COLUMN_NAME_TIMESTAMP, OUTPUT_COLUMN_NAME_KEY,
                            value_col]
            filters = {OUTPUT_COLUMN_NAME_KEY: list(data_item_col_mapping.keys())}

            query, table = self.dms.db.query(table_name, schema_name, timestamp_col=OUTPUT_COLUMN_NAME_TIMESTAMP,
                                             start_ts=start_ts, end_ts=end_ts, entities=entities, filters=filters,
                                             column_names=column_names, column_aliases=column_names,
                                             deviceid_col=OUTPUT_COLUMN_NAME_ENTITY_ID)
            df = self.dms.db.read_sql_query(sql=query.statement)

            if not df.empty:
                # The data frame must be pivoted to achieve a data frame with one column per metric
                # pd.pivot_table() can only handle numeric value columns. There is another function df.pivot() which
                # can handle non-numeric value columns but pandas 1.1.0 or higher is required to handle multi-index.
                # Switch to df.pivot() once we have pandas 1.1.0 or higher available.
                df = pd.pivot_table(df, values=value_col,
                                    index=[OUTPUT_COLUMN_NAME_ENTITY_ID, OUTPUT_COLUMN_NAME_TIMESTAMP],
                                    columns=OUTPUT_COLUMN_NAME_KEY)

                # Rename columns to name of data items
                col_data_item_mapping = {y: x for x, y in data_item_col_mapping.items()}
                df.rename(columns=col_data_item_mapping, inplace=True, errors='ignore')
            else:
                df = pd.DataFrame(columns=[OUTPUT_COLUMN_NAME_ENTITY_ID, OUTPUT_COLUMN_NAME_TIMESTAMP])
                df = df.astype({OUTPUT_COLUMN_NAME_ENTITY_ID: str, OUTPUT_COLUMN_NAME_TIMESTAMP: 'datetime64[ns]'})
                df.set_index(keys=[OUTPUT_COLUMN_NAME_ENTITY_ID, OUTPUT_COLUMN_NAME_TIMESTAMP], inplace=True)
            # Add columns for missing metrics (metric might be missing when there was no value for this metric in the
            # given time frame in the output table)
            missing_cols = set(data_item_col_mapping.keys()).difference(df.columns)
            for missing_column in missing_cols:
                df[missing_column] = np.nan

            log_data_frame("Pivoted data frame", df.head())

            dfs.append(df)

        if len(dfs) == 0:
            df_final = pd.DataFrame(columns=[self.dms.entityIdName, self.dms.eventTimestampName])
            df_final.astype({self.dms.entityIdName: str, self.dms.eventTimestampName: NP_DATETIME64_NS}, copy=False)
        else:
            # Merge the data frames together into one data frame
            if len(dfs) == 1:
                df_final = dfs[0]
            else:
                df_final = dfs[0].join(dfs[1:], how="outer")

            # Move data frame's index to columns and rename those columns
            df_final.reset_index(inplace=True)
            df_final.rename(columns={OUTPUT_COLUMN_NAME_ENTITY_ID: self.dms.entityIdName,
                                     OUTPUT_COLUMN_NAME_TIMESTAMP: self.dms.eventTimestampName}, inplace=True)

        log_data_frame("Merged data frames:", df_final.head())

        return df_final
