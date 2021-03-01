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
import pandas as pd
from sqlalchemy import String

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
    Create aggregation using expression. The calculation is evaluated for
    each data_item selected. The data item will be made available as a
    Pandas Series. Refer to the Pandas series using the local variable named
    "x". The expression must return a scalar value.

    Example:

    x.max() - x.min()

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
                                  description='Choose the data items that you would like to aggregate'))

        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='name', datatype=float, description='Spectral anomaly score (z-score)'))

        return (inputs, outputs)

    def execute(self, group):
        logger.info('Execute AggregateTimeInState')
        print('Source ', self.source,  'Name ', self.name, ' Index ', group.index)

        lg = group.size
        if lg == 0:
            return 0

        # group_exp[0] = change array, group_exp[1] = timestamps
        group_exp = group.str.split(pat=',', n=1, expand=True).astype(int)
        g0 = group_exp[0].values
        g1 = group_exp[1].values
        np.savetxt('/tmp/numpy' + str(g1[0]), g0)
        group.to_csv('/tmp/testgroup' + str(g1[0]))

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
        nonzeroMin = np.min(np.nonzero(g0 != 0))
        nonzeroMax = np.max(np.nonzero(g0 != 0))

        if nonzeroMin > 0:
            print('YES1', nonzeroMin, g0[nonzeroMin])
            if g0[nonzeroMin] < 0:
                g0[0] = 1
        else:
            print('NO 1', nonzeroMin, g0[nonzeroMin])
            if g0[0] < 0:
                g0[0] = 0

        if nonzeroMax > 0:
            print('YES2', nonzeroMax, g0[nonzeroMax], g0.size)
            if g0[nonzeroMax] > 0:
                g0[-1] = -1
                # if nonzeroMax is last, ignore
                if g0[nonzeroMax] < 0:
                    g0[-1] = 0

        y = (g0 * g1).sum()
        logger.info('AggregateTimeInState returns ' + str(y) + ' seconds, computed from ' + str(g0.size))
        return abs(y)

class StateTimePrep(BaseTransformer):
    '''
    For a selected metric calculates the amount of time in minutes it has been in that  state since the last change in state.
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
                                  description='Choose the data items that you would like to aggregate'))
        inputs.append(UISingle(name='state_name', datatype=str,  description='Enter name of the state to measure time of'))

        outputs = []
        outputs.append(
            UIFunctionOutSingle(name='name', datatype=str, description='Spectral anomaly score (z-score)'))

        return (inputs, outputs)

    def _calc(self, df):
        logger.info('Execute StateTimePrep per entity')

        index_names = df.index.names
        ts_name = df.index.names[1]  # TODO: deal with non-standard dataframes (no timestamp)

        print('Source ', self.source, 'state_name ', self.state_name, 'Name ', self.name)
        #df[self.name] = (df[self.source] == self.state_name).astype(int).diff().fillna(1).astype(int)
        df_copy = df.reset_index()

        # pair of +- seconds and regular timestamp
        v1 = eval("df_copy[self.source] " + self.state_name).astype(int).diff().values.astype(int)
        #v1 = (df_copy[self.source] > 50).astype(int).diff().values.astype(int)
        # first element is NaN
        if v1.size > 0:
            v1[0] = 0
            nonzero = np.min(np.nonzero(v1 != 0))
            if v1[nonzero] > 0:
                v1[0] = -1
            else:
                v1[0] = 1

        df_copy['__intermediate1__'] = v1
        np.savetxt('/tmp/test', df_copy['__intermediate1__'].values)
        df_copy['__intermediate2__'] = (df_copy[ts_name].astype(int)// 1000000000)
        df_copy[self.name] = df_copy['__intermediate1__'].map(str) + ',' + df_copy['__intermediate2__'].map(str)
        df_copy.drop(columns=['__intermediate1__','__intermediate2__'])
        df_copy.to_csv('/tmp/testc')

        #df_copy[self.name] = change_arr
        return df_copy.set_index(index_names)
