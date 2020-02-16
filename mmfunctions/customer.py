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
The Built In Functions module contains customer specific helper functions
'''

import numpy as np
# import scipy as sp

# from scipy.stats import energy_distance
from scipy import linalg

# import re
# import pandas as pd
import logging
# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer)
from iotfunctions.ui import (UIFunctionOutSingle, UISingleItem)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'
_IS_PREINSTALLED = False

FrequencySplit = 0.3
DefaultWindowSize = 12
SmallEnergy = 0.0000001


def l2normArray(df, target_col, source_col1, source_col2=None, source_col3=None):
    l2vib = []
    for index, row in df.iterrows():
        l2vib_element = linalg.norm(np.fromstring(row[source_col1].replace('[', ' ').replace(']', ''), sep=','))**2
        if source_col2 is not None:
            l2vib_element = \
             l2vib_element + linalg.norm(np.fromstring(row[source_col2].replace('[', ' ').replace(']', ''), sep=','))**2
        if source_col3 is not None:
            l2vib_element = \
             l2vib_element + linalg.norm(np.fromstring(row[source_col3].replace('[', ' ').replace(']', ''), sep=','))**2
        l2vib.append(l2vib_element**(1/2))
    df[target_col] = np.asarray(l2vib)
    return df


def unroll5minArray(df, source_col):
    l0, l1, l2, l3, l4 = [], [], [], [], []
    for source_col_entry in df[source_col].values:
        l0.append(eval(eval(source_col_entry)[0]))
        l1.append(eval(eval(source_col_entry)[1]))
        l2.append(eval(eval(source_col_entry)[2]))
        l3.append(eval(eval(source_col_entry)[3]))
        l4.append(eval(eval(source_col_entry)[4]))
    df[source_col + '_0'] = np.asarray(l0)
    df[source_col + '_1'] = np.asarray(l1)
    df[source_col + '_2'] = np.asarray(l2)
    df[source_col + '_3'] = np.asarray(l3)
    df[source_col + '_4'] = np.asarray(l4)
    return df


class L2Norm(BaseTransformer):
    '''
    Compute L2Norm of string encoded array
    '''
    def __init__(self, input_item1, input_item2, input_item3, output_item):
        super().__init__()

        self.input_item1 = input_item1
        self.input_item2 = input_item2
        self.input_item3 = input_item3

        self.output_item = output_item

    def execute(self, df):

        df = l2normArray(df, self.input_item1, self.input_item2, self.input_item3, self.output_item)

        msg = 'L2Norm'
        self.trace_append(msg)

        return (df)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
                name='input_item1',
                datatype=str,
                description='String encoded array of sensor readings'
                ))

        inputs.append(UISingleItem(
                name='input_item2',
                datatype=str,
                description='String encoded array of sensor readings'
                ))

        inputs.append(UISingleItem(
                name='input_item3',
                datatype=str,
                description='String encoded array of sensor readings'
                ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(
                name='output_item',
                datatype=float,
                description='L2 norm of the string encoded sensor readings'
                ))
        return (inputs, outputs)


class Unroll5MinArray(BaseTransformer):
    '''
    Unroll string encoded array of 5 sensor readings
    '''
    def __init__(self, input_item):
        super().__init__()

        self.input_item = input_item

    def execute(self, df):

        df = Unroll5MinArray(df, self.input_item)

        msg = 'Unroll'
        self.trace_append(msg)

        return (df)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(
                name='input_item',
                datatype=str,
                description='String encoded array of sensor readings'
                ))
        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)
