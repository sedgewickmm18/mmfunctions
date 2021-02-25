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

    def __init__(self, input_items, expression=None, output_items=None):
        super().__init__()

        self.input_items = input_items
        self.expression = expression
        self.output_items = output_items

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(UIMultiItem(name='input_items', datatype=None, description=('Choose the data items'
                                                                                  ' that you would like to'
                                                                                  ' aggregate'),
                                  output_item='output_items', is_output_datatype_derived=True))

        inputs.append(UIExpression(name='expression', description='Paste in or type an AS expression'))

        return (inputs, [])

    def aggregate(self, x):
        return eval(self.expression)


