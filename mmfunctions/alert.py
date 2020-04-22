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
import pandas as pd
import logging

from iotfunctions.base import (BaseTransformer, BaseEvent)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti,
                             UIExpression)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'
_IS_PREINSTALLED = False


class AlertExpressionWithFilter(BaseEvent):
    '''
    Create alerts that are triggered when data values the expression is True
    '''

    def __init__(self, expression, dimension_name, dimension_value, alert_name, **kwargs):
        self.dimension_name = dimension_name
        self.dimension_value = dimension_value
        self.expression = expression
        self.pulse_trigger = False
        self.alert_name = alert_name
        logger.info('AlertExpressionWithFilter  dim: ' + str(dimension_name) + '  exp: ' + str(expression) + '  alert: ' + str(alert_name))
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
            n1 = np.where(evl, 1, 0)
            if self.dimension_name is None or self.dimension_value is None or \
               len(self.dimension_name) == 0 or len(self.dimension_value) == 0:
                n2 = n1
                np_res = n1
            else:
                n2 = np.where(df[self.dimension_name] == self.dimension_value, 1, 0)
                np_res = np.multiply(n1, n2)

            if self.pulse_trigger:
                # walk through all subsequences starting with the longest
                # and replace all True with True, False, False, ...
                for i in range(n1.size, 2, -1):
                    for j in range(0, i-1):
                        if np.all(n1[j:i]):
                            n1[j+1:i] = np.zeros(i-j-1, dtype=bool)
                            n1[j] = i-j  # keep track of sequence length

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


class AlertExpressionWithFilterExt(AlertExpressionWithFilter):
    '''
    Create alerts that are triggered when data values the expression is True
    '''

    def __init__(self, expression, dimension_name, dimension_value, pulse_trigger, alert_name, **kwargs):
        super().__init__(expression, dimension_name, dimension_value, alert_name, **kwargs)
        if pulse_trigger is None:
            self.pulse_trigger = True
        logger.info('AlertExpressionWithFilterExt  dim: ' + str(dimension_name) + '  exp: ' + str(expression) + '  alert: ' +
                    str(alert_name) + '  pulsed: ' + str(pulse_trigger))

    def _calc(self, df):
        '''
        unused
        '''
        return df

    def execute(self, df):
        df = super().execute(df)
        logger.info('AlertExpressionWithFilterExt  generated columns: ' + str(df.columns))
        return df

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
        inputs.append(UISingle(name='pulse_trigger',
                               description="If true only generate alerts on crossing the threshold",
                               datatype=bool))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='alert_name', datatype=bool, description='Output of alert function'))
        outputs.append(UIFunctionOutSingle(name='alert_end', datatype=dt.datetime, description='End of pulse triggered alert'))
        return (inputs, outputs)
