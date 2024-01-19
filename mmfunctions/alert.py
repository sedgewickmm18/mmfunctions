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
This alert module will be integrated into bif.py every now and then
'''

import re
import ast
import datetime as dt
import numpy as np
import pandas as pd
import logging

from iotfunctions.base import (BaseEvent)
from iotfunctions.ui import (UISingle, UIFunctionOutSingle, UISingleItem, UIExpression)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
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
        self.alert_end = None
        logger.info('AlertExpressionWithFilter  dim: ' + str(dimension_name) + '  exp: ' + str(expression) + '  alert: ' + str(alert_name))
        super().__init__()

    # evaluate alerts by entity
    def _calc(self, df):
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

            # get time index
            ts_ind = df.index.get_level_values(self._entity_type._timestamp)

            if self.pulse_trigger:
                # walk through all subsequences starting with the longest
                # and replace all True with True, False, False, ...
                for i in range(np_res.size, 2, -1):
                    for j in range(0, i-1):
                        if np.all(np_res[j:i]):
                            np_res[j+1:i] = np.zeros(i-j-1, dtype=int)
                            np_res[j] = i-j  # keep track of sequence length

                if self.alert_end is not None:
                    alert_end = np.zeros(np_res.size)
                    for i in range(np_res.size):
                        if np_res[i] > 0:
                            alert_end[i] = ts_ind[i]

            else:
                if self.alert_end is not None:
                    df[self.alert_end] = df.index[0]

            logger.info('AlertExpressionWithFilter  shapes ' + str(n1.shape) + ' ' + str(n2.shape) + ' ' +
                        str(np_res.shape) + '  results\n - ' + str(n1) + '\n - ' + str(n2) + '\n - ' + str(np_res))
            df[self.alert_name] = np_res

        except Exception as e:
            logger.info('AlertExpressionWithFilter  eval for ' + expr + ' failed with ' + str(e))
            df[self.alert_name] = None
            pass

        return df

    def execute(self, df):
        '''
        unused
        '''
        return super().execute(df)

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

    def __init__(self, expression, dimension_name, dimension_value, pulse_trigger, alert_name, alert_end, **kwargs):
        super().__init__(expression, dimension_name, dimension_value, alert_name, **kwargs)
        if pulse_trigger is None:
            self.pulse_trigger = True
        if alert_end is not None:
            self.alert_end = alert_end

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

class AlertOnConstant(BaseEvent):
    '''
    Create alerts that are triggered when data values the expression is True
    '''

    def __init__(self, expression_constant, alert_name, **kwargs):
        self.expression_constant = expression_constant
        self.alert_name = alert_name
        self.expression = None
        logger.info('AlertOnConstant  expression constant: ' + str(expression_constant) + '  alert: ' + str(alert_name))
        super().__init__()

    # evaluate alerts by entity
    def _calc(self, df):
        # c = self._entity_type.get_attributes_dict()
        expr = self.expression

        if expr is None:
            return df

        df = df.copy()
        logger.info('AlertExpressionWithFilter  exp: ' + expr + '  input: ' + str(df.columns))

        try:
            #evl = ast.literal_eval(expr)
            print (type(df['accelx'].values[0]))
            evl = eval(expr)
            np_res = np.where(evl, 1, 0)

            # get time index
            ts_ind = df.index.get_level_values(self._entity_type._timestamp)

            logger.info('AlertOnConstant shapes ' + str(np_res.shape) + '  results\n - ' + str(np_res))
            df[self.alert_name] = np_res

        except Exception as e:
            logger.info('AlertExpressionWithFilter  eval for ' + expr + ' failed with ' + str(e))
            df[self.alert_name] = None
            pass

        return df

    def execute(self, df):
        '''
        Load the expression constant and proceed with the superclass
        '''
        c = self._entity_type.get_attributes_dict()
        msg = ''
        try:
            expr_json= c[self.expression_constant]
            expr = expr_json['expression']
            print('Expression ' , str(expr))
            if '${' in expr:
                expr = re.sub(r"\$\{(\w+)\}", r"df['\1']", expr)
                msg = 'Expression converted to %s. ' % expr
            else:
                msg = 'Expression (%s). ' % expr
            expr = str(expr)
            logger.info('AlertOnConstant - evaluate expression: ' + expr)

        except Exception as ee:
            print('Expression not found' , str(ee))
            expr = None
            msg = 'Expression NOT FOUND'
            pass

        self.trace_append(msg)

        self.expression = expr

        return super().execute(df)

    '''
    def get_input_items(self):
        items = set(self.dimension_name)
        items = items | self.get_expression_items(self.expression)
        return items
    '''

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIExpression(name='expression_constant',
                                   description="Define alert expression to load as constant using pandas systax. \
                                                Example: df['inlet_temperature']>50. ${pressure} will be substituted \
                                                with df['pressure'] before evaluation, ${} with df[<dimension_name>]"))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name='alert_name', datatype=bool, description='Output of alert function'))
        return (inputs, outputs)


