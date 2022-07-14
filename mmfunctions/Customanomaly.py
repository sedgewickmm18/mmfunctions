#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import logging
import numpy as np
import pandas as pd
import scipy as sp
import logging


import iotfunctions
from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEstimatorFunction, BaseSimpleAggregator)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMulti, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)
from iotfunctions.dbtables import (FileModelStore, DBModelStore)

from .anomaly import SupervisedLearningTransformer

logger = logging.getLogger(__name__)
logger.info('IOT functions version ' + iotfunctions.__version__)

PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False


class VerySimpleModel:
    def __init__(self, min, max, cycle_list):
        self.Min = min
        self.Max = max
        self.CycleList = cycle_list

# In[ ]:

class AnomalyDetectionJ1(SupervisedLearningTransformer):

    def __init__(self, input_item, threshold, Min, Max, std_cycle, outlier):
        super().__init__(features=[input_item], targets=[Min, Max, std_cycle,outlier])

        self.input_item = input_item
        self.Min = Min
        self.Max = Max
        self.std_cycle = std_cycle
        self.outlier = outlier
        self.auto_train = True

        self.whoami = 'AnomalyDetectionJ1'


    def execute(self, df):
        # set output columns to zero
        logger.debug('Called ' + self.whoami + ' with columns: ' + str(df.columns))
        df[self.Min] = 0
        df[self.Max] = 0
        df[self.std_cycle] = 0
        df[self.outlier] = 0
        return super().execute(df)

    def _calc(self, df):
        entity = df.index[0][0]

        # obtain db handler
        db = self.get_db()
        test_model_name=self.get_model_name(suffix=entity)

#         log_stuff = 'Name of the model:' + str(test_model_name) + ', Entity Value: ' + str(entity) + ', Entity Type ' + str(self.get_entity_type())
#         logger.info(log_stuff)
#         raise Exception(log_stuff)
        model_name, very_simple_model, version = self.load_model(suffix=entity)

        feature = df[self.input_item].values

        if very_simple_model is None and self.auto_train:
            print('Here 1')

            # we don't do that now, the model *has* to be there
#             mylist = [-0.26, -0.25, -0.15, -0.6, 0.44, 0.58, 0.92, 1.72, 3.41, 5.29, 5.55, 5.47, 5.55, 5.82, 5.7, 5.52, 5.45, 5.45, 5.42, 5.2, 5.03, 4.68, 3.38, 2.39, 1.11, 0.76, -0.55, -1.1, -1.13, -1.19, -1.52, 1.04, 0.92, 0.92, 0.99, 0.84, 1.0, 0.82, 0.82, 0.88, 0.89, 0.76, 0.87, 0.8, 0.7, 0.71, 0.8, 0.7, 0.7, 0.87, 0.76, 0.6, 0.71, 0.76, 0.65, 0.6, 0.63, 0.7, 0.63, 0.6, 0.59, 0.59, 0.4, -0.03, -0.66, -0.26, -0.56, -0.62, -0.18, 0.67, 1.91, 3.57, 5.22, 5.77, 6.23, 6.4, 6.41, 4.38, 3.23, 1.44, -0.45, -2.24, -2.54, -4.52, -5.81, -6.12, -5.27, -4.44, -4.48, -4.39, -5.31, -4.42, -3.39, -1.19, -1.41, -1.41, -1.36, -1.52, -0.58, -0.63, -0.89, -0.63, 1.72, 1.73, 1.4, 0.76, 0.62, 0.77, 0.82, -0.26, -0.26, -0.27, -0.27, -0.26, -0.26, -0.27, -0.27, -0.25, -0.18, 1.4, 1.57, 1.57, 1.63, 1.5, 1.54, 1.35, 1.35, 1.29, 1.26, 1.19, 1.19, 1.08, 1.14, 1.14, 1.0, 0.92, 1.1, 0.65, 0.76, 0.6, 0.15, 0.19, 0.32, 0.37, 0.43, -0.55, -1.18, -1.33, -1.48, -1.48, -1.48, -1.48, -1.44, -1.4, -1.29, -0.14, -0.03, 0.0, -0.1, -0.21, -0.25, -0.3, -0.43, -0.55, -0.66, -0.76, -0.82, -0.8, -0.29, -0.29, -0.04, -0.15, -0.55, -0.6, -0.58, -0.45, -0.26, 0.19, 0.33, 0.23, 0.23, 0.27, 0.29, 0.27, 0.29, 0.25, 0.18, -0.12, -0.41, -0.23, -0.21, -0.4, -0.34, -0.34, -0.3, -0.41, -0.44, -0.48, -0.47, -0.44, -0.32, -0.4, -0.48, -0.52, -0.21, -0.11, -0.19, -0.47, -0.62, -0.65, -0.62, 0.01, 0.15, 0.63, 0.81, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.66, 0.6, 0.34, 0.32, 0.36, 0.36, 0.34, 0.4, 0.4, 0.43, 0.41, 0.44, 0.44, 0.43, 0.44, 0.47, 0.47, 0.47, 0.48, 0.48, 0.48, 0.52, 0.18, -1.41, -1.69, -1.48, -0.48, 0.45, 1.13, 0.93, 0.41, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15, 0.34, 0.36, 0.29, 0.36, 0.3, 0.34, 0.4, 0.4, 0.34, 0.36, 0.36, 0.37, 0.37, 0.37, 0.32, 0.37, 0.37, 0.3, 0.36, 0.3, 0.36, 0.52, 0.4, 0.84, 0.93, 0.85, 0.29, 0.21, 0.96, 0.96, 0.88, 0.08, 0.19, 0.8, 0.77, 0.77, 0.01, -0.08, 0.18, 0.27, 0.01, -0.11, -0.29, -0.43, -0.4, -0.58, -0.95, -1.04, -1.06, -1.07, -1.07, -1.07, -1.13, -1.13, -1.13, -1.18, -1.28, -1.61, -1.83, -1.96, -2.22, -2.51, -2.79, -2.94, -3.13, -3.38, -3.43, -3.67, -3.83, -3.89, -4.01, -4.02, -4.02, 0.7, 1.24, 1.14, 1.15, 0.58, 0.66, 0.89, 0.92, 1.35, 1.29, -1.33, -1.51, -1.47, -1.36, -1.4, -1.39, -1.36, -1.36, -1.35, -1.29, -0.93, -0.99, -1.06, -1.28, -1.51, -1.84, -1.85, -2.22, -2.49, -2.6, -2.98, -2.99, -3.13, -3.53, -3.68, -3.76, -3.93, -3.89, -3.68, 0.98, 1.36, 1.03, 0.18, -1.25, -1.07, -0.85, -0.74, -0.23, -0.95, -0.48, -0.23, -0.15, -0.15, -0.15, -0.08, -0.08, -0.18, -0.19, -0.38, -0.65, -0.95, -1.1, -1.24, -1.62, -1.95, -2.28, -2.47, -2.76, -2.93, -3.32, -3.61, -3.74, -3.9, -3.98, -4.19, -4.28, -4.17, -4.23, 0.66, 0.84, 0.59, -0.03, -0.71, -0.67, -0.78, -1.32, -1.52, -1.22, 1.08, 2.35, 2.18, 2.38, 2.39, 2.4, 2.38, 2.38, 1.88, 1.1, -1.62, -1.66, -1.95, -1.96, -2.25, -2.47, -2.57, -2.8, -2.88, -3.09, -3.25, -3.38, -3.61, -3.67, -3.79, -3.98, -4.0, -3.93, -3.82, 1.57, 1.62, 1.32, 1.11, -0.21, 0.15, 0.7, 1.35, 1.28, 0.74, -0.14, -1.18, -1.21, -1.21, -1.21, -1.26, -1.28, -1.21, -1.19, 0.26, 0.41, 0.63, 0.47, 0.44, 0.32, 0.21, 0.14, 0.04, 0.07, 0.0, -0.04, -0.21, -0.4, -0.77, -0.8, -0.48, -0.54, -0.55, 0.04, 0.23, 0.48, -1.68, -0.58, 0.34, 0.3, 0.04, -0.66, -0.74, -0.99, -1.15, -1.4, -1.43, -1.44, -1.44, -1.44, -1.43, -1.4, -1.33, -1.02, -0.74, -0.82, -1.02, -1.46, -1.66, -1.77, -1.95, -2.24, -2.51, -2.86, -3.08, -3.17, -3.39, -3.74, -3.89, -3.79, -3.93, -3.87, 1.08, 1.73, 1.85, 1.8, 1.5, 1.3, 1.02, 0.69, 0.27, 0.19, 0.48, 0.93, 1.13, 0.69, -1.14, -1.15, -1.29, -1.29, -1.33, -1.3, -1.29, -1.29, -1.03, -0.81, -0.8, -0.99, -1.11, -1.32, -1.88, -2.03, -2.14, -2.36, -2.71, -2.91, -3.16, -3.25, -3.38, -3.53, -3.72, -3.82, -3.83, -3.78, -2.06, 1.32, 1.92, 1.17, 0.87, 0.98, 0.18, -0.1, -0.1, 1.1, 1.17, -0.71, -1.25, -1.25, -1.22, -1.25, -1.25, -1.22, -1.22, -1.21, -1.19, -0.98, -0.7, -0.69, -0.98, -1.11, -1.26, -1.39, -1.72, -1.96, -2.2, -2.39, -2.6, -2.73, -2.93, -2.99, -3.13, -3.13, -3.25, -3.28, -1.96, 0.74, 1.25, 1.3, 1.63, 1.62, 1.07, 0.85, 0.77, 0.43, 0.38, -0.07, -0.01, 0.25, 0.25, -0.38, -0.43, -0.47, 0.52, 1.63, 1.24, -1.03, -1.32, -1.33, -1.36, -1.37, -1.37, -1.37, -1.37, -1.39, -1.36, -1.03, -1.04, -1.26, -1.46, -1.8, -2.09, -2.43, -2.62, -2.86, -3.19, -3.47, -3.65, -3.95, -4.17, -4.23, -4.39, -4.33, -4.31, -4.24, 0.98, 1.4, 1.4, 1.47, 1.47, 1.28, 1.5, 1.43, 2.07, 3.34, 3.34, 3.31, 3.08, 3.02, 3.01, 3.05, 3.03, 3.09, 3.05, 2.62, 1.62, 1.04, 0.41, 1.1, 1.21, 1.21, 1.37, 2.75, 3.42, 4.06, 4.23, 4.63, 4.44, 4.38, 4.22, 4.12, 4.04, 4.0, 3.47, 0.59, -0.04, -0.15, -0.37, -1.81, -4.55, -5.04, -5.4, -4.97, -4.92, -4.46, -4.15, -3.95, -3.67, -3.41, -3.3, -3.27, -3.27, -3.2, -3.2, -3.2, -3.3, -3.31, -3.06, -3.16, -3.09, -3.17, -3.34, -3.35, -3.34, -3.47, -3.28, -2.8, -2.31, -0.89, -0.18, 0.41, 1.03, 1.03, 1.35, 1.36, 1.36, 1.21, 1.15, 1.26, 1.37, 1.4, 2.05, 2.27, 2.38, 2.25, 2.58, 2.16, 1.62, 0.33, -1.79, -0.92, -1.25, -1.54, -1.54, -1.77, -1.54, -1.25, 0.23, 0.49, 0.71, 0.58, 0.81, 1.0, 0.8, 0.8, 0.95, 0.82, 0.87, 0.74, 0.69, 0.7, 0.7, 0.54, 0.7, 0.66, 0.49, 0.54, 0.66, 0.76, 0.65, 0.36, 0.33, 1.11, 1.15, 0.91, 0.37, 0.47, 0.63, 0.63, 0.63, 0.69, 0.63, 0.63, 0.66, 0.73, 2.13, 2.42, 2.35, 2.28, 2.31, 2.2, 2.18, 2.22, 2.22, 2.06, 1.85, 1.88, 1.88, 1.91, 1.84, 1.62, 1.74, 1.57, 1.55, 1.08, 0.71, -0.21, -0.69, -0.59, -0.49, -0.18, 0.12, 0.65, 0.76, 0.59, 0.41, -1.44, -1.47, -1.44, -1.44, -1.44, -1.44, -1.43, -1.43, -1.41, -1.1, -0.69, -0.77, -0.89, -0.93, -1.11, -1.14, -1.1, -1.39, -1.5, -1.46, -1.68, -1.61, -1.14, 0.11, 0.41, 0.3, 0.26, 0.37, 0.34, 1.25, 1.26, 0.25, -1.92, -2.16, -2.18, -2.18, -2.22, -2.22, -2.18, -2.17, -0.73, 2.79, 3.02, 3.03, 3.03, 3.08, 3.05, 3.09, 3.13, 3.08, 3.02, 2.99, 2.91, 1.72, 1.7, 1.54, 1.51, 0.63, 0.45, -0.07, -1.44, -2.11, -4.08, -4.55, -5.29, -4.77, -4.45, -3.42, -2.91, -2.57, -2.54, -3.02, -2.84, -2.77, -3.24, -2.82, -2.84, -1.37, 0.71, 1.21, 1.21, 1.47, 3.41, 5.27, 5.64, 5.63, 5.49, 5.55, 5.49, 5.55, 5.33, 5.09, 5.14, 5.29, 5.31, 5.26, 5.3, 6.15, 6.3, 5.99, 5.7, 5.42, 4.66, 2.72, 1.54, -1.18, -4.7, -4.9, -6.0, -4.01, -3.93, -3.68, -3.5, -3.32, -3.2, -3.25, -3.46, -3.57, -3.69, -2.93, -2.1, -0.89, 0.54, 1.13, 1.25, 1.28, 1.13, -3.13, -3.2, -3.21, -3.22, -3.31, -3.21, -3.09, -3.125, -3.16]
            very_simple_model = VerySimpleModel(-8.07, 7.18, 0)

            try:
                db.model_store.store_model(model_name, very_simple_model)
            except Exception as e:
                logger.error('Model store failed with ' + str(e))

            print('Here')
        else:
            print('Here 5')


        print(very_simple_model)


        if very_simple_model is not None:
            #self.Min[entity] = very_simple_model.Min
            df[self.Min] = very_simple_model.Min       # set the min threshold column
            #self.Max[entity] = very_simple_model.Max
            df[self.Max] = very_simple_model.Max       # set the max threshold column
            df[self.outlier] = np.logical_and(feature < very_simple_model.Max, feature > very_simple_model.Min)

        return df.droplevel(0)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze"))

        inputs.append(UISingle(name="threshold", datatype=int,
                               description="Threshold to determine outliers by quantile. Typically set to 0.95", ))

        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="Min", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="Max", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="std_cycle", datatype=float,
                                           description="Boolean outlier condition"))
        outputs.append(UIFunctionOutSingle(name="outlier", datatype=bool,
                                           description="Boolean outlier condition"))
        return (inputs, outputs)


# In[ ]:

