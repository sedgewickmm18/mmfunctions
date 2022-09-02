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

PACKAGE_URL = 'git+https://github.ibm.com/IBM-India-CTO-ENG-Assets/MSCIL_Custom_Function.git'
_IS_PREINSTALLED = False

class VerySimpleModel:
    def __init__(self, min, max, cycle_list):
        self.Min = min
        self.Max = max
        self.CycleList = cycle_list

class AnomalyForq1(SupervisedLearningTransformer):

    def __init__(self, input_item, Min, Max, std_cycle, outlier):
        super().__init__(features=[input_item], targets=[Min, Max, std_cycle,outlier])

        self.input_item = input_item
        self.Min = Min
        self.Max = Max
        self.std_cycle = std_cycle
        self.outlier = outlier
        self.auto_train = True
        self.whoami = 'AnomalyForq1'


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
            very_simple_model = VerySimpleModel(-2.29, 2.58, 0)

            try:
                db.model_store.store_model(model_name, very_simple_model)
            except Exception as e:
                logger.error('Model store failed with ' + str(e))
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