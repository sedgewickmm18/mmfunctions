#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import logging
import pwd
import numpy as np
import pandas as pd
import scipy as sp
import pickle

import iotfunctions
from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEstimatorFunction, BaseSimpleAggregator)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMulti, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)
from iotfunctions.dbtables import (FileModelStore, DBModelStore)

from anomaly import SupervisedLearningTransformer

logger = logging.getLogger(__name__)
logger.info('IOT functions version ' + iotfunctions.__version__)

PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False

model_name = "Sample"
class VerySimpleModel:
    def __init__(self, min, max, median):
        self.Min = min
        self.Max = max
        self.Median = 0

class FileModelStore:
    STORE_TABLENAME = 'KPI_MODEL_STORE'

    def is_path_valid(self, pathname):
        if pathname is None or not isinstance(pathname, str) or len(pathname) == 0:
            return False
        try:
            return os.path.exists(pathname)
        except Exception:
            pass
        return False

    def __init__(self, pathname=None):
        if self.is_path_valid(pathname):
            if pathname[-1] != '/':
                pathname += '/'
            self.path = pathname
        else:
            self.path = ''
        logger.info('Init FileModelStore with path: ' + str(self.path))

    def __str__(self):
        str = 'FileModelStore path: ' + self.path + '\n'
        return str

    

    def retrieve_model(self, model_name, deserialize=True):

        model_name = model_name.replace("/",":")

        filename = self.path + self.STORE_TABLENAME + model_name

        model = None

        if os.path.exists(filename):
            f = open(filename, "rb")
            model = f.read()
            f.close()

        if model is not None:
            logger.info('Model %s of size %d bytes has been retrieved from filesystem' % (
                model_name, len(model) if model is not None else 0))
        else:
            logger.info('Model %s does not exist in filesystem' % (model_name))

        if model is not None and deserialize:
            try:
                model = pickle.loads(model)
            except Exception as ex:
                raise Exception(
                    'Deserialization of model %s that has been retrieved from ModelStore failed.' % model_name) from ex

        return model

    def delete_model(self, model_name):

        filename = self.STORE_TABLENAME + model_name
        if os.path.exists(filename):
            os.remove(filename)

        logger.info('Model %s has been deleted from filesystem' % (model_name))

class AnomalyThresholdNew(SupervisedLearningTransformer):

    def __init__(self, input_item, Min, Max, std_cycle, outlier):
        super().__init__(features=[input_item], targets=[Min, Max, std_cycle,outlier])

        self.input_item = input_item
        self.Min = Min
        self.Max = Max
        self.std_cycle = std_cycle
        self.outlier = outlier

        self.whoami = 'AnomalyThreshold'

#         logger.info(self.whoami + ' from ' + self.input_item + ' quantile threshold ' +  str(self.threshold) +
#                     ' exceeding boolean ' + self.output_item)

    def execute(self, df):
        # set output columns to zero
        logger.debug('Called ' + self.whoami + ' with columns: ' + str(df.columns))
        df[self.Min] = 0
        df[self.Max] = 0
        df[self.std_cycle] = 0
        df[self.outlier] = 0
        return super().execute(df)

    def store_model(self, model_name, model, user_name=None, serialize=True):

        if serialize:
            try:
                model = pickle.dumps(model)
            except Exception as ex:
                raise Exception(
                    'Serialization of model %s that is supposed to be stored in ModelStore failed.' % model_name) from ex

        model_name = model_name.replace("/",":")

        filename = self.path + self.STORE_TABLENAME + model_name
        print(filename)

        f = open(filename, "wb")
        f.write(model)
        f.close()
        
    def _calc(self, df):
        entity = df.index[0][0]

        # obtain db handler
        db = self.get_db()

        model_name, very_simple_model, version = self.load_model(suffix=entity)

        feature = df[self.input_item].values

        if very_simple_model is None and self.auto_train:
            # we don't do that now, the model *has* to be there
            very_simple_model = VerySimpleModel(-9.2, 6, 0)
            # --------set array for Median here

        if very_simple_model is not None:
            #self.Min[entity] = very_simple_model.Min
            df[self.Min] = very_simple_model.Min       # set the min threshold column
            #self.Max[entity] = very_simple_model.Max
            df[self.Max] = very_simple_model.Min       # set the max threshold column
            df[self.outlier] = np.logical_and(feature < very_simple_model.Max, feature < very_simple_model.Min)
            
            # ------change max condition
        path = "/Users/nishugarg/Desktop"
        filemodelstore = FileModelStore(path)
        filemodelstore.store_model(model_name, very_simple_model)
        print("I WAS HERE")
        #return super()._calc(df)
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

model_name = 'model.IOT_MSCIL_DEVICE.SupervisedLearningTransformer.MSCIL_Device_1'       
model_name = model_name.replace("/",":")
print(model_name)

if __name__ == '__main__':
    vsm = VerySimpleModel(1,2,1.5)
    vsm_pickle = pickle.dumps(vsm)
    print("I AM HERE")
    f = open(model_name, "wb")
    f.write(vsm_pickle)
    f.close()

    print('Done')
# In[ ]:
pwd


