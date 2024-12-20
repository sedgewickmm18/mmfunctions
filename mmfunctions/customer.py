# *****************************************************************************
# © Copyright IBM Corp. 2018-2023.  All Rights Reserved.
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

from collections import OrderedDict
import json
import datetime as dt
import pytz
import base64
import ast
import math

# import re
import numpy as np
import pandas as pd
import logging

# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer)
from iotfunctions.anomaly import (SupervisedLearningTransformer)
from iotfunctions.ui import (UIMultiItem, UISingle)

# ONNX runtime
import onnxruntime as rt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False


class ONNXRegressor(SupervisedLearningTransformer):
    """
    Return first non-null value from a list of data items.
    """

    def __init__(self, features, targets, predictions, confidence_band_prefix):

        if targets is None:
            targets = ['output_item']
        elif not isinstance(targets, list):
            targets = [targets]

        prediction_prefix = 'pred_'
        if predictions is None:
            predictions = [prediction_prefix + x for x in targets]
        if confidence_band_prefix is None:
            confidence_band_prefix = 'conf_'

        confidences = [confidence_band_prefix + x for x in targets]

        super().__init__(features, targets, predictions)
        self.whoami = 'ONNXRegressor'
        self.confidences = confidences

    def load_model(self, suffix=None):
        # TODO: Lift assumption there is only a single target
        model_name = self.generate_model_name([], self.targets[0], prefix='model', suffix=suffix)

        my_model = None
        db = self.get_db()
        try:
            my_model = db.model_store.retrieve_model(model_name, deserialize=False)  # no unpickling for ONNX binary models
            logger.info('load model %s' % str(my_model))
        except Exception as e:
            logger.error('Model retrieval failed with ' + str(e))
            pass

        return model_name, my_model

    def _calc(self, df):
        # per entity - copy for later inplace operations
        entity = df.index[0][0]

        # obtain db handler
        db = self.get_db()

        model_name, onnx_model = self.load_model(suffix=entity + '.onnx')

        features = df[self.features].values.astype(np.float32)
        #logger.info("Get features " + str(self.features) + ":" + str(features))

        if onnx_model is None:
            logger.error('ONNX model not available')
            df[self.predictions] = -0.000001
            df[self.confidences] = -0.000001

        else:
            # pass data to the model
            options = rt.SessionOptions()
            options.enable_profiling = True
            session = rt.InferenceSession(onnx_model, sess_options=options)
            ortvalue = rt.OrtValue.ortvalue_from_numpy(features)

            input_names = [x.name for x in session.get_inputs()]
            output_names = [x.name for x in session.get_outputs()]
            logger.info("Apply model: features " + str(input_names) +
                "  feature shape " + str(features.shape) + ", predictions: " + str(output_names))
            try:
                outputs = session.run(output_names, {input_names[0]: features})
                #logger.info("Output[0] shape " + str(outputs[0].shape) + ", first value " + str(outputs[0,0]))
                df[self.predictions] = outputs[0]
                if len(outputs) > 1:
                    df[self.confidences] = outputs[1]
                else:
                    df[self.confidences] = 0
            except Exception as e:
                logger.error("ONNX evaluation failed with " + str(e))

        return df.droplevel(0)

    def execute(self, df):
        logger.debug('Execute ' + self.whoami)

        # obtain db handler
        db = self.get_db()

        # check data type
        #if df[self.input_item].dtype != np.float64:
        for feature in self.features:
            if not pd.api.types.is_numeric_dtype(df[feature].dtype):
                logger.error('Regression on non-numeric feature:' + str(feature))
                return (df)

        df_copy = df.fillna(0)
        logger.info(self.whoami + ' Inference, Features: ' + str(self.features) + ' Targets: ' + str(self.targets) +
            ' Predictions: ' + str(self.predictions) + ' Confidences: ' + str(self.confidences))

        #logger.info('DF(' + str(self.features) + "): " + str(df_copy[self.features].values))

        missing_cols = [x for x in self.targets + self.predictions + self.confidences if x not in df_copy.columns]
        for m in missing_cols:
            df_copy[m] = None

        # delegate to _calc
        logger.debug('Execute ' + self.whoami + ' enter per entity execution')

        # group over entities
        group_base = [pd.Grouper(axis=0, level=0)]

        df_copy = df_copy.groupby(group_base).apply(self._calc)

        df_copy = df_copy.fillna(0)

        logger.info("Predictions: Column " + str(self.predictions) + " Content " + str(df_copy[self.predictions].values))

        logger.debug('Scoring done')

        return df_copy

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features', datatype=float, required=True))
        inputs.append(UIMultiItem(name='targets', datatype=float, required=True, output_item='predictions',
                                  is_output_datatype_derived=True))
        inputs.append(UISingle(name='confidence_band_prefix', datatype=str, description='Prefix for confidence band (default: conf_)'))

        # define arguments that behave as function outputs
        outputs = []
        return inputs, outputs

