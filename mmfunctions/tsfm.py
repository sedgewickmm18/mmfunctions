# *****************************************************************************
# © Copyright IBM Corp. 2024  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0 license
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

"""
The experimental functions module contains (no surprise here) experimental functions
"""

import datetime as dt
import logging
import re
import time
import warnings
import ast
import os
import subprocess
import importlib
from collections import OrderedDict

import numpy as np
import scipy as sp
import pandas as pd
from sqlalchemy import String

import torch

from iotfunctions.base import (BaseTransformer, BaseEvent, BaseSCDLookup, BaseSCDLookupWithDefault, BaseMetadataProvider,
                   BasePreload, BaseDatabaseLookup, BaseDataSource, BaseDBActivityMerge, BaseSimpleAggregator,
                   DataExpanderTransformer)
from iotfunctions.bif import (InvokeWMLModel)
from iotfunctions.loader import _generate_metadata
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti, UIMulti, UIExpression,
                 UIText, UIParameters)
from iotfunctions.util import adjust_probabilities, reset_df_index, asList, UNIQUE_EXTENSION_LABEL

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git@'

# Do away with numba and onnxscript logs
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.INFO)
onnx_logger = logging.getLogger('onnxscript')
onnx_logger.setLevel(logging.ERROR)


def install_and_activate_granite_tsfm():
    return True

class TSFMZeroShotScorer(InvokeWMLModel):
    """
    Call time series foundation model
    """
    def __init__(self, input_items, output_items=None, context=512, horizon=96, watsonx_auth=None):
        logger.debug(str(input_items) + ', ' + str(output_items) + ', ' + str(context) + ', ' + str(horizon))

        super().__init__(input_items, watsonx_auth, output_items)

        self.context = context
        if context <= 0:
            self.context = 512
        self.horizon = horizon
        if horizon <= 0:
            self.horizon = 96
        self.whoami = 'TSFMZeroShot'

        # allow for expansion of the dataframe
        self.allowed_to_expand = True
    
        self.init_local_model = install_and_activate_granite_tsfm()
        self.model = None              # cache model for multiple calls


    # ask for more data if we do not have enough data for context and horizon
    def check_size(self, size_df):
        return min(size_df) < self.context + self.horizon

    # TODO implement local model lookup and initialization later
    # initialize local model is a NoOp for superclass
    def initialize_local_model(self):
        logger.info('initialize local model')
        try:
            from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
            TTM_MODEL_REVISION = "main"
            # Forecasting parameters
            #context_length = 512
            #forecast_length = 96
            #install_and_activate_granite_tsfm()
            self.model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", cache_dir='/tmp', revision=TTM_MODEL_REVISION)
        except Exception as e:
            logger.error("Failed to load local model with error " + str(e))
            return False
        logger.info('local model ready')
        return True

    # inference on local model 
    def call_local_model(self, df):
        logger.info('call local model')

        logger.debug('df columns  ' + str(df.columns))
        logger.debug('df index ' + str(df.index.names))

        # size of the df should be fine
        len = self.context + self.horizon

        if self.model is not None:

            logger.debug('Forecast ' + str(df.shape[0]/self.horizon) + ' times')

            for i in range(self.context, df.shape[0], self.horizon):
                inputtensor_ = torch.from_numpy(df[i-self.context:i][self.input_items].values.astype(np.float))
                #logger.debug('shape   input ' + str(inputtensor_.shape))
                # add dimension
                #inputtensor = inputtensor_[None,:self.context,:]              # only the historic context
                inputtensor = inputtensor_[None,:,:]              # only the historic context
                #logger.debug('shape   input ' + str(inputtensor.shape))
                outputtensor = self.model(inputtensor)['prediction_outputs']  # get the forecasting horizon back
                #logger.debug('shapes   input ' + str(inputtensor.shape) + ' , output ' + str(outputtensor.shape))
                #   and update the dataframe with it
                #df.loc[df.tail(self.horizon).index, self.output_items] = outputtensor[0].detach().numpy()
                try:
                    df.loc[df[i:i + self.horizon].index, self.output_items] = outputtensor[0].detach().numpy()
                except:
                    logger.debug('Issue with ' + str(i) + ':' + str(i+self.horizon))
                    pass

        return df

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []

        inputs.append(UIMultiItem(name='input_items', datatype=float, required=True, output_item='output_items',
                                  is_output_datatype_derived=True))
        inputs.append(
            UISingle(name='context', datatype=int, required=False, description='Context - past data'))
        inputs.append(
            UISingle(name='horizon', datatype=int, required=False, description='Forecasting horizon'))
        inputs.append(UISingle(name='watsonx_auth', datatype=str,
                               description='Endpoint to the WatsonX service where model is hosted', tags=['TEXT'], required=True))

        # define arguments that behave as function outputs
        outputs=[]
        #outputs.append(UISingle(name='output_items', datatype=float))
        return inputs, outputs

