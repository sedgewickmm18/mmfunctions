# Singlevalued expression => base aggregators ==> built on group.agg
# Multivalued expression => complex aggregators ==> built on group.apply
#   not well supported by AS pipeline, deprecated, not supported for jobcontrol
import datetime as dt
import logging
import json
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy import fftpack
from scipy import signal
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import BaseTransformer,BaseSimpleAggregator,BaseComplexAggregator, BaseRegressor, BaseEstimatorFunction
#from iotfunctions.estimator import SimpleAnomaly
from iotfunctions.metadata import EntityType
from iotfunctions.db import Database
from iotfunctions import bif
from iotfunctions import ui
from iotfunctions.ui import UIMultiItem, UISingle ,UISingleItem, UIFunctionOutSingle, UIFunctionOutMulti
from iotfunctions.enginelog import EngineLogging
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C

# configure logging in the notebooks
#EngineLogging.configure_console_logging(logging.DEBUG)
logger = logging.getLogger(__name__)

PACKAGE_URL= "git+https://github.com/sedgewickmm18/mmfunctions@"

#credentials = {
#  "tenantId": "AnalyticsServiceDev",
#  "db2": {
#    "username": "bluadmin",
#    "password": "ZmM5MmE5NmZkZGZl",
#    "databaseName": "BLUDB",
#    "port": 50000,
#    "httpsUrl": "https://dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net:50000",
#    "host": "dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net"
#  }
#}

#with open('credentials_as_dev.json', encoding='utf-8') as F:
#    credentials = json.loads(F.read())

'''
The db_schema is the db2_schema name for your client database. If 
you are using the default schema of your user, there is no need to
provide a schema.
'''
#db_schema = None


'''
Use the credentials to build an AS Database connection.
'''

#db = Database(credentials=credentials)

'''

We will start with a simple custom function that multipies 2 numeric items.

This function performs data transformation. Data transformation functions
have an execute method that accepts a datafame as input and returns a 
modified dataframe.

The __init__ method describes the parameters of the function. This represents
the signature of the function and determines what configuration options
will be show for the function in the UI.

This sample function has three parameters:
    input_item_1, input_item_2 : The names of the input items to be multiplied
    output_item : The name of the output item produced

When producing functions that are going to be made available to AS users from
the AS catalog, your goal is make functions that are reusable on multiple
entity types. When building PythonExpression and PythonFunctions we 
could refer to specific data items in the function configuration because
we were describing a single instance of a function. With a custom catalog
function we avoid hardcoding anything specific to a particular entity type.
This function will be bound to specific items at the time of configuration.
When building the function we specify placeholders for data items.

AS will automatically generate a wizard to configure this function. To allow
AS to generate this wizard, you will need to describe a little bit more about
the arguments to the function. You do this by including a build_ui() class
method.

input_item_1 and input_item_2 are parameters that describe data items that
the function acts on. They are considered inputs to the function as they are
used by the function to produce another data item that serves as the output.

The build_ui() method returns a tuple containing two lists of arguments: 
the arguments that define the function inputs and the arguments that 
describe the output data items produced by the function.

Each function needs at least one input item and has to produce at least one
output item. The input need not be a data item, it could be a constant.
We will get to those later.

The ui module of iotfunctions contains a number of ui classes. For each 
input argument, choose an appropriate class. Since these inputs are
both single data items, we use th UISingleItem class.

There are only two output classes, UIFunctionOutSingle and UIFunctionOutMultiple.
Since the multiplication of two input data items produces a single 
output item (not an array of multiple data items), we use the 
UIFunctionOutSingle class.

       if not self._agg_dict is None and self._agg_dict:
            gf = group.agg(self._agg_dict)
            gfs.append(gf)
        for s in self._complex_aggregators:
            gf = group.apply(s.execute)
            gfs.append(gf)
        df = pd.concat(gfs,axis=1)


'''
class AggregateItemStatsT(BaseTransformer):
    '''
    Fills a new column with the pearson correlation coefficient of the two input columns
    '''
    def __init__(self, input_item_1, input_item_2, output_item ):
        super().__init__()
        self.input_item_1 = input_item_1
        self.input_item_2 = input_item_2
        self.output_item = output_item

    def execute(self, df):
        #df = df.copy()
        sca = df[self.input_item_1].corr(df[self.input_item_2])
        df[self.output_item] = sca
        #df[self.output_item] = df[self.input_item_1] * df[self.input_item_2]
        msg = 'AggregateItemStatsT found correlation to be: %d. ' %sca
        self.trace_append(msg)
        return df

    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(ui.UISingleItem(
                name = 'input_item_1',
                datatype=float,
                description = 'First column for correlation'
                                              ))
        inputs.append(ui.UISingleItem(
                name = 'input_item_2',
                datatype=float,
                description = 'Second column for correlation'
                                              ))
        #define arguments that behave as function outputs
        outputs = []
        outputs.append(ui.UIFunctionOutSingle(
                name = 'output_item',
                datatype=float,
                description='Column with the (scalar) coefficient'
                ))
        return (inputs,outputs)

class SpectralFeatureExtract(BaseTransformer):
    '''
    Employs spectral analysis to extract features from the time series data
    '''
    def __init__(self, input_item, windowsize, zscore, output_item):
        super().__init__()
        logger.debug(input_item)
        self.input_item = input_item

        # zscore - 3 deviation above mean
        self.zscore = zscore

        # use 24 by default - must be larger than 12
        self.windowsize = windowsize

        # overlap 
        self.windowoverlap = self.windowsize - self.windowsize // 12

        # assume 1 per sec for now
        self.frame_rate = 1

        self.output_item = output_item

    def execute(self, df):

        #print (df.index.levels[0])
        entities = np.unique(df.index.levels[0])
        logger.debug(entities)
        
        for entity in entities: 
            # per entity
            dfe = df.loc[[entity]].dropna(how='all')
            
            # interpolate gaps - data imputation
            #dfe.set_index('timestamp')
            dfe = dfe.reset_index(level=[0])
            Size = dfe[[self.input_item]].fillna(0).to_numpy().size
            dfe = dfe.interpolate(method='time')
            
            # one dimensional time series - named temperature for catchyness
            temperature = dfe[[self.input_item]].fillna(0).to_numpy().reshape(-1,)
            
            logger.debug(entity, self.input_item, self.windowsize, self.zscore, self.output_item, self.windowoverlap, temperature.size)
            
            if temperature.size > self.windowsize:
                # Fourier transform:
                #   frequency, time, spectral density
                freqsTS, timesTS, SxTS = signal.spectrogram(temperature, fs = self.frame_rate, window = 'hanning',
                                                        nperseg = self.windowsize, noverlap = self.windowoverlap,
                                                        detrend = False, scaling='spectrum')

                # cut off freqencies too low to fit into the window
                freqsTSb = (freqsTS > 2/self.windowsize).astype(int)
                freqsTS = freqsTS * freqsTSb
                freqsTS[freqsTS == 0] = 1 / self.windowsize

                # Compute energy = frequency * spectral density over time in decibel
                ETS = np.log10(np.dot(SxTS.T, freqsTS))

                # compute zscore over the energy
                ets_zscore = (ETS - ETS.mean())/ETS.std(ddof=0)
                logger.debug(entity, ETS, ets_zscore)

                # length of timesTS, ETS and ets_zscore is smaller than half the original
                #   extend it to cover the full original length 
                timesI = np.linspace(0, Size - 1, Size)
                zscoreI = np.interp(timesI, timesTS, ets_zscore)

                # absolute zscore > 3 ---> anomaly
                ets_zscoreb = (abs(zscoreI) > self.zscore).astype(float)
                df.loc[(entity,), self.output_item] = zscoreI

        msg = 'SpectralAnalysisFeatureExtract'
        self.trace_append(msg)
        return (df)

    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(ui.UISingleItem(
                name = 'input_item',
                datatype=float,
                description = 'Column for feature extraction'
                                              ))
        inputs.append(ui.UISingle(
                name = 'windowsize',
                datatype=int,
                description = 'Window size for spectral analysis - default 24'
                                              ))
        inputs.append(ui.UISingle(
                name = 'zscore',
                datatype=float,
                description = 'Zscore to be interpreted as anomaly'
                                              ))
        #define arguments that behave as function outputs
        outputs = []
        outputs.append(ui.UIFunctionOutSingle(
                name = 'output_item',
                datatype=float,
                description='zscore'
                ))
        return (inputs,outputs)

class KMeans2D(BaseTransformer):
    '''
    Fills a new column with the labels with K-Means centroids for the two input columns
    '''
    def __init__(self, nr_centroids, input_item_1, input_item_2, label):
        super().__init__()
        self.nr_centroids = nr_centroids
        self.input_item_1 = input_item_1
        self.input_item_2 = input_item_2
        self.label = label 

    def execute(self, df):
        dff = df[[self.input_item_1, self.input_item_2]]
        
        k_means = KMeans(init='k-means++', n_clusters = self.nr_centroids)

        dff = dff.fillna(0)
        #print (dff)

        k_means.fit(dff)
        df[self.label] = k_means.labels_

        msg = 'KMeans20'
        self.trace_append(msg)
        return df

    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(ui.UISingle(
                name = 'nr_centroids',
                datatype=int,
                description = 'Number of centroids'
                                              ))
        inputs.append(ui.UISingleItem(
                name = 'input_item_1',
                datatype=float,
                description = 'First column for correlation'
                                              ))
        inputs.append(ui.UISingleItem(
                name = 'input_item_2',
                datatype=float,
                description = 'Second column for correlation'
                                              ))
        #define arguments that behave as function outputs
        outputs = []
        outputs.append(ui.UIFunctionOutSingle(
                name = 'label',
                datatype=int,
                description='K-Means label'
                ))
        return (inputs,outputs)


class AggregateItemStats(BaseComplexAggregator):
    '''
    DO NOT USE - Compute the pearson coefficient of two variables
    '''
    
    def __init__(self,input_items,agg_dict,output_items=None):
        
        self.input_items = input_items
        self._agg_dict = agg_dict

        #if complex_aggregators is None:
        #    complex_aggregators = [self.dataframe_return_self, self.dataframe_correlation_pearson]

        #self._complex_aggregators = complex_aggregators
        self.input_items = input_items
        self.output_items = output_items
        
        #if output_items is None:
        #    output_items = ['%s_%s' %(x,aggregation_function) for x in self.input_items]
        
        #self.output_items = output_items
        self.output_items = ['correlation coefficient']
        self.aggregation_function = self.aggregate

        super().__init__()

    def get_aggregation_method(self):
        
        #out = self.get_available_methods().get(self.aggregation_function,None)
        #if out is None:
        #    raise ValueError('Invalid aggregation function specified: %s'
        #                     %self.aggregation_function)
        
        return aggregate

    def execute(self, df):

        if len(self.input_items) < 1:
            return np.nan
        else:
            return df[self.input_items[0]].corr(df[self.input_items[1]])

    def aggregate(self, df):

        if len(self.input_items) < 1:
            return np.nan
        else:
            return df[self.input_items[0]].corr(df[self.input_items[1]])
        
    @classmethod
    def build_ui(cls):
        
        inputs = []
        outputs = []
        inputs.append(UIMultiItem(name = 'input_items',
                                  datatype= None,
                                  description = ('Choose the data items'
                                                 ' that you would like to'
                                                 ' aggregate'),
                                  output_item = 'output_items',
                                  is_output_datatype_derived = True
                                          ))
                                  
        aggregate_names = list(cls.get_available_methods().keys())
                                  
        inputs.append(UISingle(name = 'aggregation_function',
                               description = 'Choose aggregation function',
                               values = aggregate_names))

        return (inputs,outputs)
    
    @classmethod
    def count_distinct(cls,series):
        
        return len(series.dropna().unique())                                  
        
    @classmethod
    def get_available_methods(cls):
        
        return {
                #'pearson' : 'stats.pearsonr'
                'pearson' : 'pearson'
                }

    #https://stackoverflow.com/questions/49925718/parallel-calculation-of-distance-correlation-dcor-from-dataframe

'''
CAVEAT: using pearson on two different time series data sets with potentially differing time stamp
event for speed and torque come at different times

Do I have to merge, resp outer join the data before I can apply pearson ?
( https://stackoverflow.com/questions/32215024/merging-time-series-data-by-timestamp-using-numpy-pandas )
'''

class GaussianProcess(BaseEstimatorFunction):
    '''
    Base class for building regression models
    '''
    eval_metric = staticmethod(metrics.r2_score)
    train_if_no_model = True
    
    def set_estimators(self):
        #gauss radial kernel
        params = {'kernel': [C(1.0, (1e-3, 1e3)), C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),  ExpSineSquared()],
                   'n_restarts_optimizer': [9]}
        self.estimators['gaussian_process'] = (GaussianProcessRegressor,params)

    def __init__(self, features, targets, predictions=None):
        super().__init__(features=features, targets=targets, predictions=predictions)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='features',
                                  datatype=float,
                                  required=True
                                  ))
        inputs.append(UIMultiItem(name='targets',
                                  datatype=float,
                                  required=True,
                                  output_item='predictions',
                                  is_output_datatype_derived=True
                                  ))
        return (inputs,[])


class AnomalyTest(BaseRegressor):

    def __init__(self, features, targets, difference,
                     predictions=None, alerts = None):
        self.feature = features
        self.target = targets
        self.difference = difference
        super().__init__(features=features, targets = targets, predictions = None)
    
    # fake - real life implementation expects model somewhere on Cloud Object Store
    #  see  BaseRegressor implementation in iotfunctions
    def execute(self,df):
            
        #df = super().execute(df)
        #for i,t in enumerate(self.targets):
        #    target = self.targets[i]
        #    print (i,t)
        #    df[self.output_items[i]] = (df[t] - df[target]).abs()
        df[self.difference] = (df[self.feature] - df[self.target]).abs()

            #alert = AlertHighValue(input_item = '_diff_',
            #                          upper_threshold = self.threshold,
            #                          alert_name = self.alerts[i])
            #alert.set_entity_type(self.get_entity_type())
            #df = alert.execute(df)
        
            #msg = 'AggregateItemStatsT found correlation to be: %d. ' %sca
            #self.trace_append(msg)
        return df
        
    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name='feature',
                                  datatype=float,
                                  required=True
                                          ))        
        inputs.append(UISingleItem(name='target',
                                  datatype=float,
                                  required=True
                                          ))
        #define arguments that behave as function outputs
        outputs = []
        outputs.append(UISingleItem(name = 'difference',
                                   datatype = float,
                                   required=True
                                          ))
            
        return (inputs,outputs)    


#db.register_functions([AggregateItemStats])

'''
After registration has completed successfully the function is available for
use in the AS UI.

The register_functions() method allows you to register more than one function
at a time. You can also register a whole module file with all of its functions.


'''

#from iotfunctions import bif
#db.register_module(bif)


'''

Note: The 'bif' module is preinstalled, so these functions will not actually
be registered by register_module.

This script covers the complete process for authoring and registering custom 
functions. The subject of the sample function used in this script was really
basic. To get an idea some of the more interesting things you can do in
custom functions, we encourage you to browse through the sample module.

'''

