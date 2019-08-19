import datetime as dt
import logging
import json
import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import BaseTransformer,BaseSimpleAggregator
from iotfunctions.metadata import EntityType
from iotfunctions.db import Database
from iotfunctions import ui,bif
from iotfunctions.enginelog import EngineLogging

EngineLogging.configure_console_logging(logging.DEBUG)

PACKAGE_URL= "https://github.com/sedgewickmm18/mmfunctions"

credentials = {
  "tenantId": "AnalyticsServiceDev",
  "db2": {
    "username": "bluadmin",
    "password": "ZmM5MmE5NmZkZGZl",
    "databaseName": "BLUDB",
    "port": 50000,
    "httpsUrl": "https://dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net:50000",
    "host": "dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net"
  }
}

#with open('credentials_as_dev.json', encoding='utf-8') as F:
#    credentials = json.loads(F.read())

'''
The db_schema is the db2_schema name for your client database. If 
you are using the default schema of your user, there is no need to
provide a schema.
'''
db_schema = None


'''
Use the credentials to build an AS Database connection.
'''

db = Database(credentials=credentials)

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

'''

class AggregateItemStats(BaseSimpleAggregator):
    '''
    Compute the pearson coefficient of two variables
    '''
    
    def __init__(self,input_items,output_items=None):
        
        super().__init__()
        
        self.input_items = input_items
        self.aggregation_function = aggregation_function
        
        if output_items is None:
            output_items = ['%s_%s' %(x,aggregation_function) for x in self.input_items]
        
        self.output_items = output_items
        
    def get_aggregation_method(self):
        
        out = self.get_available_methods().get(self.aggregation_function,None)
        if out is None:
            raise ValueError('Invalid aggregation function specified: %s'
                             %self.aggregation_function)
        
        return out 
        
    @classmethod
    def build_ui(cls):
        
        inputs = []
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
        
        return (inputs,[])
    
    @classmethod
    def count_distinct(cls,series):
        
        return len(series.dropna().unique())                                  
        
    @classmethod
    def get_available_methods(cls):
        
        return {
                'pearson' : 'stats.pearsonr'
                }



'''

This python file is a script as it is executable. Each time AS runs the
MultiplyTwoItems function, it will not be executing this script.
Instead it will be importing the MultiplyTwoItems class from a python module
inside a python package. To include functions in the AS catalog, you 
will need to build a python package and include this class in module of that
package.

I defined the above class in this script just for so that I could use it to 
show the structure of an AS function. To actually test this class, we will not
use the version of it that I copied into the script, we will use the official
version of it - the one that exists in the sample.py. 

'''

#from iotfunctions.sample import AggregateItemStats

'''

We have just replaced the MultiplyToItems class with the official version
from the sample package. To understand the structure and content of the
sample package look at sample.py

To test MultiplyTowItems you will need to create an instance of it and
indicate for this instance, which two items will it multiply and what
will the result data item be called.

'''

#fn = AggregateItemStats(
#        input_item_1='x1',
#        input_item_2='x2',
#        output_item='y')

#df = fn.execute_local_test(generate_days=1,to_csv=True)
#print(df)

'''

This automated local test builds a client side entity type,
adds the function to the entity type, generates data items that match
the function arguments, generates a dataframe containing the columns
referenced in these arguments, executes the function on this data and
writes the execution results to a file.

Note: Not all functions can be tested locally like this as some require
a real entity type with real tables and a real connection to the AS service

'''

''' 

The automated test assumes that data items are numeric. You can also
specify datatypes by passing a list of SQL Alchemy column objects.

'''

#cols = [
#    Column('string_1', String(255))
#        ]

#df = fn.execute_local_test(generate_days = 1,to_csv=True,
#                           columns = cols)

'''
Custom functions must be registered in the AS function catalog before
you can use them. To register a function:
    
'''

db.register_functions([AggregateItemStats])

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

