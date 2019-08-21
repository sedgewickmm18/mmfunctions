#!/usr/bin/python3
import datetime as dt
import sys
import logging
import json
import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import BaseTransformer,BaseSimpleAggregator
from iotfunctions.metadata import EntityType
from iotfunctions.db import Database
from iotfunctions import bif
from iotfunctions import ui
from iotfunctions.enginelog import EngineLogging
from mmfunctions import functions

PACKAGE_URL= "https://github.com/sedgewickmm18/mmfunctions"

EngineLogging.configure_console_logging(logging.DEBUG)

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    np.random.seed([3,1415])
    df = pd.DataFrame(dict(
           col1 = np.random.randint(400,500,40),
           col2 = np.random.randint(400,500,40),
           #col2 = np.random.laplace(400,50,40)
        ))
    print (df)

print ("Instantiate 1")
ais = functions.AggregateItemStats(['col1','col2'],None)
print (*ais.output_items, sep = "\n")

# if in test mode call execute() and exit
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ent = ais._build_entity_type()
    ais.set_entity_type(ent)
    scal = ais.execute(df)
    print (scal)

print ("Instantiate 2")
ais = functions.AggregateItemStatsT('col1','col2','col3')

# if in test mode call execute() and exit
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ais.set_entity_type(ais._build_entity_type())
    dff = ais.execute(df)
    print (dff)
    print ("Instantiated - done")
    sys.exit()

print ("Instantiated")

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


#fn = AggregateItemStats(
#        input_item_1='x1',
#        input_item_2='x2',
#        output_item='y')

#df = fn.execute_local_test(generate_days=1,to_csv=True)
#print(df)


#cols = [
#    Column('string_1', String(255))
#        ]

#df = fn.execute_local_test(generate_days = 1,to_csv=True,
#                           columns = cols)

#db.register_functions([functions.AggregateItemStats])

db.register_module(functions)

