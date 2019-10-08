#!/usr/bin/python3
import datetime as dt
import sys
import logging
import json
import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
#from iotfunctions.base import BaseTransformer,BaseSimpleAggregator
#from iotfunctions.bif import SimpleAnomaly
#from iotfunctions.metadata import EntityType
from iotfunctions.db import Database
#from iotfunctions import bif
#from iotfunctions import ui
from iotfunctions.enginelog import EngineLogging
from mmfunctions import functions

PACKAGE_URL= "https://github.com/sedgewickmm18/mmfunctions"

credentials = {
  "tenantId": "AnalyticsServiceDev",
  "as_api_host": "https://api-dev.connectedproducts.internetofthings.ibmcloud.com",
  "as_api_key": "a-69xgm4-8bdgtvnsv4",
  "as_api_token": "9X_tMKdupOiJ!mzaPV",
  "config" : {
      "objectStorageEndpoint" : "https://s3-api.us-geo.objectstorage.softlayer.net",
      "bos_runtime_bucket" : "analytics-runtime-analyticsservicedev-799d2008b460",
      "bos_logs_bucket" : "analytics-logs-analyticsservicedev-32703c52ec8b"
  },
  "objectStorage": {
      "username" : "58ddd86b5de8468b819d385046f17033",
      "password" : "ee0d6c5521ce9ff100f91b0e37d4eb8cc1a038b5a6d05b38",
      "region" : "us",
      "endpoint" : "https://s3-api.us-geo.objectstorage.softlayer.net"
  },
  "db2": {
    "username": "bluadmin",
    "password": "ZmM5MmE5NmZkZGZl",
    "databaseName": "BLUDB",
    "port": 50000,
    "httpsUrl": "https://dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net:50000",
    "host": "dashdb-enterprise-yp-dal13-74.services.dal.bluemix.net"
  }
}


EngineLogging.configure_console_logging(logging.DEBUG)

'''
The db_schema is the db2_schema name for your client database. If 
you are using the default schema of your user, there is no need to
provide a schema.
'''
db_schema = None


'''
Use the credentials to build an AS Database connection.
'''

if (len(sys.argv) <= 1) or (sys.argv[1] != 'test'):
   db = Database(credentials=credentials)
   print (db.cos_load)


# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    np.random.seed([3,14323])
    df = pd.DataFrame(dict(
           col1 = np.random.randint(400,500,40),
           col2 = np.random.randint(400,500,40),
           #col2 = np.random.laplace(400,50,40)
        ))
    print (df)

df2 = pd.read_csv('./anomalyoutput.csv',
                  parse_dates=['timestamp_date','metricTS_date'],
                  dtype={"$O-AnomalyIndex":float})
df2 = df2.sort_values(by='timestamp')

print ("Instantiate 1")
ais = functions.AggregateItemStats(['col1','col2'],None)
print (*ais.output_items, sep = "\n")

# if in test mode call execute() and exit
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ent = ais._build_entity_type()
    ais.set_entity_type(ent)
    scal = ais.execute(df)
    print (scal)
    print ("Instantiated - done")

print ("Instantiate 2")
ais = functions.AggregateItemStatsT('col1','col2','col3')

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ais.set_entity_type(ais._build_entity_type())
    dff = ais.execute(df)
    print (dff)
    print ("Instantiated - done")

print ("Instantiate 3")
ais = functions.KMeans2D(10, 'col1','col2','col4')

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ais.set_entity_type(ais._build_entity_type())
    dff = ais.execute(df)
    print (dff)
    print ("Instantiated - done")

print ("Instantiate 4")
ais = functions.GaussianProcess(['col1'],['col2'],['col5'])

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test') and False:
    Let = ais._build_entity_type()
    Let.db = db
    ais.set_entity_type(Let)
    dff = ais.execute(df)
    print (dff)
    print ("Instantiated - done")

print ("Instantiate 5")
ais = functions.AnomalyTest('col1','col2','col6')

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ais.set_entity_type(ais._build_entity_type())
    dff = ais.execute(df)
    print (dff)
    print ("Instantiated - done")

print ("Instantiate 6")
ais = functions.SpectralFeatureExtract('Val', zscore=3, windowsize=24, output_item='zscore')

# if in test mode call execute()
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    ais.set_entity_type(ais._build_entity_type())
    dff = ais.execute(df2)
    print (dff)
    print ("Instantiated - done")

print ("Instantiated")

# if there is a 2nd argument do not register but exit
if (len(sys.argv) > 1):
    sys.exit()

EngineLogging.configure_console_logging(logging.DEBUG)

#with open('credentials_as_dev.json', encoding='utf-8') as F:
#    credentials = json.loads(F.read())

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

