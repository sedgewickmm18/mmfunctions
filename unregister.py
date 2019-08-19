#!/usr/bin/python3
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
from iotfunctions import bif
from iotfunctions import ui
from iotfunctions.enginelog import EngineLogging
from mmfunctions import functions

print ("Instantiate")
ais = functions.AggregateItemStats('blah',stats.pearsonr)
print (*ais.output_items, sep = "\n")
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

db.unregister_functions(["AggregateItemStats"])

