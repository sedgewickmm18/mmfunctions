#!/usr/bin/python3
import datetime as dt
import sys
import logging
import json
from iotfunctions.db import Database
from mmfunctions import anomaly
from iotfunctions.enginelog import EngineLogging

PACKAGE_URL = "https://github.com/sedgewickmm18/mmfunctions"

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

db_schema = None
if (len(sys.argv) <= 1) or (sys.argv[1] != 'test'):
    db = Database(credentials=credentials)
    print (db.cos_load)


# if in test mode call execute()
ais = anomaly.SpectralAnomalyScore('Val', windowsize=12, output_item='zscore')
kis = anomaly.KMeansAnomalyScore('Val', windowsize=4, output_item='kscore')
gis = anomaly.NoDataAnomalyScore('Val', windowsize=12, output_item='kscore')

print ("Instantiated")

# if there is a 2nd argument do not register but exit
if (len(sys.argv) > 1):
    sys.exit()

EngineLogging.configure_console_logging(logging.DEBUG)

#with open('credentials_as_dev.json', encoding='utf-8') as F:
#    credentials = json.loads(F.read())

#db.register_module(functions)

#db.register_functions([anomaly.SpectralAnomalyScore])
#db.register_functions([anomaly.KMeansAnomalyScore])
#db.register_functions([anomaly.NoDataAnomalyScore])
#db.register_functions([anomaly.SimpleAnomaly])
#db.register_functions([anomaly.GeneralizedAnomalyScore2])
#db.register_functions([anomaly.FFTbasedGeneralizedAnomalyScore2])
db.register_functions([anomaly.SaliencybasedGeneralizedAnomalyScore])

