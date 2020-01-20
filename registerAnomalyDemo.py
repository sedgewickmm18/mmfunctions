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
  "tenantId" : "Monitor-Demo",
  "db2":{
    "username":"bluadmin",
    "password":"MjZkZTEwN2FjMWY1",
    "databaseName":"BLUDB",
    "port":50000,
    "httpsUrl":"https://dashdb-enterprise-yp-dal12-125.services.dal.bluemix.net:8443",
    "host":"dashdb-enterprise-yp-dal12-125.services.dal.bluemix.net"
  },
 "iotp":{
    "url":"https://vrvzh6.internetofthings.ibmcloud.com/api/v0002",
    "orgId":"vrvzh6",
    "host":"vrvzh6.messaging.internetofthings.ibmcloud.com",
    "port":8883,
    "asHost":"api-beta.connectedproducts.internetofthings.ibmcloud.com",
    "apiKey":"a-vrvzh6-am4fwylysp",
    "apiToken":"F+PF@HQhe_N*ecS4gb"
  },
  "messageHub":{
     "brokers":["kafka03-prod02.messagehub.services.us-south.bluemix.net:9093","kafka01-prod02.messagehub.services.us-south.bluemix.net:9093","kafka05-prod02.messagehub.services.us-south.bluemix.net:9093","kafka02-prod02.messagehub.services.us-south.bluemix.net:9093","kafka04-prod02.messagehub.services.us-south.bluemix.net:9093"],
     "username":"S0Zzzp1zZsF4hotd",
     "password":"S1wyAP2jy9R2cmqcZwSfoqL5gByWSqn9"
   },
   "objectStorage":{
      "region":"global",
      "username":"5c9b5139a83d4f68bd2fe458a2117fac",
      "password":"5b282a68b84d070bd8674da4771998c6cb0743a4a0288129"
    },
    "config":{"objectStorageEndpoint":"https://undefined",
        "bos_logs_bucket":"analytics-logs-monitor-demo-9c6aaaf268a1","bos_runtime_bucket":"analytics-runtime-monitor-demo-395729cb6a06","mh_topic_analytics_alerts":"analytics-alerts-Monitor-Demo"
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
ais = anomaly.SpectralAnomalyScore('Val', windowsize=12, output_item='zscore')
kis = anomaly.KMeansAnomalyScore('Val', windowsize=4, output_item='kscore')

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
#db.register_functions([anomaly.NoDataAnomalyScoreNew])
#db.register_functions([anomaly.SimpleAnomaly])
#db.register_functions([anomaly.SimpleRegressor])
db.register_functions([anomaly.SaliencybasedGeneralizedAnomalyScore])

