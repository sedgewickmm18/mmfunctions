#!/usr/bin/python3
import sys
import logging
import json
from iotfunctions.db import Database
from mmfunctions import anomaly
from iotfunctions.enginelog import EngineLogging

PACKAGE_URL = "https://github.com/sedgewickmm18/mmfunctions"

credentials = {
  "_id": "26590cfdeb9829d3875a5dc098fb009a",
  "tenantId": "beta-3",
  "db2": {
    "username": "bluadmin",
    "password": "NjYxYzIwODRkMjAx",
    "databaseName": "BLUDB",
    "port": 50000,
    "httpsUrl": "https://dashdb-enterprise-yp-dal12-134.services.dal.bluemix.net:8443",
    "host": "dashdb-enterprise-yp-dal12-134.services.dal.bluemix.net"
  },
  "iotp": {
    "url": "https://ieraqj.internetofthings.ibmcloud.com/api/v0002",
    "orgId": "ieraqj",
    "host": "ieraqj.messaging.internetofthings.ibmcloud.com",
    "port": 8883,
    "asHost": "api-beta.connectedproducts.internetofthings.ibmcloud.com",
    "apiKey": "a-ieraqj-f2ijqgada3",
    "apiToken": "NVGp62xEHxVoe!z!uU"
  },
  "objectStorage": {
    "region": "global",
    "username": "49d43c9eaa704fe7ba160cec0db54657",
    "password": "2a1120e8710ada2507b61e885351a2248f81b90783d2c9dc"
  },
  "config": {
    "objectStorageEndpoint": "https://undefined",
    "bos_logs_bucket": "analytics-logs-beta-3-337dfd48872c",
    "bos_runtime_bucket": "analytics-runtime-beta-3-fa4e4e54e2e8",
    "mh_topic_analytics_alerts": "analytics-alerts-beta-3"
  }
}


EngineLogging.configure_console_logging(logging.DEBUG)
db_schema = None


if (len(sys.argv) <= 1) or (sys.argv[1] != 'test'):
    db = Database(credentials=credentials)
    print(db.cos_load)


# if in test mode call execute()
ais = anomaly.SpectralAnomalyScore('Val', windowsize=12, output_item='zscore')
kis = anomaly.KMeansAnomalyScore('Val', windowsize=4, output_item='kscore')

print("Instantiated")

# if there is a 2nd argument do not register but exit
if (len(sys.argv) > 1):
    sys.exit()

EngineLogging.configure_console_logging(logging.DEBUG)

# with open('credentials_as_dev.json', encoding='utf-8') as F:
#     credentials = json.loads(F.read())

#db.register_module(functions)

#db.register_functions([anomaly.SpectralAnomalyScore])
#db.register_functions([anomaly.KMeansAnomalyScore])
#db.register_functions([anomaly.NoDataAnomalyScoreNew])
#db.register_functions([anomaly.SimpleAnomaly])
#db.register_functions([anomaly.SimpleRegressor])
#db.register_functions([anomaly.GeneralizedAnomalyScore2])
#db.register_functions([anomaly.FFTbasedGeneralizedAnomalyScore2])
db.register_functions([anomaly.SaliencybasedGeneralizedAnomalyScore])
