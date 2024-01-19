#!/usr/bin/python3
import datetime as dt
import sys
import logging
import json
from iotfunctions.db import Database
from mmfunctions import bif,anomaly
from iotfunctions.enginelog import EngineLogging

PACKAGE_URL = "https://github.com/sedgewickmm18/mmfunctions"

# connect to the json document with the credentials
with open('credentials_as_monitor31.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())

# as much logging as possible
EngineLogging.configure_console_logging(logging.DEBUG)

# no specific schema
db_schema = None

# Connect to Monitor's REST API
db = Database(credentials=credentials)

# Example to register a constant
expression_constant = {'expression': 'df["accelx"].values.astype(np.float) > 0.0'}

class ExprConst:
    def __init__(self, name, json):
        self.name = name
        self.datatype = 'STRING'
        self.description = 'ExprConst'
        self.value = json

    def to_metadata(self):
        meta = {'name': self.name, 'dataType': self.datatype, 'description': self.description,
                'value': self.value}
        return meta

# but we don't do it
#db.register_constants([ExprConst('my_alert_expression', expression_constant)])
# to get rid of a registered constant
#db.unregister_constants('my_alert_expression')

db.register_functions([anomaly.KDEAnomalyScore1d])
