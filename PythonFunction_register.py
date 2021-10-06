#!/usr/bin/python3
#Import packages and libraries

import datetime as dt
import json
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
import iotfunctions.bif as bif
from iotfunctions.metadata import EntityType, LocalEntityType
from iotfunctions.db import Database
from iotfunctions.dbtables import FileModelStore


#Connect to the service
with open('credentials_as_monitor_demo.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())
db_schema = None
db = Database(credentials=credentials)

#Write the function

def f(df, parameters = None):
    adjusted_distance = df['distance'] * 0.9
    return adjusted_distance

#Save the function to a local model store
model_store = FileModelStore()
model_store.store_model('adjusted_distance', f)

