#!/usr/bin/python3
import datetime as dt
import json
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
import iotfunctions.bif as bif
from iotfunctions.metadata import EntityType, LocalEntityType
from iotfunctions.db import Database
from iotfunctions.dbtables import FileModelStore

with open('credentials_as_monitor_demo.json', encoding='utf-8') as F:
     credentials = json.loads(F.read())
db_schema = None
model_store = FileModelStore()
db = Database(credentials=credentials, model_store=model_store)

sample_entity_type = bif.PythonFunction(
    function_code = 'adjusted_distance',
    input_items = ['distance'],
    output_item = 'adjusted_distance',
    parameters = {}
        )

df = sample_entity_type.execute_local_test(db=db, generate_days=1, to_csv=True)

print(df)
