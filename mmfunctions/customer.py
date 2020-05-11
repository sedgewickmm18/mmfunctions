# *****************************************************************************
# © Copyright IBM Corp. 2018-2020.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

'''
The Built In Functions module contains customer specific helper functions
'''

import datetime as dt
import numpy as np
import pytz

# import re
# import pandas as pd
import logging
import wiotp.sdk

# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer)
from iotfunctions.ui import (UISingleItem, UIMultiItem)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False


# On connect MQTT Callback.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code : " + str(rc))


# on publish MQTT Callback.
def on_publish(client, userdata, mid):
    print("Message Published.")


class UnrollData(BaseTransformer):
    '''
    Compute L2Norm of string encoded array
    '''
    def __init__(self, vibrationx, vibrationy, vibrationz, speed, power):
        super().__init__()

        self.vibrationx = vibrationx
        self.vibrationy = vibrationy
        self.vibrationz = vibrationz
        self.speed = speed
        self.power = power

        # HARDCODED SINGLE ENTITY + Output device type
        self.config = {"identity": {"orgId": "vrvzh6", "typeId": "MMOutputDevice", "deviceId": "MMDeviceID"},
                       "auth": {"token": "mmdevice"}}

    def execute(self, df):

        # ONE ENTITY FOR NOW
        # connect
        client = wiotp.sdk.device.DeviceClient(config=self.config, logHandlers=None)

        client.on_connect = on_connect  # On Connect Callback.
        client.on_publish = on_publish  # On Publish Callback.
        client.connect()

        Now = dt.datetime.now(pytz.timezone("UTC"))

        # assume single entity
        for ix, row in df.iterrows():
            # columns with 15 elements
            vibx = row[self.vibrationx].to_numpy()
            viby = row[self.vibrationy].to_numpy()
            vibz = row[self.vibrationz].to_numpy()

            # columns with 5 elements
            speed = row[self.speed].to_numpy()
            power = row[self.power].to_numpy()

            for i in range(15):
                js = {'vibx': vibx[i], 'viby': viby[i], 'vibz': vibz[i],
                      'speed': speed[i // 3], 'power': power[i // 3]}

                client.publishEvent(eventId="MMEventNew", msgFormat="json", data=str(js),
                                    qos=0, onPublish=None, Date=Now + dt.datetime.timedelta(0, 20 * i))

        msg = 'UnrollData'
        self.trace_append(msg)

        return (df)

    @classmethod
    def build_ui(cls):

        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(
                name='group1_in',
                datatype=None,
                description='String encoded array of sensor readings, 15 readings per 5 mins',
                output_item='group1_out',
                is_output_datatype_derived=True, output_datatype=None
                ))
        inputs.append(UIMultiItem(
                name='group2_in',
                datatype=None,
                description='String encoded array of sensor readings, 5 readings per 5 mins',
                output_item='group2_out',
                is_output_datatype_derived=True, output_datatype=None
                ))

        # define arguments that behave as function outputs
        outputs = []
        return (inputs, outputs)
