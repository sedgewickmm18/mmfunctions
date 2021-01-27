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

import json
import datetime as dt
import pytz
import base64
import ast
import math

# import re
import pandas as pd
import logging
import wiotp.sdk

# import warnings
# import json
# from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.base import (BaseTransformer)
from iotfunctions.ui import (UIMultiItem, UISingle)

logger = logging.getLogger(__name__)
PACKAGE_URL = 'git+https://github.com/sedgewickmm18/mmfunctions.git'
_IS_PREINSTALLED = False

USING_DB = True


# On connect MQTT Callback.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code : " + str(rc))


# on publish MQTT Callback.
def on_publish(client, userdata, mid):
    print("Message Published.")


# on publish MQTT Callback.
def on_disconnect(client, userdata, rc):
    print("Disconnected with result code : " + str(rc))


class UnrollData(BaseTransformer):
    '''
    Unroll string encoded array and send it back to shadow device
    '''
    def __init__(self, group1_in, group2_in, group1_out, group2_out):
        super().__init__()

        self.group1_in = group1_in
        self.group2_in = group2_in
        self.group1_out = group1_out
        self.group2_out = group2_out

        # HARDCODED SINGLE ENTITY + Output device type
        #self.config = {"identity": {"orgId": "vrvzh6", "typeId": "MMDeviceTypeShadow", "deviceId": "MMShadow1"},
        #               "auth": {"token": "mmshadow1"}}

    def execute(self, df):

        df_new = pd.DataFrame(columns=['evt_timestamp', 'deviceid', 'rms_x', 'rms_y', 'rms_z', 'power', 'speed',
                                       'logicalinterface_id', 'eventtype', 'format', 'rcv_timestamp_utc', 'updated_utc'])

        #
        c = self._entity_type.get_attributes_dict()
        try:
            auth_token = c['auth_token']
            print('Auth Token ' , str(auth_token))
        except Exception as ae:
            print('Auth Token missing ' , str(ae))

        i_am_device = False
        try:
            if 'auth' in auth_token:
                if 'identity' in auth_token.auth:
                    if 'orgId' in auth_token.auth.identity:
                        i_am_device = True
        except Exception:
            pass

        try:
            if 'pem' in auth_token:
                base64_message = base64.b64decode(auth_token['pem'])
                f = open("cafile.pem", "wb")
                f.write(base64_message)
                f.close()
        except Exception:
            pass

        try:
            del auth_token['options']
        except Exception:
            pass

        # ONE ENTITY FOR NOW
        # connect
        print('Unroll Data execute')
        #client = wiotp.sdk.device.DeviceClient(config=self.config, logHandlers=None)
        client = None
        if i_am_device:
            client = wiotp.sdk.device.DeviceClient(config=auth_token, logHandlers=None)
        else:
            client = wiotp.sdk.application.ApplicationClient(config=auth_token, logHandlers=None)

        client.on_connect = on_connect  # On Connect Callback.
        client.on_publish = on_publish  # On Publish Callback.
        client.on_disconnect = on_disconnect # On Disconnect Callback.
        client.connect()

        Now = dt.datetime.now(pytz.timezone("UTC"))
        print(Now)

        # assume single entity
        for ix, row in df.iterrows():
            # columns with 15 elements
            #device_id = ix[0].replace('Device','Shadow') - device id is identical !
            device_id = ix[0]

            None5 = [None, None, None, None, None]
            None15 = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

            try:
                vibx = ast.literal_eval(row['rms_x'])
            except Exception as e1:
                vibx = None15
                #print (' eval of ' + str(row['rms_x']) + ' failed with ' + str(e1))
                continue
                pass

            try:
                viby = ast.literal_eval(row['rms_y'])
            except Exception as e2:
                viby = None15
                print (' eval of ' + str(row['rms_y']) + ' failed with ' + str(e2))
                continue
                pass

            try:
                vibz = ast.literal_eval(row['rms_z'])
            except Exception as e3:
                vibz = None15
                print (' eval of ' + str(row['rms_z']) + ' failed with ' + str(e3))
                continue
                pass

            # columns with 5 elements
            try:
                speed = ast.literal_eval(row['accel_speed'])
            except Exception as e4:
                speed = None5
                print (' eval of ' + str(row['accel_speed']) + ' failed with ' + str(e4))
                continue
                pass

            try:
                power = ast.literal_eval(row['accel_power'])
            except Exception as e5:
                power = None5
                print (' eval of ' + str(row['accel_power']) + ' failed with ' + str(e5))
                continue
                pass

            for i in range(15):
                try:
                    vibx[i] = float(vibx[i])
                except Exception:
                    pass
                try:
                    viby[i] = float(viby[i])
                except Exception:
                    pass
                try:
                    vibz[i] = float(vibz[i])
                except Exception:
                    pass

            for i in range(5):
                try:
                    speed[i] = float(speed[i])
                except Exception:
                    pass
                try:
                    power[i] = float(power[i])
                except Exception:
                    pass

            list_of_rows = []
            for i in range(15):
                try:
                    # device_id, timestamp
                    list_of_rows.append([device_id, ix[1] + pd.Timedelta(seconds=20*i - 300),
                                         # rms, accel
                                         vibx[i], viby[i], vibz[i], speed[i // 3], power[i // 3],
                                         # logicalinterface_id,  eventtype, format
                                         'Shadow_pump_de_gen5', 'ShadowPumpDeGen5', 'json',
                                         # rcv_timestamp_utc, updated_utc
                                         ix[1] + pd.Timedelta(seconds=20*i - 300), ix[1] + pd.Timedelta(seconds=20*i - 300)])
                except Exception as ee:
                    print('Index - ', i, '   ', str(ee))
                    break

                try:
                    jsin = {'evt_timestamp': (ix[1] + pd.Timedelta(seconds=20*i - 300)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z',
                            #'evt_timestamp': (ix[1] + pd.Timedelta(seconds=20*i - 300)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            # 2020-05-26T10:24:56.098000.
                            'rms_x': vibx[i], 'rms_y': viby[i], 'rms_z': vibz[i],
                            'speed': speed[i // 3], 'power': power[i // 3]}
                except Exception as ee:
                    print('Index', i, '   ', str(ee))
                    break

                jsdump = json.dumps(jsin)
                js = json.loads(jsdump)

                if not USING_DB:
                    print('sending ', js, ' to ', device_id)
                    if i_am_device:
                        client.publishEvent(eventId="MMEventOutputType", msgFormat="json", data=js)
                    else:
                        client.publishEvent(typeId="Shadow_pump_de_gen5", deviceId=device_id, eventId="ShadowPumpDeGen5",
                                            msgFormat="json", data=js, qos=0)
                                                        #client.publishEvent(typeId="MMDeviceTypeShadow", deviceId=device_id, eventId="MMEventOutputType",
                        #                    msgFormat="json", data=js, qos=0)  # , onPublish=eventPublishCallback)
        if USING_DB:
            print('writing ', df_new.columns, ' to ', device_id)
            db = self.get_db()
            print('DataBase is ', db)
            db.write_frame(df_new, 'IOT_SHADOW_PUMP_DE_GEN')
            print('DONE')

        msg = 'UnrollData'
        self.trace_append(msg)

        client.disconnect()

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
