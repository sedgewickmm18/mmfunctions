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

from collections import OrderedDict
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

None5 = [None, None, None, None, None]
None15 = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

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

        list_of_entity = []
        list_of_ts = []
        list_of_vibx = []
        list_of_viby = []
        list_of_vibz = []
        list_of_speed = []
        list_of_power = []
        list_of_devicetype = []
        list_of_log_id = []
        list_of_eventtype = []
        list_of_format = []

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


        if not USING_DB:
            client = None
            if i_am_device:
                client = wiotp.sdk.device.DeviceClient(config=auth_token, logHandlers=None)
            else:
                client = wiotp.sdk.application.ApplicationClient(config=auth_token, logHandlers=None)

            client.on_connect = on_connect  # On Connect Callback.
            client.on_publish = on_publish  # On Publish Callback.
            client.on_disconnect = on_disconnect # On Disconnect Callback.
            client.connect()

        #Now = dt.datetime.now(pytz.timezone("UTC"))
        Now = dt.datetime.now
        print(Now)

        # retrieve last recorded timestamps by entity
        db = self.get_db()
        try:
            date_recorder = db.model_store.retrieve_model('Armstark')
            if date_recorder is None:
                date_recorder = OrderedDict()
        except Exception:
            date_recorder = OrderedDict()
            pass
        logger.info('Date recorder ' + str(date_recorder))

        new_date_recorder = date_recorder.copy()

        # Count rows with old data
        old_data_rows = 0
        once =  100
        once2 = 100
        once3 = 100

        # assume single entity
        for ix, row in df.iterrows():
            # columns with 15 elements
            #device_id = ix[0].replace('Device','Shadow') - device id is identical !
            device_id = ix[0]

            '''
            if once > 0:
                once -= 1
                print('Power ', row['accel_power'], ' Speed ', row['accel_speed'])
            '''

            # ignore row if time is smaller than last recorded time
            #last_date = dt.datetime.strptime('2021-01-12 19:19:30', '%Y-%m-%d %H:%M:%S') #Now
            last_date = None
            try:
                last_date = date_recorder[device_id]
            except Exception:
                last_date = None
                pass

            '''
            if last_date is not None:
                logger.debug('last date for ' + str(device_id) + ' is ' + str(last_date) +
                             ' compare with ' + str(ix[1]) + 'comparison results: ' + str(ix[1] < last_date) +
                             '/' + str(ix[1] > last_date))
            '''

            if last_date is None:
                logger.debug('Okay')
            elif ix[1] <= last_date:
                #logger.debug('Ignore event from date ' + str(ix[1]))
                #date_recorder[device_id] = last_date
                old_data_rows += 1
                continue

            new_date_recorder[device_id] = ix[1]

            try:
                vibx_ = ast.literal_eval(row['rms_x'])
            except Exception as e1:
                vibx_ = None15.copy()
                #print (' eval of ' + str(row['rms_x']) + ' failed with ' + str(e1))
                continue
                pass

            try:
                viby_ = ast.literal_eval(row['rms_y'])
            except Exception as e2:
                viby_ = None15.copy()
                #print (' eval of ' + str(row['rms_y']) + ' failed with ' + str(e2))
                continue
                pass

            try:
                vibz_ = ast.literal_eval(row['rms_z'])
            except Exception as e3:
                vibz_ = None15.copy()
                #print (' eval of ' + str(row['rms_z']) + ' failed with ' + str(e3))
                continue
                pass

            # columns with 5 elements
            try:
                speed_ = ast.literal_eval(row['accel_speed'])
            except Exception as e4:
                speed_ = None5.copy()
                #print (' eval of ' + str(row['accel_speed']) + ' failed with ' + str(e4))
                continue
                pass

            try:
                power_ = ast.literal_eval(row['accel_power'])
            except Exception as e5:
                power_ = None5.copy()
                #print (' eval of ' + str(row['accel_power']) + ' failed with ' + str(e5))
                continue
                pass

            if once3 > 0:
                once3 -= 1
                print('Power ', power_, 'Speed ', speed_)

            vibx = None15.copy()
            viby = None15.copy()
            vibz = None15.copy()
            power = None5.copy()
            speed = None5.copy()
            for i in range(15):
                try:
                    vibx[i] = float(vibx_[i])
                except Exception:
                    pass
                try:
                    viby[i] = float(viby_[i])
                except Exception:
                    pass
                try:
                    vibz[i] = float(vibz_[i])
                except Exception:
                    pass

            for i in range(5):
                try:
                    speed[i] = float(speed_[i])
                except Exception:
                    pass
                try:
                    power[i] = float(power_[i])
                except Exception:
                    pass

            if once2 > 0:
                once2 -= 1
                print('Power vs Speed ', power, speed)

            if power[0] == speed[0]:
                if power[0] is not None:
                    logger.error('BUG')

            for i in range(15):
                print (len(vibx), len(viby), len(vibz), len(speed), len(power))
                try:
                    # device_id, timestamp
                    list_of_entity.append(device_id)
                    list_of_ts.append(ix[1] + pd.Timedelta(seconds=20*i - 300))
                    list_of_vibx.append(vibx[i])
                    list_of_viby.append(viby[i])
                    list_of_vibz.append(vibz[i])
                    list_of_speed.append(speed[i // 3])
                    list_of_power.append(power[i // 3])
                    list_of_devicetype.append('pump_de_gen5')
                    list_of_log_id.append('Shadow_pump_de_gen5')
                    list_of_eventtype.append('ShadowPumpDeGen5')
                    list_of_format.append('json')

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
            print('writing ', len(list_of_ts))
            db = self.get_db()
            table = db.get_table('IOT_SHADOW_PUMP_DE_GEN5')
            cols = [column.key for column in table.columns]
            my_cols = ['evt_timestamp', 'rms_x', 'rms_y', 'rms_z', 'POWER', 'speed', 'devicetype', 'deviceid',
                       'logicalinterface_id', 'eventtype', 'format', 'rcv_timestamp_utc', 'updated_utc']

            print('DataBase is ', db, ' columns are ', cols, ' ,writing to ', my_cols)

            df_new = pd.DataFrame(list(zip(list_of_ts, list_of_vibx, list_of_viby, list_of_vibz,
                                           list_of_power, list_of_speed, list_of_devicetype, list_of_entity,
                                           list_of_log_id, list_of_eventtype, list_of_format, list_of_ts, list_of_ts)),
                                  columns=my_cols)

            df_new = df_new.set_index(['deviceid', 'evt_timestamp'])
            df_new = df_new[~df_new.index.duplicated(keep='first')]

            #['evt_timestamp', 'rms_x', 'rms_z', 'rms_y', 'POWER', 'spee
            #d', 'devicetype', 'deviceid', 'logicalinterface_id', 'eventtype', 'format', 'rcv_timestamp_utc', 'updated_utc']

            db.write_frame(df_new, 'IOT_SHADOW_PUMP_DE_GEN5')
            print('DONE')

        # write back last recorded date
        try:
            logger.info('Ignored ' + str(old_data_rows) + ' old events')
            db.model_store.store_model('Armstark', new_date_recorder)
        except Exception:
            pass


        msg = 'UnrollData'
        self.trace_append(msg)

        if not USING_DB:
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
