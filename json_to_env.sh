#!/usr/bin/python3

import datetime as dt
import sys
import logging
import json

with open('credentials_as_monitor_demo.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())

print("#!/bin/sh")
print("export PYTHONUNBUFFERED=1")
print("export DB_SCHEMA=" + credentials['db2']['username'])
print("export MH_USER=" + credentials['messageHub']['username'])
print("export MH_PASSWORD=" + credentials['messageHub']['password'])
brokers = ':'.join(credentials['messageHub']['brokers'])
print("export MH_BROKERS_SASL=" + brokers)
print("export MH_DEFAULT_ALERT_TOPIC=" + credentials['config']['mh_topic_analytics_alerts'])
print("export API_KEY=" + credentials['iotp']['apiKey'])
print("export API_TOKEN=" + credentials['iotp']['apiToken'])
print("export API_BASEURL=https://" + credentials['iotp']['asHost'])
print("export DB_TYPE=DB2")

native_connection_string = 'DATABASE=%s;HOSTNAME=%s;PORT=%s;PROTOCOL=TCPIP;UID=%s;PWD=%s;SECURITY=ssl;' % (
                    credentials['db2']['databaseName'], credentials['db2']['host'],
                    credentials['db2']['port'], credentials['db2']['username'],
                    credentials['db2']['password'],)

print ("export DB_CONNECTION_STRING=\"" + native_connection_string + "\"")
if len(sys.argv) > 0:
        print ("python3 main.py --tenant-id " + credentials['tenantId'] + " --entity-type-id " + sys.argv[1])
else:
        print ("python3 main.py --tenant-id " + credentials['tenantId'] + " --entity-type-id ")
