# encoding: utf-8
"""
Created on September 22, 2016
@author: thom.hopmans
"""
import time
from datetime import datetime


def parse_datetime_as_unixtstamp(datetime_obj):
    unix_tstamp = str(int(time.mktime(datetime_obj.timetuple())))
    return unix_tstamp


def parse_unixtstamp_as_datetime(unix_tstamp):
    datetime_obj = datetime.fromtimestamp(int(unix_tstamp))  # .strftime('%Y-%m-%d %H:%M:%S')
    return datetime_obj

