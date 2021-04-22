from sqlalchemy import *
import pandas as pd
from datetime import datetime
from jinjasql import JinjaSql
import pysftp
import re

def get_datalake_conn(user_name, password):
    engine = create_engine('postgresql://' + user_name + ':' + password + '@10.50.8.212:60906/mimic') \
        .execution_options(autocommit=True)
    conn = engine.connect()
    return conn

def db_send_update_from_file(conn, sql_file, params = {}) :
    j = JinjaSql()
    f = open(sql_file, 'r',encoding='UTF-8')
    # f = open(sql_file, 'r')
    template = f.read().replace('}}', ' | sqlsafe }}')
    f.close()
    query = j.prepare_query(template, params)[0]
    return conn.execute(query)

def db_get_query_from_file(conn, sql_file, params = {}) :
    j = JinjaSql()
    f = open(sql_file, 'r')
    template = f.read().replace('}}', ' | sqlsafe }}')
    f.close()
    query = j.prepare_query(template, params)[0]
    return pd.read_sql_query(con = conn, sql = query)

