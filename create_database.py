import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.sql import text
from sqlalchemy_utils import database_exists, create_database
import subprocess
import gzip


#subprocess.run("curl_dataset.sh", check=True)
#url = f"mariadb+pymysql://admin:pass@localhost/netfloox"
url = URL.create(
    drivername="postgresql",
    username="citus",
    host="c-groupe4.tlvz7y727exthe.postgres.cosmos.azure.com",
    database="netfloox",
    password="floox2024!"
)
engine = create_engine(url, echo=True, connect_args={'sslmode': "require"})

df=pd.read_table("downloads/name.basics.tsv")
df.to_sql(name='name_basics', con=engine, if_exists = 'fail', index=False)