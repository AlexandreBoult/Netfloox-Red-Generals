import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy_utils import database_exists, create_database
import subprocess
import gzip


#subprocess.run("curl_dataset.sh", check=True)
url = f"mariadb+pymysql://admin:pass@localhost/netfloox"
engine = create_engine(url, echo=True)

df=pd.read_table("downloads/name.basics.tsv")
df.to_sql(name='name_basics', con=engine, if_exists = 'fail', index=False)