import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import subprocess


subprocess.run("curl_dataset.sh", check=True)

url = f"mariadb+pymysql://user:pass@localhost/netfloox"
engine = create_engine(url, echo=True)
