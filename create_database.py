import imports
import numpy as np
import sqlite3 as sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

directory_path = "downloads"
all_items = os.listdir(directory_path)
files = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]

file_names=[f for f in files]

conn = sqlite3.connect("output.sqlite")
cursor = conn.cursor()

for file_name in file_names:
    cursor.execute(f"CREATE TABLE {os.path.basename(file_name)[:-4]} (column1 TEXT, column2 TEXT, column3 TEXT)")

    with open(file_name, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            cursor.execute(f"""
            INSERT INTO {os.path.basename(file_name)[:-4]} VALUES (?, ?, ?)
            """, row)

conn.commit()
conn.close()