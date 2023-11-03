import pickle

import pandas as pd


import os

filenames = []
for x in os.listdir("/mnt/c/Users/tansi/Documents/MEng_Project_2/nyc_open_data/nyc_open_data"):
    if x.endswith(".csv"):
        filenames.append(f"nyc_open_data/{x}")
df = pd.DataFrame({"name": filenames})
df["file_format"] = "csv"
df.to_csv("current_dataset_nyc_open_data.csv")
