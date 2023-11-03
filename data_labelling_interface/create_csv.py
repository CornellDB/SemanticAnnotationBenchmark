import pickle

import pandas as pd


def load_data():
    with open("cta-test-table-wise.pkl", "rb") as f:
        data = pickle.load(f)
    return data


table_names = []
data = load_data()
for example in data:
    table_name = example[0]
    table_names.append(f"sotab/{table_name}.csv")
    # lines = example[1].split("\n")
    # table = [line.split("||") for line in lines[1:-1]]
    # column_name = lines[0].strip().split("||")
    # if column_name[-1] == "":
    #     column_name = column_name[:-1]
    #     table = [row[:-1] for row in table]
    # df = pd.DataFrame(table, columns=column_name)
    # df.to_csv(f"{table_name}.csv")
df = pd.DataFrame({"name": table_names})
df["file_format"] = "csv"
df.to_csv("current_dataset.csv")
