import argparse
import os
import pickle
from dotenv import dotenv_values
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import boto3
import pandas as pd
import numpy as np

MAX_COL_COUNT = 10
MAX_ROWS = 100

BUCKET = "semantic-annotation-tables"
config = dotenv_values("../.env")
s3 = boto3.client(
    "s3", aws_access_key_id=config["AWS_ACCESS_KEY_ID"], aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)
os.environ["OPENAI_API_KEY"] = config["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://semanticannotation-aiservices.openai.azure.com"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

# New prompt
semantic_concept_template = """

Answer the question based on the task below. If the question cannot be answered using the information provided answer with "I don't know".

Task: Suggest a semantic concept for each column of a given table. Answer with the semantic concept for each column with the format Column1: semantic concept.

Table: {input}

Semantic concepts:

"""

gpt_4 = AzureChatOpenAI(deployment_name="gpt-4", temperature=0)
prompt = PromptTemplate(template=semantic_concept_template, input_variables=["input"])
llm_chain_4 = LLMChain(prompt=prompt, llm=gpt_4)


def convert_to_column_major(example: str) -> str:
    lines = example.split("\n")
    col_major = [col.strip() + ": " for col in lines[0].split("||")]
    for line in lines[1:]:
        for i, val in enumerate(line.split("||")):
            col_major[i] += val + ", "
    debug_eg = "\n".join(col_major)  # Not needed to remove last row like when handling sotab
    return debug_eg


def evaluate(df: pd.DataFrame) -> list[str]:
    columns_count = len(df.columns)
    new_column_header = " || ".join([f"Column {i}" for i in range(columns_count)]) + "\n"

    rows = []
    for _, row in df.iterrows():
        row = [str(item) for item in row]
        rows.append(" || ".join(row))
    table_preds = llm_chain_4.run({"input": new_column_header + "\n".join(rows[:10])})
    if "Semantic concepts:" in table_preds:
        table_preds = table_preds.split("Class:")[1]

    # Break predictions into either \n or ,
    if ":" in table_preds:
        separator = ":"
    elif "-" in table_preds:
        separator = "-"
    else:
        separator = ","

    col_preds = table_preds.split(separator)[1:]
    predictions = []
    for col_idx in range(columns_count):
        # print(idx, gt)
        if int(col_idx) >= len(col_preds):
            predictions.append("-")
        else:
            pred = col_preds[int(col_idx)]
            # Remove break lines
            if "\n" in pred:
                pred = pred.split("\n")[0].strip()
            # Remove commas
            if "," in pred:
                pred = pred.split(",")[0].strip()
            # Remove paranthesis
            if "(" in pred:
                pred = pred.split("(")[0].strip()
            # Remove points
            if "." in pred:
                pred = pred.split(".")[0].strip()
            # Lower-case prediction
            pred = pred.strip().lower()
            predictions.append(pred)
    return predictions


def get_data_from_s3(example_name: str, file_format: str):
    print(example_name)
    obj = s3.get_object(Bucket=BUCKET, Key=example_name)
    df = None
    if file_format == "csv":
        df = pd.read_csv(obj["Body"])
    elif file_format == "csv_3rows":
        df = pd.read_csv(obj["Body"], header=[0, 1, 2])
    elif file_format == "tsv":
        df = pd.read_csv(obj["Body"], sep="\t")
    return df.iloc[:MAX_ROWS, :MAX_COL_COUNT]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions.")
    parser.add_argument(
        "--current-dataset-filename",
        type=str,
        default="current_dataset.csv",
    )
    args = parser.parse_args()
    current_dataset_df = pd.read_csv(args.current_dataset_filename)
    predictions = []
    for table_id in range(len(current_dataset_df)):
        example = current_dataset_df["name"].iloc[table_id]
        file_format = current_dataset_df["file_format"].iloc[table_id]
        df = get_data_from_s3(example, file_format)
        print(df.head())
        pred = evaluate(df)
        predictions.append(str(pred))

    df = pd.DataFrame(predictions, columns=["output"])
    df.to_csv("tab_ann_gpt_outputs.csv")
    print(df)
