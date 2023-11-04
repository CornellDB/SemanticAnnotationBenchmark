import json
from datetime import datetime

import boto3
import pandas as pd
import pinecone
import streamlit as st
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text
import logging


from semantic_search import query

logging.basicConfig(
    filename="semantic_annotation.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("app.py")
st.set_page_config(page_title="Semantic Annotation Benchmark", layout="wide")

CUSTOM_OPTION = "Another option..."
BUCKET = "semantic-annotation-tables"
MAX_ROWS = 100
MAX_COLS = 10


def get_data_from_s3(example_name: str, file_format: str):
    logger.info(example_name)
    obj = s3.get_object(Bucket=BUCKET, Key=example_name)
    df = None
    if file_format == "csv":
        df = pd.read_csv(obj["Body"], index_col=0)
    elif file_format == "csv_3rows":
        df = pd.read_csv(obj["Body"], index_col=0, header=[0, 1, 2])
    elif file_format == "tsv":
        df = pd.read_csv(obj["Body"], sep="\t")
    return df.iloc[:MAX_ROWS, :MAX_COLS]


def get_initial_table():
    logger.info("Get initial table")
    st.session_state.table_id = conn.query(
        f"SELECT current_table_id FROM annotators WHERE name = '{st.session_state.selected_annotator}';", ttl=0
    ).iloc[0]["current_table_id"]
    logger.info(f"Table id {st.session_state.table_id}")
    example = tables["name"].iloc[st.session_state.table_id]
    file_format = tables["file_format"].iloc[st.session_state.table_id]
    logger.info(f"{example}, {file_format}")
    df = get_data_from_s3(example, file_format)
    st.session_state.df = df.rename(
        columns={column: f"{column} (Column {idx})" for idx, column in enumerate(df.columns)}
    )


def get_next_table():
    logger.info("Get next table")
    while True:
        st.session_state.table_id += 1
        example = tables["name"].iloc[st.session_state.table_id]
        label_count = conn.query(f"SELECT count(table_name) FROM labels WHERE table_name = '{example}';", ttl=0).iloc[
            0
        ]["count(table_name)"]
        logger.info(f"Label count {label_count}, table {st.session_state.table_id}")
        if label_count < 3:
            break
    with conn.session as s:
        s.execute(
            text(
                f"UPDATE annotators SET current_table_id = {st.session_state.table_id} WHERE name = '{st.session_state.selected_annotator}';"
            )
        )
        s.commit()
    logger.info("Updated")
    example = tables["name"].iloc[st.session_state.table_id]
    file_format = tables["file_format"].iloc[st.session_state.table_id]
    df = get_data_from_s3(example, file_format)
    st.session_state.df = df.rename(
        columns={column: f"{column} (Column {idx})" for idx, column in enumerate(df.columns)}
    )


def search_glossary(text_search: str) -> list[str]:
    if text_search:
        results = query(model, index, text_search)
        if results:
            return [match["id"] for match in results["matches"]]
    return []


def validate_submission() -> bool:
    logger.info("Validating submission")
    # Checking if all the fields are non empty
    for i in range(len(st.session_state.df.columns)):
        if st.session_state[f"col{i}"] is None and st.session_state[f"unable_col{i}"] == False:
            return False
    return True


def upload_submission():
    if not validate_submission():
        st.session_state["submission_success"] = False
        return
    logger.info("Append label")
    column_labels = []
    custom_labeled_cols = []
    suggested_concepts = {}
    for i in range(len(st.session_state.df.columns)):
        if st.session_state[f"unable_col{i}"]:
            column_labels.append("Unable to label")
        elif st.session_state[f"col{i}"] == CUSTOM_OPTION:
            column_labels.append(st.session_state[f"custom_col{i}"])
            custom_labeled_cols.append(i)
        else:
            column_labels.append(st.session_state[f"col{i}"])
        suggested_concepts[st.session_state[f"concept_col{i}"]] = (
            st.session_state[f"concept_search_col{i}"] if f"concept_search_col{i}" in st.session_state else []
        )
        st.session_state[f"col{i}"] = ""
        st.session_state[f"concept_col{i}"] = ""
        st.session_state[f"custom_col{i}"] = ""
        st.session_state[f"unable_col{i}"] = False
    new_row = {
        "table_name": tables["name"].iloc[st.session_state.table_id],
        "date": datetime.utcnow(),
        "annotator": st.session_state.selected_annotator,
        "labels": str(column_labels),
        "custom_labeled_cols": str(custom_labeled_cols),
        "suggested_terms": json.dumps(suggested_concepts),
    }
    with conn.session as s:
        s.execute(
            text(
                "INSERT INTO labels (table_name, date, annotator, labels, custom_labeled_cols, suggested_terms) VALUES (:table_name, :date, :annotator, :labels, :custom_labeled_cols, :suggested_terms);"
            ),
            params=new_row,
        )
        s.commit()
    get_next_table()
    st.session_state["submission_success"] = True


# Load env file with API KEY using full path
config = dotenv_values("../.env")
s3 = boto3.client(
    "s3", aws_access_key_id=config["AWS_ACCESS_KEY_ID"], aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)
# Create the SQL connection to pets_db as specified in your secrets file.
conn = st.connection("semantic_annotation_backend_db", type="sql")


@st.cache_resource
def init_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    return model


@st.cache_resource
def init_index():
    pinecone.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    index_name = "semantic-annotation"
    return pinecone.Index(index_name)


model = init_model()
index = init_index()

tables = pd.read_csv("current_dataset_subset.csv")
st.title("Semantic Annotation Benchmark Creator")

# Insert some data with conn.session.
with conn.session as s:
    s.execute(text("CREATE TABLE IF NOT EXISTS annotators (name TEXT, current_table_id INTEGER);"))
    s.execute(
        text(
            "CREATE TABLE IF NOT EXISTS labels (table_name TEXT, date TEXT, annotator TEXT, labels TEXT, custom_labeled_cols TEXT, suggested_terms TEXT);"
        )
    )
    row_length = conn.query("SELECT count(name) FROM annotators;", ttl=0)
    if row_length.loc[0]["count(name)"] == 0:
        s.execute(
            text("DELETE FROM annotators;"),
        )
        annotators_init = {"Lionel": 0, "Udayan": 0, "Sainyam Galhotra": 0, "Participant 1": 0}
        for k, id in annotators_init.items():
            s.execute(
                text("INSERT INTO annotators (name, current_table_id) VALUES (:name, :current_table_id);"),
                params=dict(name=k, current_table_id=id),
            )
    s.commit()
annotators_names = conn.query("SELECT name FROM annotators", ttl=0)["name"].to_list()
selected_annotator = st.selectbox("Annotator", annotators_names, key="selected_annotator", on_change=get_initial_table)

if "df" not in st.session_state:
    get_initial_table()


left_column, right_column = st.columns([3, 2], gap="medium")


with left_column:
    st.subheader("Current table to label")
    st.write(f"Table Id: {st.session_state.table_id}")
    st.dataframe(st.session_state.df, use_container_width=True)
    next_table = st.button("Get next table", on_click=get_next_table)

with right_column:
    st.subheader("Annotate labels")
    for i in range(len(st.session_state.df.columns)):
        with st.expander(f"Column {i}"):
            suggestion = st.text_input(f"Search for concepts", key=f"concept_col{i}")
            options = search_glossary(suggestion)
            if options:
                st.session_state[f"concept_search_col{i}"] = options
                options.append(CUSTOM_OPTION)
            selection = st.selectbox(f"Select suggested concepts", options, key=f"col{i}")
            if selection == CUSTOM_OPTION:
                otherOption = st.text_input("Enter your other option...", key=f"custom_col{i}")
            unable_to_label = st.checkbox("Unable to label", key=f"unable_col{i}")
    submit_form = st.button("Submit", on_click=upload_submission)
    if submit_form:
        if st.session_state["submission_success"]:
            st.success("Submitted")
        else:
            st.warning(
                "Please label all the columns with a selected concept or suggested concept or indicate that you were unable to label."
            )

# df = conn.query("SELECT * from annotators;", ttl=0)
# st.write(df)

# df2 = conn.query("SELECT * from labels;", ttl=0)
# st.write(df2)
