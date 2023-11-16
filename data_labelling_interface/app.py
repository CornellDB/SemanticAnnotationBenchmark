import json
import logging
import sqlite3
from datetime import datetime

import boto3
import pandas as pd
import pinecone
import streamlit as st
from dotenv import dotenv_values
from sentence_transformers import SentenceTransformer

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
DEFAULT_ANNOTATOR = "<Select your name>"
BUCKET = "semantic-annotation-tables"
MAX_ROWS = 100
MAX_COLS = 10


def execute_sql_query(command: str):
    # Create the SQL connection to semantic_annotation_backend_db as specified in your secrets file.
    conn = sqlite3.connect("semantic_annotation_backend.db")
    result = conn.execute(command).fetchall()
    conn.close()
    return result


def execute_sql_command(command: str, params=None):
    conn = sqlite3.connect("semantic_annotation_backend.db")
    if params:
        conn.execute(command, params)
    else:
        conn.execute(command)
    conn.commit()
    conn.close()


def get_data_from_s3(example_name: str, file_format: str):
    logger.info(example_name)
    obj = s3.get_object(Bucket=BUCKET, Key=example_name)
    df = None
    if file_format == "csv":
        df = pd.read_csv(obj["Body"])
    elif file_format == "csv_3rows":
        df = pd.read_csv(obj["Body"], header=[0, 1, 2])
    elif file_format == "tsv":
        df = pd.read_csv(obj["Body"], sep="\t")
    return df.iloc[:MAX_ROWS, :MAX_COLS]


def get_initial_table():
    logger.info("Get initial table")
    if st.session_state.selected_annotator == DEFAULT_ANNOTATOR:
        return
    st.session_state.table_id = execute_sql_query(
        f"SELECT current_table_id FROM annotators WHERE name = '{st.session_state.selected_annotator}';"
    )[0][0]
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
    while st.session_state.table_id < len(tables) - 1:
        st.session_state.table_id += 1
        example = tables["name"].iloc[st.session_state.table_id]
        label_count = execute_sql_query(f"SELECT count(table_name) FROM labels WHERE table_name = '{example}';")[0][0]
        logger.info(f"Label count {label_count}, table {st.session_state.table_id}")
        if label_count < 3:
            break

    if st.session_state.table_id == len(tables):
        st.success("No more tables to label")
    else:
        execute_sql_command(
            f"UPDATE annotators SET current_table_id = ? WHERE name = ?;",
            (st.session_state.table_id, st.session_state.selected_annotator),
        )
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
        if len(st.session_state[f"col{i}"]) == 0 and st.session_state[f"unable_col{i}"] == False:
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
        # st.session_state[f"col{i}"] = []
        st.session_state[f"concept_col{i}"] = ""
        st.session_state[f"custom_col{i}"] = ""
        st.session_state[f"unable_col{i}"] = False
        st.session_state[f"currently_selected_col{i}"] = []
    new_row = {
        "table_name": tables["name"].iloc[st.session_state.table_id],
        "date": datetime.utcnow(),
        "annotator": st.session_state.selected_annotator,
        "labels": str(column_labels),
        "custom_labeled_cols": str(custom_labeled_cols),
        "suggested_terms": json.dumps(suggested_concepts),
    }
    execute_sql_command(
        f"INSERT INTO labels (table_name, date, annotator, labels, custom_labeled_cols, suggested_terms) VALUES (?, ?, ?, ?, ?, ?)",
        (
            new_row["table_name"],
            new_row["date"],
            new_row["annotator"],
            new_row["labels"],
            new_row["custom_labeled_cols"],
            new_row["suggested_terms"],
        ),
    )
    get_next_table()
    st.session_state["submission_success"] = True


# Load env file with API KEY using full path
config = dotenv_values("../.env")
s3 = boto3.client(
    "s3", aws_access_key_id=config["AWS_ACCESS_KEY_ID"], aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)


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

tables = pd.read_csv("current_dataset.csv")
st.title("Semantic Annotation Benchmark Creator")

execute_sql_command("CREATE TABLE IF NOT EXISTS annotators (name TEXT, current_table_id INTEGER);")
execute_sql_command(
    "CREATE TABLE IF NOT EXISTS labels (table_name TEXT, date TEXT, annotator TEXT, labels TEXT, custom_labeled_cols TEXT, suggested_terms TEXT);"
)
row_length = execute_sql_query("SELECT count(name) FROM annotators;")[0][0]
if row_length == 0:
    execute_sql_command("DELETE FROM annotators;")
    dummy_participants = {f"Participant {i}": 0 for i in range(10)}
    annotators_init = {"Lionel": 0, "Udayan": 0, "Sainyam Galhotra": 0, **dummy_participants}
    for k, id in annotators_init.items():
        execute_sql_command(f"INSERT INTO annotators (name, current_table_id) VALUES (?, ?);", (k, id))
annotators_names = [row[0] for row in execute_sql_query("SELECT name FROM annotators")]
selected_annotator = st.selectbox(
    "Annotator", [DEFAULT_ANNOTATOR] + annotators_names, key="selected_annotator", on_change=get_initial_table
)

if selected_annotator != DEFAULT_ANNOTATOR:
    if "df" not in st.session_state:
        get_initial_table()

    left_column, right_column = st.columns([3, 2], gap="large")

    with left_column:
        st.subheader("Current table to label")
        st.write(f"Table Id: {st.session_state.table_id}")
        st.write(f"Table File Name: {tables['name'].iloc[st.session_state.table_id]}")
        st.dataframe(st.session_state.df, use_container_width=True)
        next_table = st.button("Get next table", on_click=get_next_table)

    with right_column:
        st.subheader("Annotate labels")
        for i in range(len(st.session_state.df.columns)):
            with st.expander(f"Column {i}"):
                suggestion = st.text_input(f"Search for concepts", key=f"concept_col{i}")
                options = search_glossary(suggestion)
                if options:
                    if suggestion not in options:
                        options = [suggestion] + options
                    st.session_state[f"concept_search_col{i}"] = options
                    # options.append(CUSTOM_OPTION)
                # selection = st.selectbox(f"Select suggested concepts", options, key=f"col{i}")
                # Display the multiselect widget with dynamically filtered options
                if f"currently_selected_col{i}" not in st.session_state:
                    st.session_state[f"currently_selected_col{i}"] = []
                selected_options = st.multiselect(
                    "Select suggested concepts:",
                    options=st.session_state[f"currently_selected_col{i}"] + options,
                    default=st.session_state[f"currently_selected_col{i}"],
                    key=f"col{i}",
                )
                st.session_state[f"currently_selected_col{i}"] = selected_options if len(selected_options) > 0 else []
                other_option = st.text_input("Specify your own custom concept...", key=f"custom_col{i}")
                unable_to_label = st.checkbox("Unable to label", key=f"unable_col{i}")
        submit_form = st.button("Submit", on_click=upload_submission)
        if submit_form:
            if st.session_state["submission_success"]:
                st.success("Submitted")
            else:
                st.warning(
                    "Please label all the columns with a selected concept or suggested concept or indicate that you were unable to label."
                )

css = """
<style>
    section.main>div {
        padding-bottom: 1rem;
    }
    [data-testid="column"] {
        overflow-y: auto;
        overflow-x: hidden;
        max-height: 55vh;
    }
    [data-testid="column"]>div>div {
        padding-right: 10px;
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)
