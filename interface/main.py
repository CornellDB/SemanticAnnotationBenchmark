import pickle

from dotenv import dotenv_values
import streamlit as st
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
from semantic_search import query
from csv import writer
from datetime import datetime


CUSTOM_OPTION = "Another option..."


def load_data():
    with open("../data/cta-test-table-wise.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def get_next_table():
    st.session_state.table_id += 1
    example = examples[st.session_state.table_id]
    lines = example.split("\n")
    table = [line.split("||") for line in lines[1:]]
    column_name = lines[0].strip().split("||")
    if column_name[-1] == "":
        column_name = column_name[:-1]
        table = [row[:-1] for row in table]
    st.session_state.df = pd.DataFrame(table, columns=column_name)


def append_labels():
    column_labels = []
    for i in range(len(st.session_state.df.columns)):
        if st.session_state[f"col{i}"] == CUSTOM_OPTION:
            column_labels.append(st.session_state[f"custom_col{i}"])
        else:
            column_labels.append(st.session_state[f"col{i}"])
    new_row = [data[st.session_state.table_id][0], datetime.utcnow(), selected_annotator, column_labels]
    with open("labels.csv", "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_row)


# Load env file with API KEY using full path
config = dotenv_values("../.env")


@st.cache_resource
def init_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    return model


@st.cache_resource
def init_index():
    pinecone.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    index_name = "semantic-annotation"
    return pinecone.Index(index_name)


st.set_page_config(page_title="Semantic Annotation Benchmark", layout="wide")

model = init_model()
index = init_index()

data = load_data()
examples = [example[1] for example in data]
st.title("Semantic Annotation Benchmark Creator")
annotators = ["Lionel", "Udayan", "Sainyam Galhotra"]
selected_annotator = st.selectbox("Annotator", annotators)


def search_glossary(text_search: str) -> list[str]:
    if text_search:
        results = query(model, index, text_search)
        if results:
            return [match["metadata"]["original"] for match in results["matches"]]
        return results
    return []


left_column, right_column = st.columns([3, 2], gap="medium")

if "df" not in st.session_state:
    st.session_state.table_id = -1
    get_next_table()

with left_column:
    st.subheader("Current table to label")
    st.dataframe(st.session_state.df, use_container_width=True)
    next_table = st.button("Get next table", on_click=get_next_table)

with right_column:
    st.subheader("Annotate labels")
    for i in range(len(st.session_state.df.columns)):
        with st.expander(f"Column {i}"):
            suggestion = st.text_input(f"Search for concepts", key=f"concept_col{i}")
            options = search_glossary(suggestion)
            if options:
                options.append(CUSTOM_OPTION)
            selection = st.selectbox(f"Select suggested concepts", options, key=f"col{i}")
            if selection == CUSTOM_OPTION:
                otherOption = st.text_input("Enter your other option...", key=f"custom_col{i}")
    submit_form = st.button("Submit")

    # Checking if all the fields are non empty
    if submit_form:
        for i in range(len(st.session_state.df.columns)):
            if st.session_state[f"col{i}"] is not None:
                append_labels()
                st.success("Submitted")
                break
        else:
            st.warning("Please fill at least one field")
