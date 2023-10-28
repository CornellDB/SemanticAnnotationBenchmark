import os
from dotenv import dotenv_values
import pinecone
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm


def upsert_data(index_name: str, data: list[str], model):
    index = pinecone.Index(index_name)
    terms = []
    for i, term in enumerate(tqdm(data)):
        terms.append(
            (
                str(i),
                model.encode(term).tolist(),
                {"original": term},
            )  # vector id  # dense vector  # metadata
        )
        if len(terms) == 100:
            index.upsert(vectors=terms)
            terms = []
    return index.upsert(vectors=terms)


def query(model, index, query_string: str):
    return index.query(
        top_k=5,
        include_metadata=True,
        vector=model.encode(query_string).tolist(),
    )


if __name__ == "__main__":
    # Load env file with API KEY using full path
    config = dotenv_values("../.env")

    df = pd.read_csv("ibm_glossary_terms.csv")
    concepts = []
    with open("dbpedia_concepts2.txt", "r") as f:
        dbpedia_concepts = f.readlines()
    with open("wikidata_concepts2.txt", "r") as f:
        wikidata_concepts = f.readlines()
    concepts = [item.strip("\n") for item in dbpedia_concepts]
    for concept in wikidata_concepts:
        concepts.append(concept.strip("b").strip("\n").strip("'"))
    print(concepts[:5], concepts[-5:])
    model = SentenceTransformer("all-MiniLM-L6-v2")  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    pinecone.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    index_name = "semantic-annotation"
    # if index_name not in pinecone.list_indexes():
    # pinecone.create_index(index_name, dimension=384)
    ibm_concepts = df["terms"].to_list()
    upsert_data(index_name, ibm_concepts + concepts, model)
    # index = pinecone.Index(index_name)
    # print(query(model, index, "federated"))
