from collections import defaultdict
from dotenv import dotenv_values
import pinecone
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm


def upsert_data(index_name: str, entity_source_dict: dict[str, list], model):
    index = pinecone.Index(index_name)
    terms = []
    for term, source in tqdm(entity_source_dict.items()):
        terms.append(
            (
                term,
                model.encode(term).tolist(),
                {"source": source},
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
        vector=model.encode(query_string.lower()).tolist(),
    )


if __name__ == "__main__":
    # Load env file with API KEY using full path
    config = dotenv_values("../.env")
    all_entities = defaultdict(list)

    df = pd.read_csv("ibm_glossary_terms.csv")
    ibm_concepts = df["terms"].to_list()
    for concept in ibm_concepts:
        all_entities[concept.lower()].append("ibm")
    with open("dbpedia_concepts2.txt", "r") as f:
        dbpedia_concepts = f.readlines()
    with open("wikidata_concepts2.txt", "r") as f:
        wikidata_concepts = f.readlines()
    for item in dbpedia_concepts:
        all_entities[item.strip("\n").lower()].append("dbpedia")
    for concept in wikidata_concepts:
        all_entities[concept.strip("b").strip("\n").strip("'").lower()].append("wikidata")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    pinecone.init(api_key=config["PINECONE_API_KEY"], environment="gcp-starter")
    index_name = "semantic-annotation"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)
        upsert_data(index_name, all_entities, model)
    # index = pinecone.Index(index_name)
    # print(query(model, index, "federated"))
