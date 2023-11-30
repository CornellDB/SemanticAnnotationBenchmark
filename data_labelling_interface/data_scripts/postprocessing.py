import argparse
import ast
from collections import defaultdict, Counter
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import dotenv_values
import openai

from utils import get_hierarchy, preprocess_text, calculate_bertscore_f1


config = dotenv_values("../../.env")
# openai.api_key = config["OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_API_KEY"] = config["AZURE_OPENAI_API_KEY"]
# openai.base_url = "https://sainyam-openai.openai.azure.com/"
openai.api_type = "azure_ad"
openai.api_version = "2023-05-15"
client = openai.AzureOpenAI(
    api_version="2023-03-15-preview", azure_endpoint="https://semanticannotation-aiservices.openai.azure.com"
)


def merge_col_labels(hierarchies: list[tuple[list[str], int]]) -> list[str]:
    """Merge label hierarchies from different annotators into a single ground truth hierarchy

    Args:
        hierarchies (list[tuple[list[str], int]]): list of label hierarchies from different annotators

    Returns:
        list[str]: single ground truth hierarchy
    """
    # Track parents of each concept
    ancestor_map = defaultdict(set)
    for hierarchy, annotator_idx in hierarchies:
        for i in range(1, len(hierarchy)):
            ancestor_map[hierarchy[i - 1]].add(hierarchy[i])

    # Track the count for each concept
    concept_count_by_annotators = defaultdict(set)
    for hierarchy, annotator_idx in hierarchies:
        for concept in hierarchy:
            concept_count_by_annotators[concept].add(annotator_idx)
    concept_count = {}
    for concept, unique_annotators in concept_count_by_annotators.items():
        concept_count[concept] = len(unique_annotators)

    # Filter concepts with low count
    filtered_concepts = list(filter(lambda x: x[1] > 1, concept_count.items()))

    # Use bfs to find longest path
    queue = [(concept, []) for concept, _ in filtered_concepts]
    longest_path = []

    while queue:
        cur_node, cur_path = queue.pop(0)
        new_path = cur_path + [cur_node]
        if cur_node not in ancestor_map:
            if len(new_path) > len(longest_path):
                longest_path = new_path
        else:
            for parent in ancestor_map[cur_node]:
                if parent not in new_path:  # To prevent cycles
                    queue.append((parent, new_path))

    print(longest_path)
    return longest_path


def main(label_filename: str, dataset_filename: str):
    # Load csv file with current labels from
    labels_df = pd.read_csv(label_filename)
    # Load csv file with table names annotated in dataset
    current_dataset_df = pd.read_csv(dataset_filename)
    labels_df["labels"] = labels_df["labels"].apply(ast.literal_eval)
    combined_labels = labels_df.groupby("table_name")["labels"].agg(lambda x: list(x))

    postprocessed_table = {
        "table_name": [],
        "raw_labels": [],
        "most_granular_concept_synonyms": [],
        "label_hierarchies": [],
    }
    for table_name in tqdm(current_dataset_df["name"][:40]):
        all_labels = []
        raw_labels = []
        most_granular_synonyms = []
        for label_groups in zip(*combined_labels[table_name]):
            labels = []
            labels_without_idx = []
            for idx, label_group in enumerate(label_groups):
                if isinstance(label_group, str) and label_group == "Unable to label":
                    continue
                for label in label_group:
                    preprocessed_label = preprocess_text(label)
                    labels.append((preprocessed_label, idx))
                    labels_without_idx.append(preprocessed_label)
            print("=" * 10)
            print(f"labels {labels}")

            # Generate hierarchy from a term
            hierarchies = []
            for label, annotator_idx in labels:
                # Preprocess concepts
                hierarchy = [preprocess_text(concept, True) for concept in get_hierarchy(label, client)]
                hierarchies.append((hierarchy, annotator_idx))
            print("=" * 10)
            print(f"hierarchies {hierarchies}")

            # Group common label hierarchies
            merged_label = merge_col_labels(hierarchies)
            print("=" * 10)
            print(f"merged labels {merged_label}")

            # Group up annotator suggestions as well as the most granular concept that GPT suggested
            granular_concepts = set(labels_without_idx)
            for hierarchy, _ in hierarchies:
                granular_concepts.add(hierarchy[0])

            # Get synonyms of most granular level
            synonyms = []
            if merged_label:
                most_granular_concept = merged_label[0]
                if most_granular_concept not in granular_concepts:
                    synonyms.append(most_granular_concept)
                else:
                    granular_concepts_list = list(granular_concepts)
                    bertscores = calculate_bertscore_f1(most_granular_concept, granular_concepts_list)
                    for idx, granular_concept in enumerate(granular_concepts_list):
                        print(granular_concept, most_granular_concept)
                        if bertscores[idx] > 0.6:
                            print(bertscores[idx])
                            synonyms.append(granular_concept)

            print("=" * 10)
            print(f"most granular synonyms labels {synonyms}")
            raw_labels.append(labels_without_idx)
            most_granular_synonyms.append(synonyms)
            all_labels.append(merged_label)
        postprocessed_table["table_name"].append(table_name)
        postprocessed_table["raw_labels"].append(raw_labels)
        postprocessed_table["label_hierarchies"].append(all_labels)
        postprocessed_table["most_granular_concept_synonyms"].append(most_granular_synonyms)
    new_df = pd.DataFrame(postprocessed_table)
    new_df.to_csv("postprocessed_labels.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process labels.")
    parser.add_argument(
        "--label-filename",
        type=str,
        default="../labels/labels_2023-11-28 05_25_47.457077.csv",
    )
    parser.add_argument(
        "--current-dataset-filename",
        type=str,
        default="../current_dataset.csv",
    )
    args = parser.parse_args()
    main(args.label_filename, args.current_dataset_filename)
