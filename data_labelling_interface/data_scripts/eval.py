import argparse
import ast
import csv
import os
from dotenv import dotenv_values
import openai
import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

from utils import get_hierarchy, preprocess_text, calculate_bertscore_f1

bertscore = load("bertscore")
rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


config = dotenv_values("../../.env")
os.environ["AZURE_OPENAI_API_KEY"] = config["AZURE_OPENAI_API_KEY"]
# openai.base_url = "https://sainyam-openai.openai.azure.com/"
openai.api_type = "azure_ad"
openai.api_version = "2023-05-15"
client = openai.AzureOpenAI(
    api_version="2023-03-15-preview", azure_endpoint="https://semanticannotation-aiservices.openai.azure.com"
)


def rougescore_based_matching(pred: str, labels: list[str]) -> float:
    """Calculate score by finding the most granular hierarchy level where prediction matches using rouge

    Args:
        pred (str): prediction
        labels (list[str]): label hierarchy

    Returns:
        float: score (lower if pred matches at a more generic hierarchy level)
    """
    print("Rougescore based matching")
    print(f"Pred {pred}, Label hierarchy {labels}")
    for hierarchy_level, label in enumerate(labels):
        print(preprocess_text(pred), preprocess_text(label))
        rouge_results = rouge.score(prediction=preprocess_text(pred, True), target=preprocess_text(label, True))
        if rouge_results["rouge1"].fmeasure > 0.5:
            print(f"Found match {label}, rougescore {rouge_results['rouge1']}")
            return 1 - hierarchy_level / len(labels)
    return 0


def bertscore_based_matching(pred: str, labels: list[str]) -> float:
    """Calculate score by finding the most granular hierarchy level where prediction matches using bertscore

    Args:
        pred (str): prediction
        labels (list[str]): label hierarchy

    Returns:
        float: score (lower if pred matches at a more generic hierarchy level)
    """
    print("bertscore based matching")
    print(f"Pred {pred}, Label hierarchy {labels}")
    if len(labels) == 0:
        return 0
    bertscores = calculate_bertscore_f1(pred, labels)
    for hierarchy_level, label in enumerate(labels):
        if bertscores[hierarchy_level] > 0.6:
            print(f"Found match {label}, bert score {bertscores[0]}")
            return 1 - hierarchy_level / len(labels)
    return 0


def hierarchy_matching(pred: str, labels: str):
    """Run rouge score to compare two hierarchies

    Args:
        pred (str): prediction hierarchy
        labels (str): label hierarchy

    Returns:
        Score: rouge Score object
    """
    print("hierarchy based matching")
    print(f"Pred hierarchy {pred}, Label hierarchy {labels}")
    rouge_results = rouge.score(prediction=pred, target=labels)
    print(rouge_results)
    return rouge_results["rouge1"]


def eval(postprocessed_labels: str, inference_outputs: str, results_file: str):
    """Run evaluation on csv with inference output and csv with postprocessed labels

    Args:
        postprocessed_labels (str): filename of csv with postprocessed labels
        inference_outputs (str): filename of csv with inference output
    """
    gt_labels = pd.read_csv(postprocessed_labels)
    gt_labels["label_hierarchies"] = gt_labels["label_hierarchies"].apply(ast.literal_eval)
    preds = pd.read_csv(inference_outputs)
    preds["output"] = preds["output"].apply(ast.literal_eval)

    rougebased_score = bertbased_score = total_columns = 0
    hierarchy_precision = hierarchy_recall = hierarchy_fmeasure = 0
    for idx, pred in enumerate(preds["output"][: len(gt_labels)]):
        label_hierarchies = gt_labels["label_hierarchies"][idx]
        print("=" * 10, gt_labels["table_name"][idx], "=" * 10)
        col_idx = 0
        for col_labels, col_pred in zip(label_hierarchies, pred):
            if len(col_labels) == 0:
                col_idx += 1
                continue
            total_columns += 1
            bertbased_score += bertscore_based_matching(col_pred, col_labels)
            rougebased_score += rougescore_based_matching(col_pred, col_labels)
            col_pred_hierarchy = [preprocess_text(concept, True) for concept in get_hierarchy(col_pred, client)]
            hierarchy_score = hierarchy_matching(" ".join(col_pred_hierarchy), " ".join(col_labels))
            with open("debug_hierarchy.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(
                    [
                        [
                            idx,
                            col_idx,
                            " ".join(col_pred_hierarchy),
                            " ".join(col_labels),
                            hierarchy_score.precision,
                            hierarchy_score.recall,
                            hierarchy_score.fmeasure,
                        ]
                    ]
                )
            hierarchy_precision += hierarchy_score.precision
            hierarchy_recall += hierarchy_score.recall
            hierarchy_fmeasure += hierarchy_score.fmeasure
            col_idx += 1

    with open(f"{results_file}.txt", "w") as f:
        f.write(f"Total columns {total_columns}\n")
        f.write(f"Total score (bertscore) {bertbased_score}\n")
        f.write(f"Accuracy (bertscore) {bertbased_score / total_columns}\n")
        f.write(f"Total score (rougescore) {rougebased_score}\n")
        f.write(f"Accuracy (rougescore) {rougebased_score / total_columns}\n")
        f.write(f"Hierarchy precision (hierarchy score) {hierarchy_precision / total_columns}\n")
        f.write(f"Hierarchy recall (hierarchy score) {hierarchy_recall / total_columns}\n")
        f.write(f"Hierarchy fmeasure (hierarchy score) {hierarchy_fmeasure / total_columns}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a submission folder for the assignment.")
    parser.add_argument(
        "--postprocessed-labels",
        type=str,
        default="postprocessed_labels.csv",
    )
    parser.add_argument(
        "--inference-outputs",
        type=str,
        default="outputs.csv",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="experiment_results",
    )
    args = parser.parse_args()
    eval(args.postprocessed_labels, args.inference_outputs, args.results)
