from functools import lru_cache
import re
import time
from evaluate import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


bertscore = load("bertscore")


def calculate_bertscore_f1(pred: str, labels: list[str]) -> list[float]:
    """Calculate bert scores of prediction on multiple possible labels

    Args:
        pred (str): prediction
        labels (list[str]): labels to evaluate on

    Returns:
        list[float]: f1 bertscore on each label
    """
    bertscores = bertscore.compute(
        predictions=[pred] * len(labels),
        references=labels,
        lang="en",
        model_type="microsoft/deberta-large-mnli",
    )["f1"]
    return bertscores


@lru_cache()
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_hierarchy(concept: str, client) -> list[str]:
    time.sleep(2)
    prompt = (
        f"Give me the dbpedia ontology hierarchy for {concept}. "
        "If the concept cannot be found in the ontology, use the closest possible one."
        "Do not include the prefix in the concepts (e.g. foaf, owl, rdf)."
        "If no suitable concept can be found, just return the original concept."
        "Please ONLY give me in a form of a comma delimited list, with the most granular concept first and the most general concept last."
        "It must be without any reasoning or explanation."
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        temperature=0,
    )
    # print(chat_completion.choices[0])
    with open("usage.txt", "a") as f:
        f.write(
            f"{chat_completion.usage.completion_tokens}, {chat_completion.usage.prompt_tokens}, {chat_completion.usage.total_tokens}\n"
        )
    return chat_completion.choices[0].message.content.split(",")


def camel_case_split(s: str) -> str:
    """Split camel case words into a string with words separated

    Args:
        s (str): camel case string

    Returns:
        str: joined words
    """
    # use map to add an underscore before each uppercase letter
    modified_string = list(map(lambda x: "_" + x if x.isupper() else x, s))
    # join the modified string and split it at the underscores
    split_string = "".join(modified_string).split("_")
    # remove any empty strings from the list
    split_string = list(filter(lambda x: x != "", split_string))
    return " ".join(split_string)


def preprocess_text(text: str, stopword_removal_and_lemmatization: bool = False) -> str:
    """Preprocess text by removing leading and trailing spaces, punctuations, lower casing.
    Further preprocessing can be done by removing stopwords and doing lemmatization.

    Args:
        text (str): original string
        stopword_removal_and_lemmatization (bool, optional): To do further preprocessing. Defaults to False.

    Returns:
        str: preprocessed string
    """
    text = text.strip()
    text = re.sub(r"([^\w\s]|_)", "", text)
    text = " ".join([camel_case_split(word) for word in text.split()])
    text = text.lower()

    if stopword_removal_and_lemmatization:
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Remove stop words and punctuation
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
        text = " ".join(tokens)

    return text


def get_synonyms(word: str) -> set[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


if __name__ == "__main__":
    print(preprocess_text("achieving tested the teaching", True))
