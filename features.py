#!/usr/bin/env python

"""
Authors: Björn Overbeek & Oscar Zwagers
Description: A program that will train a classifier with the given train set to
predict if a piece of text has offensive language or not.
"""

import itertools
import pickle
from typing import NamedTuple
from enum import Enum
import os
import argparse

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from src.util import get_data
import spacy
from pytablewriter import MarkdownTableWriter

nlp = spacy.load("en_core_web_sm")
LABELS = ["OFF", "NOT"]


# Function to open the data and to run the program with certain arguments


def create_arg_parser() -> argparse.ArgumentParser:
    """
    There are some basic arguments that can be given to the program, such as the directory where
    to find the train.tsv, dev.tsv and test.tsv files. The program can also be run with the test
    data, instead of the dev data. You can also choose to run the optimized models instead of the
    baseline models.
    :return: the arguments that are given to the program
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the classifier with the test data.",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default="data",
        help="The directory where the data is stored.",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Run the optimized models.",
    )
    return parser


# The different options that can be given to the classifier.

class Algorithm(Enum):
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREES = "decision_trees"
    RANDOM_FORESTS = "random_forests"
    KNN = "knn"
    SVC = "svc"
    LINEAR_SVC = "linear_svc"


class Vectorizer(Enum):
    BAG_OF_WORDS = "bag_of_words"
    TFI_DF = "tfi_df"
    BOTH = "both"


class Ngrams(Enum):
    UNIGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3


class POS(Enum):
    """
    The different POS taggers that can be used. They are encoded into vectors by making use of the
    bag of words vectorizer (CountVectorizer).

    The features are then combined with the features from the vectorizer that is used for the text.
    """
    STANDARD = "standard"
    FINEGRAINED = "finegrained"
    NONE = "none"


class Preprocessing(Enum):
    LEMMATIZE = "lemmatize"
    NONE = "none"


class Token(NamedTuple):
    """
    A token is a word in a sentence. We extract more information from this token, such as the POS
    tags. We store this information in this class before training the models, so we don't have to
    extract the information every time we train a model. This speeds up training and testing time.

    The POS tags are extracted using the spaCy library.
    """
    text: str
    lemma: str
    pos_standard: str
    pos_finegrained: str


class Options(NamedTuple):
    """
    The options with which the classifier is trained.
    """
    algorithm: Algorithm  # 6 options
    vectorizer: Vectorizer  # 3 options
    ngram: Ngrams  # 3 options
    pos: POS  # 3 options
    preprocessing: Preprocessing  # 2 options

    # 6 * 3 * 3 * 3 * 2 = 324 options


# Functions to add additional information to the tokens

def add_additional_information(documents: list[str]) -> list[list[Token]]:
    """
    Adds additional information to the tokens of the documents. This information includes the POS
    tags, so they don't have to be extracted every time the classifier is trained. This speeds up
    training and testing time.

    :param documents: the docs to which the POS tags are added
    :return: the docs with the POS tags
    """
    result = []

    for document in documents:
        doc = nlp(document)
        tokens = [
            Token(
                text=token.text,
                pos_standard=token.pos_,
                pos_finegrained=token.tag_,
                lemma=token.lemma_
            ) for token in doc
        ]

        result.append(tokens)

    return result


# Functions to create the vectorizers

def create_count_vectorizer(
        preprocessor, tokenizer, ngram: Ngrams
) -> CountVectorizer:
    return CountVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        ngram_range=(ngram.value, ngram.value),
    )


def create_tfidf_vectorizer(
        preprocessor, tokenizer, ngram: Ngrams
) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        ngram_range=(ngram.value, ngram.value),
    )


def create_vectorizer(options: Options) -> any:
    """
    Given the options to train a classifier, a vectorizer is created. This vectorizer is used to
    encode the text into vectors, which can then be used to train the classifier.
    :param options: a tuple of options with which the classifier is trained
    :return: a vectorizer is returned
    """
    vec = []
    match options.vectorizer:
        case Vectorizer.BAG_OF_WORDS:
            vec.append(("bow", create_count_vectorizer(
                lambda inp: get_token_text(inp, options.preprocessing), identity, options.ngram
            )))
        case Vectorizer.TFI_DF:
            vec.append(("tfidf", create_tfidf_vectorizer(
                lambda inp: get_token_text(inp, options.preprocessing), identity, options.ngram
            )))
        case Vectorizer.BOTH:
            vec.append(("bow", create_count_vectorizer(
                lambda inp: get_token_text(inp, options.preprocessing), identity, options.ngram
            )))
            vec.append(("tfidf", create_tfidf_vectorizer(
                lambda inp: get_token_text(inp, options.preprocessing), identity, options.ngram
            )))

    match options.pos:
        case POS.STANDARD | POS.FINEGRAINED:
            name = "pos_standard" if options.pos == POS.STANDARD else "pos_finegrained"
            vec.append((name, create_count_vectorizer(
                lambda inp: get_token_pos_tag(inp, options.pos), identity, options.ngram
            )))

    return FeatureUnion(vec)


# Helper functions for creating the vectorizers

def identity(inp: any) -> any:
    """Returns the input."""
    return inp


def get_token_text(inp: list[Token], preprocessing: Preprocessing) -> str:
    """Returns the text of the token. This can be a preprocessed token."""
    match preprocessing:
        case Preprocessing.LEMMATIZE:
            return " ".join([token.lemma for token in inp])
        case Preprocessing.NONE:
            return " ".join([token.text for token in inp])


def get_token_pos_tag(inp: list[Token], pos: POS) -> str:
    """Returns the POS tags of the tokens."""
    match pos:
        case POS.STANDARD:
            tags = [token.pos_standard for token in inp]
            return " ".join(tags)
        case POS.FINEGRAINED:
            tags = [token.pos_finegrained for token in inp]
            return " ".join(tags)
        case POS.NONE:
            return ""


# Functions to create the classifier algorithms

def create_classifier_algorithm(algorithm: Algorithm):
    """Given the algorithm name, a classifier from sklearn is returned."""

    match algorithm:
        case Algorithm.NAIVE_BAYES:
            # cls = MultinomialNB(alpha=5, fit_prior=True)
            # used by "Zeyad at SemEval-2019 Task 6: That’s Offensive! An All-Out Search For An
            # Ensemble To Identify And Categorize Offense in Tweets"

            # Emad at SemEval-2019 Task 6: Offensive Language Identification using Traditional
            # Machine Learning and Deep Learning approaches
            return MultinomialNB()
        case Algorithm.DECISION_TREES:
            return DecisionTreeClassifier()
        case Algorithm.RANDOM_FORESTS:
            return RandomForestClassifier()
        case Algorithm.KNN:
            return KNeighborsClassifier()
        case Algorithm.SVC:
            # SINAI at SemEval-2019 Task 6: Incorporating lexicon knowledge into SVM learning to
            # identify and categorize offensive language in social media
            # has SVC with C = 1.0, which is standard
            return SVC()
        case Algorithm.LINEAR_SVC:
            return LinearSVC(dual=False)


# Functions to create the classifier with the vectorizer

def create_classifier(options: Options) -> Pipeline:
    """
    Create a classifier with a pipeline using a vectorizer and a classifier
    algorithm. Options for the vectorizer and classifier can be specified with
    the command line options.
    """
    vec = create_vectorizer(options)
    cls = create_classifier_algorithm(options.algorithm)
    return Pipeline([('vec', vec), ('cls', cls)])


def train_classifier(
        options: Options, train_docs: list[list[Token]], train_labels: list[str],
) -> Pipeline:
    """
    Run the classifier with the given algorithm and train and test data.
    """
    classifier = create_classifier(options)
    classifier.fit(train_docs, train_labels)
    return classifier


class Scores(NamedTuple):
    """
    The scores of the classifier.
    """
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int | str]]

    def __str__(self):
        output = f"# Scores for {self.name}\n"
        results = [
            ["Accuracy", self.accuracy],
            ["Precision (macro)", self.precision],
            ["Recall (macro)", self.recall],
            ["F1 (macro)", self.f1],
        ]
        output += str(MarkdownTableWriter(headers=["Score", "Value"], value_matrix=results))
        output += f"\n### Confusion matrix:\n"
        cm = MarkdownTableWriter(headers=LABELS, value_matrix=self.confusion_matrix)
        output += str(cm)

        return output


def create_model_name(options: Options) -> str:
    """
    Create a name for the model based on the options.
    """
    return f"{options.algorithm.value}__{options.vectorizer.value}__{options.ngram.value}-gram__" \
        + f"{options.pos.value}__{options.preprocessing.value}"


def calculate_scores(
        options: Options, predictions: list[str], labels: list[str], precision: int
) -> Scores:
    """
    Calculate the scores of the classifier and return them as a dict.
    """

    return Scores(
        name=create_model_name(options),
        accuracy=round(sklearn.metrics.accuracy_score(labels, predictions), precision),
        precision=round(
            sklearn.metrics.precision_score(labels, predictions, average="macro"), precision
        ),
        recall=round(sklearn.metrics.recall_score(labels, predictions, average="macro"), precision),
        f1=round(sklearn.metrics.f1_score(labels, predictions, average="macro"), precision),
        confusion_matrix=sklearn.metrics.confusion_matrix(
            labels, predictions, labels=LABELS
        ).tolist(),
    )


def run_models(
        all_options: list[Options],
        train_docs: list[list[Token]],
        train_labels: list[str],
        docs: list[list[Token]],
        labels: list[str],
        precision: int = 3,
) -> None:
    for options in all_options:
        model = train_classifier(options, train_docs, train_labels)
        predictions = model.predict(docs)
        scores = calculate_scores(options, predictions, labels, precision)
        print(scores)


def save_model(model: Pipeline, options: Options) -> None:
    """
    Save the model to a file.
    """
    filename = f"models/{create_model_name(options)}.model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def create_all_options() -> list[Options]:
    """
    Create all possible options to train a classifier. This is done by creating all possible
    combinations of the different options.
    :return: a list of all possible options
    """

    # This code was created by ChatGPT, because I didn't want to use 5 nested for loops. I did not
    # know a method to create all combinations without the use of nested for loops.
    return [
        Options(
            algorithm=alg,
            vectorizer=vec,
            ngram=ng,
            pos=p,
            preprocessing=prep
        ) for alg, vec, ng, p, prep in itertools.product(
            Algorithm, Vectorizer, Ngrams, POS, Preprocessing
        )
    ]


def main():
    args = create_arg_parser().parse_args()

    data = get_data(args.dirname)

    # These docs and labels are used for testing. You can use the dev data or the test data.
    (docs, labels) = (data.test.documents, data.test.labels) if args.test else (
        data.development.documents, data.development.labels
    )

    (train_docs, train_labels, docs, labels) = (
        data.training.documents[:64], data.training.labels[:64], docs[:64], labels[:64]
    )

    train_docs = add_additional_information(train_docs)
    docs = add_additional_information(docs)

    all_options = [Options(
        algorithm=Algorithm.NAIVE_BAYES,
        vectorizer=Vectorizer.BAG_OF_WORDS,
        ngram=Ngrams.UNIGRAM,
        pos=POS.NONE,
        preprocessing=Preprocessing.NONE
    )]

    # if args.optimized:
    #     all_options = create_all_options()
    # else:
    #     all_options = [
    #         Options(
    #             algorithm=algorithm,
    #             vectorizer=Vectorizer.BAG_OF_WORDS,
    #             ngram=Ngrams.UNIGRAM,
    #             pos=POS.NONE,
    #             preprocessing=Preprocessing.NONE
    #         ) for algorithm in Algorithm
    #     ]

    run_models(all_options, train_docs, train_labels, docs, labels)


if __name__ == '__main__':
    main()
