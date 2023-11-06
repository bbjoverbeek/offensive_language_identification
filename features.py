#!/usr/bin/env python

"""
Authors: Björn Overbeek & Oscar Zwagers
Description: A program that will train a classifier with the given train set to
predict if a piece of text has offensive language or not.
"""

import itertools
import pickle
import os
import argparse

import emoji
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from tqdm import tqdm

from src.evaluate import calculate_scores
from src.util import load_data, OffensiveWordReplaceOption, Token, Ngrams, Options, \
    ContentBasedFeatures, SentimentFeatures, Vectorizer, POS, Preprocessing, Algorithm, Document, \
    add_additional_information
import spacy
from transformers import pipeline


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
        "--scores_dir",
        type=str,
        default="scores",
        help="The directory where the scores are stored.",
    )

    return parser


# Functions to add additional information to the tokens


# Functions to create the vectorizers

def create_count_vectorizer(
        preprocessor, tokenizer, ngram: Ngrams
) -> CountVectorizer:
    return CountVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        token_pattern=None,
        ngram_range=(ngram.value, ngram.value),
    )


def create_tfidf_vectorizer(
        preprocessor, tokenizer, ngram: Ngrams
) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        token_pattern=None,
        ngram_range=(ngram.value, ngram.value),
    )


def create_feature_vectorizer(options: Options) -> FunctionTransformer:
    match (options.content_based_features, options.sentiment_features):
        case (ContentBasedFeatures.USE, SentimentFeatures.USE):
            return FunctionTransformer(
                lambda inp: [
                    [
                        doc.length_document,
                        doc.average_token_length,
                        doc.fraction_uppercase,
                        doc.fraction_emoji,
                        doc.sentiment
                    ]
                    for doc in inp
                ],
                validate=False
            )
        case (ContentBasedFeatures.USE, SentimentFeatures.NONE):
            return FunctionTransformer(
                lambda inp: [
                    [
                        doc.length_document,
                        doc.average_token_length,
                        doc.fraction_uppercase,
                        doc.fraction_emoji,
                    ]
                    for doc in inp
                ],
                validate=False
            )
        case (ContentBasedFeatures.NONE, SentimentFeatures.USE):
            return FunctionTransformer(
                lambda inp: [
                    [doc.sentiment]
                    for doc in inp
                ],
                validate=False
            )
        case (ContentBasedFeatures.NONE, SentimentFeatures.NONE):
            return FunctionTransformer(
                lambda inp: [
                    []
                    for _doc in inp
                ],
                validate=False
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

    features_vec = create_feature_vectorizer(options)
    vec.append(("features", features_vec))

    return FeatureUnion(vec)


# Helper functions for creating the vectorizers

def identity(inp: any) -> any:
    """Returns the input."""
    return inp


def get_token_text(inp: Document, preprocessing: Preprocessing) -> str:
    """Returns the text of the token. This can be a preprocessed token."""
    match preprocessing:
        case Preprocessing.LEMMATIZE:
            return " ".join([token.lemma for token in inp.tokens])
        case Preprocessing.NONE:
            return " ".join([token.text for token in inp.tokens])


def get_token_pos_tag(inp: Document, pos: POS) -> str:
    """Returns the POS tags of the tokens."""
    match pos:
        case POS.STANDARD:
            tags = [token.pos_standard for token in inp.tokens]
            return " ".join(tags)
        case POS.FINEGRAINED:
            tags = [token.pos_finegrained for token in inp.tokens]
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
        options: Options, train_docs: list[Document], train_labels: list[str],
) -> Pipeline:
    """
    Run the classifier with the given algorithm and train and test data.
    """
    classifier = create_classifier(options)
    classifier.fit(train_docs, train_labels)
    return classifier


def create_model_name(options: Options) -> str:
    """
    Create a name for the model based on the options.
    """
    return f"{options.algorithm.value}__{options.vectorizer.value}__{options.ngram.value}-gram__" \
        + f"{options.pos.value}__{options.preprocessing.value}" \
        + f"__{options.content_based_features.value}__{options.sentiment_features.value}" \
        + f"__{options.offensive_word_replacement.value}"


def run_models(
        all_options: list[Options],
        train_docs: list[Document],
        train_labels: list[str],
) -> None:
    models = []
    for options in tqdm(all_options, desc="Running models", leave=False):
        model = train_classifier(options, train_docs, train_labels)
        name = create_model_name(options)
        models.append((model, name))
    pickle.dump(models, open("./model.bin", "wb"))


def create_all_options(offensive_word_replace_option: OffensiveWordReplaceOption) -> list[Options]:
    """
    Create all possible options to train a classifier. This is done by creating all possible
    combinations of the different options.
    :return: a list of all possible options
    """

    # This code was created by ChatGPT, because I didn't want to use 5 nested for loops. I did not
    # know a method to create all combinations without the use of nested for loops.
    return [
        Options(
            offensive_word_replacement=offensive_word_replace_option,
            algorithm=alg,
            vectorizer=vec,
            ngram=ng,
            pos=p,
            preprocessing=prep,
            content_based_features=cont,
            sentiment_features=sent
        ) for alg, vec, ng, p, prep, cont, sent in itertools.product(
            Algorithm,
            Vectorizer,
            Ngrams,
            POS,
            Preprocessing,
            ContentBasedFeatures,
            SentimentFeatures
        )
    ]


def predict(model: Pipeline, docs: list[Document], options: Options, labels: list[str],
            precision: int, directory: str
            ) -> list[str]:
    predictions = model.predict(docs)

    name = create_model_name(options)
    scores = calculate_scores(name, predictions, labels, precision)

    directory = os.path.join(os.getcwd(), directory)
    if os.path.exists(directory) is False:
        os.mkdir(directory)

    with open(os.path.join(directory, f"{scores.name}.md"), "w") as f:
        f.write(scores.format(True))

    all_scores_file = os.path.join(directory, "all_scores.csv")
    if os.path.exists(all_scores_file) is False:
        with open(all_scores_file, "w") as f:
            f.write("name,accuracy,precision,recall,f1\n")

    with open(all_scores_file, "a") as f:
        f.write(scores.format(False))


def main():
    args = create_arg_parser().parse_args()

    offensive_word_replace_option = OffensiveWordReplaceOption.NONE
    data = load_data(args.dirname, offensive_word_replace_option, True)
    train_docs = add_additional_information(data.training.documents, False)

    all_options = [Options(
        offensive_word_replacement=offensive_word_replace_option,
        vectorizer=Vectorizer.BAG_OF_WORDS,
        ngram=Ngrams.UNIGRAM,
        pos=POS.NONE,
        algorithm=Algorithm.NAIVE_BAYES,
        preprocessing=Preprocessing.NONE,
        content_based_features=ContentBasedFeatures.NONE,
        sentiment_features=SentimentFeatures.NONE
    )]

    # all_options = create_all_options(offensive_word_replace_option)

    run_models(all_options, train_docs, data.training.labels)


if __name__ == '__main__':
    main()
