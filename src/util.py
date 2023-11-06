"""Util file for training models to identify offensive language"""
import os
from typing import NamedTuple
from enum import Enum
import json

import emoji
import spacy
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import pipeline


class DataType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class OffensiveWordReplaceOption(Enum):
    NONE = "none"
    REPLACE = "replace"
    REMOVE = "remove"

    @staticmethod
    def from_str(option):
        """Creates an OffensiveWordReplaceOption from a string"""
        match option:
            case 'none':
                return OffensiveWordReplaceOption.NONE
            case 'replace':
                return OffensiveWordReplaceOption.REPLACE
            case 'remove':
                return OffensiveWordReplaceOption.REMOVE
            case _:
                raise NotImplementedError('Use none, replace, or remove')


class DataItems(NamedTuple):
    documents: list[str]
    labels: list[str]


class Data(NamedTuple):
    training: DataItems
    development: DataItems
    test: DataItems

    def to_dataset(self) -> DatasetDict:
        """Converts the data to a Huggingface Dataset"""

        dataset = DatasetDict(
            {
                'train': Dataset.from_dict(
                    {
                        'text': self.training.documents,
                        'labels': [
                            1 if label == 'OFF' else 0 for label in self.training.labels
                        ],
                    }
                ),
                'dev': Dataset.from_dict(
                    {
                        'text': self.development.documents,
                        'labels': [
                            1 if label == 'OFF' else 0
                            for label in self.development.labels
                        ],
                    }
                ),
                'test': Dataset.from_dict(
                    {
                        'text': self.test.documents,
                        'labels': [
                            1 if label == 'OFF' else 0 for label in self.test.labels
                        ],
                    }
                ),
            }
        )

        return dataset


def load_offensive_words(file_path: str) -> set[str]:
    """Loads the offensive words from the given file path and returns them as a set"""

    offensive_words = set()

    with open(file_path, 'r', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip()
            if line:
                offensive_words.add(line)

    return offensive_words


def create_filename(
        dirname: str,
        data_type: DataType,
        replace_option: OffensiveWordReplaceOption,
        preprocessed: bool = False,
) -> str:
    """Creates a filename for the given data type and replace option"""

    filename = data_type.value
    filename += (
        "." + replace_option.value
        if replace_option != OffensiveWordReplaceOption.NONE
        else ""
    )
    filename += ".preprocessed" if preprocessed else ""
    filename += ".tsv"
    filename = os.path.join(os.getcwd(), dirname, filename)
    return filename


def load_data_file(
        dirname: str,
        data_type: DataType,
        replace_option: OffensiveWordReplaceOption,
        preprocess: bool = False,
) -> DataItems:
    filename = create_filename(dirname, data_type, replace_option, preprocess)
    tweets = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip()
            if line:
                tweet, label = line.split('\t')
                tweets.append(tweet)
                labels.append(label)
    return DataItems(tweets, labels)


def load_data(
        dirname: str, replace_option: OffensiveWordReplaceOption, preprocessed: bool = False
) -> Data:
    """Loads the data from the given directory and returns it"""

    training = load_data_file(dirname, DataType.TRAIN, replace_option, preprocessed)
    development = load_data_file(dirname, DataType.DEV, replace_option, preprocessed)
    test = load_data_file(dirname, DataType.TEST, replace_option, preprocessed)
    return Data(training, development, test)


def write_data_file(
        dirname: str,
        data_type: DataType,
        data: DataItems,
        replace_option: OffensiveWordReplaceOption,
) -> None:
    filename = create_filename(dirname, data_type, replace_option, True)

    with open(filename, 'w', encoding='utf-8') as out:
        for tweet, label in zip(data.documents, data.labels):
            out.write(tweet + '\t' + label + '\n')


def write_data(
        data: Data, dirname: str, replace_option: OffensiveWordReplaceOption
) -> None:
    write_data_file(dirname, DataType.TRAIN, data.training, replace_option)
    write_data_file(dirname, DataType.DEV, data.development, replace_option)
    write_data_file(dirname, DataType.TEST, data.test, replace_option)
    return None


def parse_config(config_file: str, default_config: dict[str, str]) -> dict[str, str]:
    """Completes the missing values from the config file with default values"""

    with open(config_file, 'r', encoding='utf-8') as inp:
        config = json.load(inp)

    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    return config


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


class ContentBasedFeatures(Enum):
    USE = "content_features"
    NONE = "none"


class SentimentFeatures(Enum):
    USE = "sentiment_features"
    NONE = "none"


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
    content_based_features: ContentBasedFeatures  # 2 options
    sentiment_features: SentimentFeatures  # 2 options
    offensive_word_replacement: OffensiveWordReplaceOption

    # 6 * 3 * 3 * 3 * 2 * 2 * 2 * 3 = 3888 options

class Document:
    """
    A document is a piece of text. This will be a tweet.
    """
    text: str
    tokens: list[Token]
    sentiment: float = None
    length_document: int = None
    average_token_length: float = None
    fraction_uppercase: float = None
    fraction_emoji: float = None

    def __init__(self, text: str, tokens: list[Token]):
        self.text = text
        self.tokens = tokens

    def __str__(self):
        return f"{{\ntext: {self.text}, \ntokens: {self.tokens}, \nsentiment: {self.sentiment}, " \
            + f"\nlength_document: {self.length_document}, " \
            + f"\naverage_token_length: {self.average_token_length}, " \
            + f"\nfraction_uppercase: {self.fraction_uppercase}, " \
            + f"\nfraction_emoji: {self.fraction_emoji}\n}}"

    def create_features(self, sentiment: float) -> None:
        amount_chars = len(self.text)
        self.fraction_emoji = emoji.emoji_count(self.text) / amount_chars
        self.fraction_uppercase = len(
            [char for char in self.text if char.isupper()]) / amount_chars
        self.length_document = len(self.text)
        self.average_token_length = sum(
            [len(token.text) for token in self.tokens]
        ) / len(self.tokens)
        self.sentiment = sentiment


def add_additional_information(docs: list[str], calculate_sentiment: bool = True) -> list[Document]:
    """
    Adds additional information to the tokens of the documents. This information includes the POS
    tags, so they don't have to be extracted every time the classifier is trained. This speeds up
    training and testing time.

    :param calculate_sentiment: when you won't use the sentiment score in the features, you can set
    this to "false" to speed up the process
    :param docs: the docs to which the POS tags are added
    :return: the docs with the POS tags
    """
    result = []

    nlp = spacy.load("en_core_web_sm")
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analyser = pipeline(
        "sentiment-analysis", SENTIMENT_MODEL
    )

    for text in tqdm(docs, desc="Adding additional information"):
        doc = nlp(text)
        tokens = [
            Token(
                text=token.text,
                pos_standard=token.pos_,
                pos_finegrained=token.tag_,
                lemma=token.lemma_
            ) for token in doc
        ]
        document = Document(text=text, tokens=tokens)
        sentiment = sentiment_analyser(text)[0] if calculate_sentiment else {"score": 0.0}
        document.create_features(sentiment["score"])
        result.append(document)

    for x in result:
        print(x)

    return result
