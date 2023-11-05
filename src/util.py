"""Util file for training models to identify offensive language"""
import os
from typing import NamedTuple
from enum import Enum


class DataType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class OffensiveWordReplaceOption(Enum):
    NONE = "none"
    REPLACE = "replace"
    REMOVE = "remove"


class DataItems(NamedTuple):
    documents: list[str]
    labels: list[str]


class Data(NamedTuple):
    training: DataItems
    development: DataItems
    test: DataItems


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
        preprocessed: bool = False
) -> str:
    """Creates a filename for the given data type and replace option"""

    filename = data_type.value
    filename += "." + replace_option.value \
        if replace_option != OffensiveWordReplaceOption.NONE else ""
    filename += ".preprocessed" if preprocessed else ""
    filename += ".tsv"
    filename = os.path.join(os.getcwd(), dirname, filename)
    return filename


def load_data_file(
        dirname: str,
        data_type: DataType,
        replace_option: OffensiveWordReplaceOption,
        preprocess: bool = False
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
        replace_option: OffensiveWordReplaceOption
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
