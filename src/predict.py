import argparse
import json
import os.path
import pickle
from enum import Enum
from pathlib import Path

import joblib
from sklearn import pipeline

from util import DataItems, load_data_file, DataType, OffensiveWordReplaceOption, \
    add_additional_information, Tokenizer, identity, features_vec_both, features_vec_content, \
    features_vec_sentiment, features_vec_none, parse_config


def create_default_config(create_file: bool = False) -> dict[str, str | bool]:
    """Creates a config file with default parameters, and returns them"""
    default_config = {
        'data_dir': './data',
        'preprocessed': True,  # True, False
        'replace_option': 'none',  # none, replace, remove
        'evaluation_set': 'dev',  # dev, test
    }

    if create_file:
        with open('features.json', 'x', encoding='utf-8') as config_file:
            json.dump(default_config, config_file, indent=4)

    return default_config


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the predict script.
    :return: The argument parser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help='Config file to use',
    )

    parser.add_argument(
        "--model", type=str, help="The model to use."
    )
    parser.add_argument(
        "--test-data", type=str, help="The test data to use."
    )
    parser.add_argument(
        "--directory", type=str, help="The directory to use."
    )
    parser.add_argument(
        "--predictions-directory", type=str, help="The predictions output directory."
    )

    parser.add_argument(
        "--model-type", type=str, help="The model type."
    )

    return parser


class ModelType(Enum):
    BASELINE = "baseline"
    FEATURES = "features"
    LSTM = "lstm"
    LLM = "llm"


def predict_baseline(name: str, test_data: DataItems) -> dict[str, list[str]]:
    with open(name, "rb") as f:
        model: tuple[pipeline.Pipeline, str] = pickle.load(f)

    predictions = {}
    docs = add_additional_information(test_data.documents, True)

    prediction = model[0].predict(docs).tolist()
    predictions[model[1]] = prediction

    return predictions


def predict_features(name: str, test_data: DataItems) -> dict[str, list[str]]:
    models: list[tuple[pipeline.Pipeline, str]] = joblib.load(name)
    predictions = {}
    docs = add_additional_information(test_data.documents, True)

    for model, name in models:
        prediction = model.predict(docs).tolist()
        predictions[name] = prediction

    return predictions


def write_predictions(directory: str, predictions: dict[str, list[str]]) -> None:
    for name, prediction in predictions.items():
        filename = os.path.join(directory, name + ".txt")
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(prediction))


def main() -> None:
    args = create_arg_parser().parse_args()

    config = parse_config(args.config_file, create_default_config())

    data_type = DataType.TEST if config["evaluation_set"] == "test" else DataType.DEV
    offensive_word_replace_option = OffensiveWordReplaceOption.from_str(config["replace_option"])

    test_data = load_data_file(
        args.directory, data_type, offensive_word_replace_option, config["preprocessed"]
    )

    model_type = ModelType(args.model_type)
    match model_type:
        case ModelType.BASELINE:
            predict_baseline(args.model, test_data)
        case ModelType.FEATURES:
            predictions = predict_features(args.model, test_data)
            print(predictions)
            write_predictions(args.predictions_directory, predictions)
        case ModelType.LSTM:
            pass
        case ModelType.LLM:
            pass


if __name__ == "__main__":
    main()
