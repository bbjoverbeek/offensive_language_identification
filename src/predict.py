import argparse
import json
import os.path
import pickle
from enum import Enum
from pathlib import Path
from datasets import Dataset
from transformers import pipeline as tf_pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score

import joblib
from sklearn import pipeline

from util import (
    DataItems,
    load_data_file,
    DataType,
    OffensiveWordReplaceOption,
    add_additional_information,
    Tokenizer,
    identity,
    features_vec_both,
    features_vec_content,
    features_vec_sentiment,
    features_vec_none,
    parse_config,
)


def create_default_config(create_file: bool = False) -> dict[str, str | bool]:
    """Creates a config file with default parameters, and returns them"""
    default_config = {
        'data_dir': './data',
        'preprocessed': True,  # True, False
        'replace_option': 'none',  # none, replace, remove
        'evaluation_set': 'dev',  # dev, test
    }

    if create_file:
        with open('config.json', 'x', encoding='utf-8') as config_file:
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

    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--test-data", type=str, help="The test data to use.")
    parser.add_argument("--directory", type=str, help="The directory to use.")
    parser.add_argument(
        "--predictions-directory", type=str, help="The predictions output directory."
    )

    parser.add_argument("--model-type", type=str, help="The model type.")

    return parser


class ModelType(Enum):
    BASELINE = "baseline"
    FEATURES = "features"
    LSTM = "lstm"
    PLM = "plm"


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


<<<<<<< HEAD
def predict_lstm(model, test_data: Dataset) -> dict[str, list[str]]:
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    predictions = model.predict(test_data['train'])

    predictions = np.argmax(Y_pred, axis=1)

    return {'lstm': predictions.tolist()}


def predict_plm(
    model_id: str, model_path: str, test_data: Dataset
) -> dict[str, list[str]]:
=======
def predict_plm(model_id: str, model_path: str, test_data: Dataset) -> dict[str, list[str]]:
>>>>>>> 1d758352cfc4bd24954e638536b8cafd3f622048
    """Load the fine-tuned llm and predict on the test or dev set"""

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    predictions = []
    pipe = tf_pipeline(
        'text-classification', model=model, tokenizer=tokenizer, device=0
    )
    for item in test_data:
        predictions.append(pipe(item['text']))

    return {model_id: [prediction[0]['label'] for prediction in predictions]}


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
    offensive_word_replace_option = OffensiveWordReplaceOption.from_str(
        config["replace_option"]
    )

    test_data = load_data_file(
        args.directory, data_type, offensive_word_replace_option, config["preprocessed"]
    )

<<<<<<< HEAD
    test_dataset = Dataset.from_dict({'text': test_data[0], 'labels': test_data[1]})
=======
    test_data_dataset = Dataset.from_dict({'text': test_data[0], 'labels': test_data[1]})
>>>>>>> 1d758352cfc4bd24954e638536b8cafd3f622048

    model_type = ModelType(args.model_type)
    match model_type:
        case ModelType.BASELINE:
            predict_baseline(args.model, test_data)
        case ModelType.FEATURES:
            predictions = predict_features(args.model, test_data)
            write_predictions(args.predictions_directory, predictions)
        case ModelType.LSTM:
            predictions = predict_lstm(args.model, test_dataset)
            write_predictions(args.predictions_directory, predictions)
        case ModelType.PLM:
            predictions = predict_plm(config['model_id'], args.model, test_dataset)
            write_predictions(args.predictions_directory, predictions)

if __name__ == "__main__":
    main()
