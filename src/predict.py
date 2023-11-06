import argparse
import os.path
import pickle
from enum import Enum

from sklearn import pipeline

from src.util import DataItems, load_data_file, DataType, OffensiveWordReplaceOption, \
    add_additional_information


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the predict script.
    :return: The argument parser.
    """

    parser = argparse.ArgumentParser()

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
    with open(name, "rb") as f:
        models: list[tuple[pipeline.Pipeline, str]] = pickle.load(f)
    predictions = {}
    docs = add_additional_information(test_data.documents, True)

    for model, name in models:
        prediction = model.predict(docs).tolist()
        predictions[name] = prediction

    return predictions


def write_predictions(directory: str, predictions: dict[str, list[str]]) -> None:
    for name, prediction in predictions.items():
        filename = os.path.join(directory, name + ".txt")

        with open(filename, "w") as f:
            f.write("\n".join(prediction))


def main() -> None:
    args = create_arg_parser().parse_args()
    data_type = DataType.TEST if args.test_data == "test" else DataType.DEV
    offensive_word_replace_option = OffensiveWordReplaceOption.from_str(
        "none"
    )

    test_data = load_data_file(
        args.directory, data_type, offensive_word_replace_option, True
    )

    match args.model_type:
        case ModelType.BASELINE:
            predict_baseline(args.name, test_data)
        case ModelType.FEATURES:
            predict_features(args.name, test_data)
        case ModelType.LSTM:
            pass
        case ModelType.LLM:
            pass


if __name__ == "__main__":
    main()
