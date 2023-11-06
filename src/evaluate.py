import argparse
import json
import os
from typing import NamedTuple

import sklearn
from pytablewriter import MarkdownTableWriter

from util import DataType, OffensiveWordReplaceOption, load_data_file, LABELS, parse_config


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
        "--directory", type=str, help="The directory to use."
    )
    parser.add_argument(
        "--predictions-directory", type=str, help="The predictions input directory."
    )

    parser.add_argument(
        "--evaluation-directory", type=str, help="The evaluation output directory."
    )

    return parser


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

    def format(self, extensive: bool = False) -> str:
        """
        Creates a string from the scores, so they can be written to the console or a file.
        :param extensive: the extensive format includes the confusion matrix
        :return: the formatted scores
        """
        match extensive:
            case True:
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
            case False:
                results = [
                    self.name,
                    self.accuracy,
                    self.precision,
                    self.recall,
                    self.f1,
                ]
                return ",".join(map(str, results)) + "\n"


def calculate_scores(
        name: str, predictions: list[str], labels: list[str], precision: int
) -> Scores:
    """
    Calculate the scores of the classifier and return them as a dict.
    """

    return Scores(
        name=name,
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


def main():
    args = create_arg_parser().parse_args()

    config = parse_config(args.config_file, create_default_config())

    data_type = DataType.TEST if config["evaluation_set"] == "test" else DataType.DEV
    offensive_word_replace_option = OffensiveWordReplaceOption.from_str(config["replace_option"])

    test_data = load_data_file(
        args.directory, data_type, offensive_word_replace_option, config["preprocessed"]
    )

    files = os.listdir(args.predictions_directory)
    files = [f for f in files if os.path.isfile(args.predictions_directory + '/' + f)]

    for file in files:
        predictions = []
        with open(args.predictions_directory + '/' + file, 'r') as f:
            for line in f:
                predictions.append(line.strip())

        scores = calculate_scores(file, predictions[:10], test_data.labels[:10], 3)

        directory = os.path.join(os.getcwd(), args.evaluation_directory)

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


if __name__ == "__main__":
    main()
