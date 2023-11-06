from typing import NamedTuple

import sklearn
from pytablewriter import MarkdownTableWriter

from src import LABELS


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
