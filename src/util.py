"""Util file for training models to identify offensive language"""


def load_data(file_path: str) -> tuple[list[str], list[str]]:
    """Loads the data from the given file path and returns the tweets and labels"""

    tweets = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip()
            if line:
                tweet, label = line.split('\t')
                tweets.append(tweet)
                labels.append(int(label))

    return tweets, labels
