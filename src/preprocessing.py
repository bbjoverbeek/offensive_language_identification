import emoji
import re
from hashformers import TransformerWordSegmenter as WordSegmenter
from util import load_offensive_words, load_data, OffensiveWordReplaceOption

ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    segmenter_model_type="incremental",
    reranker_model_name_or_path="google/flan-t5-base",
    reranker_model_type="seq2seq"
)


def emoji_to_text(text: str) -> str:
    """
    Converts emoji to text
    :param text: the text to convert
    :return: the converted text
    """
    return emoji.demojize(text, delimiters=("", ""))


def url_to_http(text: str) -> str:
    """
    Converts a URL to http
    :param text: the text to convert
    :return: the converted text
    """
    return re.sub(r'URL', 'HTTP', text)


def hashtag_segmentation(text: str) -> str:
    """
    Segments all hashtags that can be found in a piece of text (most of the time a tweet). This
    means that the hashtag '#thisisatest' will be segmented to 'this is a test'.
    :param text: a piece of text, most of the time a tweet
    :return: the text with segmented hashtags
    """
    hashtags = re.findall(r"#(\w+)", text)
    texts = ws.segment(hashtags)
    for (item, hashtag) in zip(texts, hashtags):
        re.sub(r'#' + hashtag, item, text)
    return text




def offensive_word_replacement(
        text: str,
        offensive_words_pattern: re.Pattern,
        option: OffensiveWordReplaceOption
) -> str:
    return text


def create_offensive_words_pattern(offensive_words: set[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(offensive_words) + r")\b", re.IGNORECASE)


def preprocess(
        text: str,
        offensive_words_pattern: re.Pattern,
        option: OffensiveWordReplaceOption
) -> str:
    """
    Preprocesses the given text
    :param text: the text to preprocess
    :return: the preprocessed text
    """
    text = emoji_to_text(text)
    text = url_to_http(text)
    text = hashtag_segmentation(text)
    text = offensive_word_replacement(text, offensive_words_pattern, option)
    return text


def main():
    offensive_words = load_offensive_words("../data/offensive_words.txt")
    pattern = create_offensive_words_pattern(offensive_words)
    data = load_data("../data", OffensiveWordReplaceOption.NONE)

    tweet = data.training.documents[0]
    print(tweet)
    preprocessed = preprocess(tweet, pattern, OffensiveWordReplaceOption.NONE)
    print(preprocessed)





if __name__ == "__main__":
    main()
