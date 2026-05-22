import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


FALLBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def _load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOP_WORDS


def build_preprocessor():
    stop_words = _load_stop_words()
    lemmatizer = WordNetLemmatizer()
    url_re = re.compile(r"http[s]?://\S+|www\.\S+")
    punct_re = re.compile(rf"[{re.escape(string.punctuation)}]")

    def preprocess(text):
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        text = url_re.sub(" ", text)
        text = punct_re.sub(" ", text)
        tokens = [tok for tok in text.split() if tok and tok not in stop_words]
        try:
            tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        except LookupError:
            pass
        return " ".join(tokens)

    return preprocess
