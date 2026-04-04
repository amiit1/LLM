"""Core NLP service functions.

Each function in this module is intentionally independent and reusable.
The API layer calls these functions, but they can also be reused in scripts,
tests, or other interfaces.
"""

from functools import lru_cache

import spacy
from nltk.stem import PorterStemmer

from utils.helpers import normalize_whitespace


@lru_cache(maxsize=1)
def get_nlp():
    """Load and cache the spaCy model once for better performance.

    Why this approach:
    - Loading models repeatedly is expensive.
    - A cached singleton-like loader keeps things fast and simple.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from error


def tokenize(text: str) -> list[str]:
    """Split text into tokens using spaCy tokenization."""
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)
    return [token.text for token in doc if not token.is_space]


def lemmatize(text: str) -> list[str]:
    """Return lemmatized form of each token.

    Lemmatization is dictionary-based and context-aware, which generally
    produces valid base words.
    """
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)
    return [token.lemma_ for token in doc if not token.is_space]


def stem(text: str) -> list[str]:
    """Return stems using NLTK's PorterStemmer.

    Stemming is rule-based and does not use linguistic context.
    This is why stems can sometimes look unnatural.
    """
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)
    stemmer = PorterStemmer()

    stems: list[str] = []
    for token in doc:
        if token.is_space:
            continue

        # Apply stemming to alphabetic tokens; keep punctuation as-is.
        if token.is_alpha:
            stems.append(stemmer.stem(token.text.lower()))
        else:
            stems.append(token.text)

    return stems


def pos_tag(text: str) -> list[dict[str, str]]:
    """Return part-of-speech information for each token."""
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)

    results: list[dict[str, str]] = []
    for token in doc:
        if token.is_space:
            continue

        results.append(
            {
                "token": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "description": spacy.explain(token.tag_) or "",
            }
        )

    return results


def ner(text: str) -> list[dict[str, str]]:
    """Extract named entities from text.

    NER uses the statistical model in spaCy to identify people, places,
    organizations, dates, and more.
    """
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)

    entities: list[dict[str, str]] = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_) or "",
            }
        )

    return entities


def analyze(text: str) -> dict:
    """Run all NLP preprocessing tasks and return a unified response.

    This parses the text once to avoid duplicate work and improve efficiency.
    """
    cleaned_text = normalize_whitespace(text)
    doc = get_nlp()(cleaned_text)
    stemmer = PorterStemmer()

    tokens = [token.text for token in doc if not token.is_space]
    lemmas = [token.lemma_ for token in doc if not token.is_space]

    stems: list[str] = []
    pos_tags: list[dict[str, str]] = []

    for token in doc:
        if token.is_space:
            continue

        stems.append(stemmer.stem(token.text.lower()) if token.is_alpha else token.text)
        pos_tags.append(
            {
                "token": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "description": spacy.explain(token.tag_) or "",
            }
        )

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_) or "",
        }
        for ent in doc.ents
    ]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "stems": stems,
        "pos_tags": pos_tags,
        "entities": entities,
    }
