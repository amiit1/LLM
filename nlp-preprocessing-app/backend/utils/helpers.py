"""Utility helpers shared across the backend.

Small reusable helpers avoid duplicated utility logic in service files.
"""


def normalize_whitespace(text: str) -> str:
    """Normalize extra spaces/newlines into single spaces.

    This makes NLP output more consistent without changing word order.
    """
    return " ".join(text.split())
