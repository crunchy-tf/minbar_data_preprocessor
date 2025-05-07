# services/data_preprocessor/app/processing/cleaning.py
import re
import sys # Import sys for lxml check
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
from app.core.config import logger # Use shared logger
from typing import Optional # <--- THIS LINE WAS ADDED

# Ignore specific BeautifulSoup warnings if they become too noisy
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

URL_REGEX = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
EMAIL_REGEX = re.compile(r'\S+@\S+')
MENTION_REGEX = re.compile(r'@\w+')
HASHTAG_REGEX = re.compile(r'#\w+')
# Keep letters (incl. Arabic), numbers, and whitespace. Remove most punctuation/symbols.
NON_ALPHANUM_SPACE_REGEX = re.compile(r'[^\w\s]', flags=re.UNICODE)
WHITESPACE_REGEX = re.compile(r'\s+')

def basic_text_clean(text: Optional[str]) -> str:
    """
    Performs basic text cleaning: HTML, URLs, emails, mentions, hashtags,
    special chars, normalizes whitespace. Handles None input.
    """
    if text is None:
        return ""
    if not isinstance(text, str) or not text.strip():
        return "" # Return empty string if not a string or only whitespace

    # 1. Remove HTML tags
    try:
        # Use lxml if installed (added to Dockerfile dependencies), otherwise fallback
        parser = "lxml" if 'lxml' in sys.modules else "html.parser"
        soup = BeautifulSoup(text, parser)
        text = soup.get_text(separator=" ")
    except Exception as e:
        logger.warning(f"BeautifulSoup failed on text snippet: {text[:100]}... Error: {e}")
        # Fallback: try to remove tags with regex if BS fails badly, though less robust
        text = re.sub(r'<[^>]+>', ' ', text)


    # 2. Remove URLs
    text = URL_REGEX.sub(' ', text)

    # 3. Remove email addresses
    text = EMAIL_REGEX.sub(' ', text)

    # 4. Remove mentions and hashtags
    text = MENTION_REGEX.sub(' ', text)
    text = HASHTAG_REGEX.sub(' ', text)

    # 5. Remove special characters and punctuation (keeping letters, numbers, whitespace)
    text = NON_ALPHANUM_SPACE_REGEX.sub(' ', text)

    # 6. Normalize whitespace (replace multiple spaces/newlines/tabs with a single space)
    text = WHITESPACE_REGEX.sub(' ', text).strip()

    logger.trace(f"Cleaned text snippet: {text[:100]}...")
    return text