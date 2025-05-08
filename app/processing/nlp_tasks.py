# services/data_preprocessor/app/processing/nlp_tasks.py
from typing import Tuple, List, Optional, Dict, Any

# Language Detection
from langdetect import detect, DetectorFactory, LangDetectException
try:
    DetectorFactory.seed = 0 # For consistent results across runs
except NameError: pass

# NLTK
import nltk
import nltk.data # Explicitly import nltk.data
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
# nltk.tokenize.word_tokenize is used, but oftenpunkt needs to be found first.

# spaCy
import spacy

# Stanza
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    stanza = None

from app.core.config import settings, logger

# --- NLTK Path Configuration ---
# NLTK should automatically pick up the NLTK_DATA environment variable set in the Dockerfile.
# Logging the path NLTK is using can be helpful for debugging.
logger.info(f"NLTK will search for data in these paths (includes NLTK_DATA env var if set): {nltk.data.path}")

# --- Global NLP Resources (Lazy Loaded) ---
_nlp_spacy_en: Optional[spacy.Language] = None
_nlp_spacy_fr: Optional[spacy.Language] = None
_nlp_stanza_ar: Optional['stanza.Pipeline'] = None
_nltk_en_stopwords: Optional[set] = None
_nltk_fr_stopwords: Optional[set] = None
_nltk_ar_stopwords: Optional[set] = None
_lemmatizer_en: Optional[WordNetLemmatizer] = None
_resources_loaded = False
STANZA_RESOURCES_DIR = "/app/.stanza_resources" # This should match ENV var in Dockerfile

# --- Resource Loading Functions ---
def _load_nltk_resources():
    """Loads required NLTK data, relying on build download and configured path."""
    global _nltk_en_stopwords, _nltk_fr_stopwords, _nltk_ar_stopwords, _lemmatizer_en

    # NLTK_DATA env var should make NLTK aware of the main data directory.
    # Log current paths again just to be sure at the point of loading.
    logger.debug(f"Inside _load_nltk_resources, NLTK paths: {nltk.data.path}")

    required_nltk_datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    try:
        logger.info("Verifying required NLTK resources are available...")
        all_found_locally = True
        # NLTK_DATA (e.g., /usr/local/share/nltk_data) should be in nltk.data.path
        # If it's not, something is fundamentally wrong with env var propagation or NLTK setup.
        # The Dockerfile sets NLTK_DATA.

        for dataset in required_nltk_datasets:
            try:
                # Construct the resource name NLTK uses for find()
                # e.g., 'tokenizers/punkt' or 'corpora/wordnet'
                resource_path_in_nltk = ""
                if dataset == 'punkt':
                     resource_path_in_nltk = f'tokenizers/{dataset}'
                elif dataset == 'stopwords':
                     resource_path_in_nltk = f'corpora/{dataset}'
                elif dataset == 'wordnet':
                     resource_path_in_nltk = f'corpora/{dataset}' # WordNet is expected as corpora/wordnet
                elif dataset == 'omw-1.4':
                     resource_path_in_nltk = f'corpora/{dataset}' # OMW is expected as corpora/omw-1.4

                nltk.data.find(resource_path_in_nltk)
                logger.debug(f"NLTK resource '{dataset}' (as '{resource_path_in_nltk}') found.")
            except LookupError:
                all_found_locally = False
                # More detailed error message
                logger.critical(
                    f"CRITICAL: NLTK resource '{dataset}' (searched as '{resource_path_in_nltk}') "
                    f"not found! Ensure Dockerfile download step succeeded and NLTK_DATA env var is correctly set and used. "
                    f"Current NLTK search paths: {nltk.data.path}"
                )

        if not all_found_locally:
             logger.error("One or more essential NLTK datasets are missing. Check Docker build logs and NLTK data paths configuration.")
             return False

        logger.info("All required NLTK resources verified successfully.")

        # Load resources
        if not _nltk_en_stopwords: _nltk_en_stopwords = set(nltk_stopwords.words('english'))
        if not _nltk_fr_stopwords: _nltk_fr_stopwords = set(nltk_stopwords.words('french'))
        if not _nltk_ar_stopwords: _nltk_ar_stopwords = set(nltk_stopwords.words('arabic'))
        logger.info(f"Loaded NLTK stopwords: EN({len(_nltk_en_stopwords or [])}), FR({len(_nltk_fr_stopwords or [])}), AR({len(_nltk_ar_stopwords or [])})")

        # WordNetLemmatizer requires 'wordnet' and 'omw-1.4' (Open Multilingual Wordnet)
        if not _lemmatizer_en: _lemmatizer_en = WordNetLemmatizer()
        logger.info("Initialized NLTK WordNetLemmatizer.")
        return True

    except Exception as e:
        logger.error(f"Unexpected error during NLTK resource loading/verification: {e}", exc_info=True)
        return False

def _load_spacy_models():
    """Loads spaCy models."""
    global _nlp_spacy_en, _nlp_spacy_fr
    if _nlp_spacy_en and _nlp_spacy_fr: return True
    logger.info("Loading spaCy models...")
    models_loaded = True
    try:
        if not _nlp_spacy_en:
            _nlp_spacy_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("Loaded spaCy English model (en_core_web_sm)")
    except OSError:
        logger.critical("Failed to load spaCy English model 'en_core_web_sm'. Ensure it was downloaded during build.")
        models_loaded = False
    try:
        if not _nlp_spacy_fr:
            _nlp_spacy_fr = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
            logger.info("Loaded spaCy French model (fr_core_news_sm)")
    except OSError:
        logger.critical("Failed to load spaCy French model 'fr_core_news_sm'. Ensure it was downloaded during build.")
        models_loaded = False
    return models_loaded

def _load_stanza_models():
    """Loads the Stanza Arabic pipeline."""
    global _nlp_stanza_ar
    if not STANZA_AVAILABLE:
        logger.warning("Stanza library not installed. Arabic NLP processing will use basic NLTK.")
        return False
    if _nlp_stanza_ar: return True

    logger.info("Loading Stanza Arabic model...")
    try:
        # STANZA_RESOURCES_DIR is set as an ENV var in Dockerfile, Stanza should pick it up.
        # If not, you can explicitly pass model_dir=STANZA_RESOURCES_DIR
        _nlp_stanza_ar = stanza.Pipeline('ar', dir=STANZA_RESOURCES_DIR, processors='tokenize,lemma,mwt', use_gpu=False)
        logger.info(f"Loaded Stanza Arabic pipeline using dir: {STANZA_RESOURCES_DIR}.")
        return True
    except Exception as e:
        logger.critical(f"Failed to load Stanza Arabic model (dir: {STANZA_RESOURCES_DIR}): {e}", exc_info=True)
        logger.error("Ensure the model was downloaded correctly in the Dockerfile step and STANZA_RESOURCES_DIR env var is correct.")
        return False

def ensure_nlp_resources():
    """Loads all necessary NLP resources if not already loaded."""
    global _resources_loaded
    if _resources_loaded:
        return True
    logger.info("Ensuring all NLP resources are loaded...")
    # Explicitly log the NLTK paths being used by the application *before* loading attempts
    logger.info(f"NLTK effective search paths at ensure_nlp_resources call: {nltk.data.path}")

    nltk_ok = _load_nltk_resources()
    spacy_ok = _load_spacy_models()
    stanza_ok = _load_stanza_models()

    _resources_loaded = nltk_ok and spacy_ok and (stanza_ok if STANZA_AVAILABLE else True) # Stanza is optional if not installed
    if not STANZA_AVAILABLE and not stanza_ok: # Only critical if Stanza IS available but failed to load
         _resources_loaded = False

    if not _resources_loaded:
         logger.critical(f"One or more essential NLP resources failed to load: NLTK_OK={nltk_ok}, Spacy_OK={spacy_ok}, Stanza_OK={stanza_ok (STANZA_AVAILABLE=STANZA_AVAILABLE)}")
         logger.critical("NLP processing capabilities will be limited or unavailable.")
    else:
         logger.info("All required NLP resources loaded successfully.")
    return _resources_loaded


def detect_language(text: str) -> Optional[str]:
    if not isinstance(text, str) or text.isspace() or len(text) < 10:
        return None
    try:
        lang = detect(text[:500])
        return lang
    except LangDetectException:
        logger.debug(f"Language detection failed (langdetect exception) for text snippet: {text[:100]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during language detection: {e}", exc_info=True)
        return None

def process_text_nlp(text: str, lang: str) -> Dict[str, List[str]]:
    if not _resources_loaded:
        logger.error(f"Skipping NLP processing for lang '{lang}' due to failed resource loading.")
        return {"tokens": [], "tokens_processed": [], "lemmas": []}
    if not text or not lang:
        return {"tokens": [], "tokens_processed": [], "lemmas": []}

    tokens: List[str] = []
    tokens_processed: List[str] = []
    lemmas: List[str] = []
    logger.trace(f"Processing NLP for lang '{lang}' on text: {text[:100]}...")

    try:
        if lang == 'en' and _nlp_spacy_en and _nltk_en_stopwords and _lemmatizer_en:
            doc = _nlp_spacy_en(text.lower())
            tokens = [token.text for token in doc if token.is_alpha]
            tokens_processed = [token.text for token in doc if token.is_alpha and not token.is_stop]
            lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        elif lang == 'fr' and _nlp_spacy_fr and _nltk_fr_stopwords:
            doc = _nlp_spacy_fr(text.lower())
            tokens = [token.text for token in doc if token.is_alpha]
            tokens_processed = [token.text for token in doc if token.is_alpha and not token.is_stop]
            lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        elif lang == 'ar' and STANZA_AVAILABLE and _nlp_stanza_ar and _nltk_ar_stopwords:
            if not _nlp_stanza_ar: # Should be caught by _resources_loaded, but double check
                logger.error("Skipping Arabic NLP: Stanza model not available/loaded.")
                return {"tokens": [], "tokens_processed": [], "lemmas": []}
            doc = _nlp_stanza_ar(text)
            raw_tokens = []
            lemmas_with_stops = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    raw_tokens.append(word.text)
                    lemmas_with_stops.append(word.lemma)
            tokens = raw_tokens
            stop_words_ar = _nltk_ar_stopwords or set()
            processed_indices = {
                i for i, lemma in enumerate(lemmas_with_stops)
                if lemma is not None and lemma.lower() not in stop_words_ar
                   and tokens[i] is not None and tokens[i].isalpha() # Ensure token is alpha
            }
            lemmas = [lemmas_with_stops[i] for i in processed_indices if lemmas_with_stops[i] is not None]
            tokens_processed = [tokens[i] for i in processed_indices if tokens[i] is not None] # Ensure token exists for index
        else:
            if lang not in ['en', 'fr', 'ar']:
                logger.warning(f"Skipping NLP processing for unsupported language: '{lang}'.")
            else:
                 # This case implies a resource specific to the lang failed (e.g., _nlp_spacy_en is None)
                 logger.error(f"Skipping NLP processing for lang '{lang}' due to missing/failed model for this language.")
    except Exception as e:
        logger.error(f"Unexpected error during NLP processing pipeline for lang '{lang}': {e}", exc_info=True)
        return {"tokens": [], "tokens_processed": [], "lemmas": []}

    logger.trace(f"NLP results - Tokens: {len(tokens)}, Processed: {len(tokens_processed)}, Lemmas: {len(lemmas)}")
    return {"tokens": tokens, "tokens_processed": tokens_processed, "lemmas": lemmas}