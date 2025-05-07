# services/data_preprocessor/app/processing/nlp_tasks.py
from typing import Tuple, List, Optional, Dict, Any
# No os import needed now

# Language Detection
from langdetect import detect, DetectorFactory, LangDetectException
try:
    DetectorFactory.seed = 0 # For consistent results across runs
except NameError: pass # Handle if DetectorFactory doesn't have seed attribute

# NLTK (EN/FR/AR - Base/Fallback/Stopwords)
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as nltk_word_tokenize

# spaCy (EN/FR - Efficient)
import spacy

# Stanza (AR - Replaces Camel Tools)
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    stanza = None # Make sure stanza doesn't cause NameErrors later


from app.core.config import settings, logger # Use shared logger

# --- Global NLP Resources (Lazy Loaded) ---
_nlp_spacy_en: Optional[spacy.Language] = None
_nlp_spacy_fr: Optional[spacy.Language] = None
_nlp_stanza_ar: Optional['stanza.Pipeline'] = None # Stanza pipeline for Arabic
_nltk_en_stopwords: Optional[set] = None
_nltk_fr_stopwords: Optional[set] = None
_nltk_ar_stopwords: Optional[set] = None
_lemmatizer_en: Optional[WordNetLemmatizer] = None
_resources_loaded = False
STANZA_RESOURCES_DIR = "/app/.stanza_resources" # Directory specified in Dockerfile

# --- Resource Loading Functions ---
def _load_nltk_resources():
    """Loads required NLTK data, relying on build download."""
    global _nltk_en_stopwords, _nltk_fr_stopwords, _nltk_ar_stopwords, _lemmatizer_en

    # *** REMOVED nltk.data.path manipulation - rely on system default paths ***
    # logger.debug(f"NLTK search paths: {nltk.data.path}") # Keep for debugging if needed

    required_nltk_datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    try:
        logger.info("Verifying required NLTK resources are available...")
        all_found_locally = True
        for dataset in required_nltk_datasets:
            try:
                # Determine the correct find path based on NLTK conventions
                if dataset == 'stopwords': nltk.data.find(f'corpora/{dataset}')
                elif dataset in ['punkt', 'punkt_tab']: nltk.data.find(f'tokenizers/{dataset}')
                elif dataset == 'wordnet': nltk.data.find(f'corpora/{dataset}')
                elif dataset == 'omw-1.4': nltk.data.find(f'corpora/{dataset}')
            except LookupError:
                all_found_locally = False
                # Log critical error if not found, as build should guarantee it
                logger.critical(f"CRITICAL: NLTK resource '{dataset}' not found! Ensure Dockerfile download step succeeded and verified.")
                # Exit immediately if essential resource is missing
                return False

        if not all_found_locally:
             # This part should ideally not be reached if the checks above work
             logger.error("One or more essential NLTK datasets are missing. Check Docker build logs.")
             return False

        logger.info("All required NLTK resources verified successfully.")

        # Load resources
        if not _nltk_en_stopwords: _nltk_en_stopwords = set(nltk_stopwords.words('english'))
        if not _nltk_fr_stopwords: _nltk_fr_stopwords = set(nltk_stopwords.words('french'))
        if not _nltk_ar_stopwords: _nltk_ar_stopwords = set(nltk_stopwords.words('arabic'))
        logger.info(f"Loaded NLTK stopwords: EN({len(_nltk_en_stopwords or [])}), FR({len(_nltk_fr_stopwords or [])}), AR({len(_nltk_ar_stopwords or [])})")
        if not _lemmatizer_en: _lemmatizer_en = WordNetLemmatizer() # Requires wordnet
        logger.info("Initialized NLTK WordNetLemmatizer.")
        return True # Return True if loading succeeded

    except Exception as e:
        logger.error(f"Unexpected error during NLTK resource loading/verification: {e}", exc_info=True)
        return False

# --- _load_spacy_models and _load_stanza_models remain unchanged ---
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
    if _nlp_stanza_ar: return True # Already loaded

    logger.info("Loading Stanza Arabic model...")
    try:
        # Point to the directory where models were downloaded in Dockerfile
        _nlp_stanza_ar = stanza.Pipeline('ar', dir=STANZA_RESOURCES_DIR, processors='tokenize,lemma', use_gpu=False)
        logger.info(f"Loaded Stanza Arabic pipeline from {STANZA_RESOURCES_DIR}.")
        return True
    except Exception as e:
        # Log as critical since Stanza is the primary AR processor
        logger.critical(f"Failed to load Stanza Arabic model from {STANZA_RESOURCES_DIR}: {e}", exc_info=True)
        logger.error("Ensure the model was downloaded correctly in the Dockerfile step.")
        return False
# --- ensure_nlp_resources, detect_language, process_text_nlp remain unchanged from previous version ---
def ensure_nlp_resources():
    """Loads all necessary NLP resources if not already loaded."""
    global _resources_loaded
    if _resources_loaded:
        return True
    logger.info("Ensuring all NLP resources are loaded...")
    nltk_ok = _load_nltk_resources()
    spacy_ok = _load_spacy_models()
    stanza_ok = _load_stanza_models()

    # Resources are considered loaded only if ALL primary tools load successfully
    _resources_loaded = nltk_ok and spacy_ok and stanza_ok
    if not _resources_loaded:
         # Log detailed status
         logger.critical(f"One or more essential NLP resources failed to load: NLTK_OK={nltk_ok}, Spacy_OK={spacy_ok}, Stanza_OK={stanza_ok}")
         logger.critical("NLP processing capabilities will be limited or unavailable.")
    else:
         logger.info("All required NLP resources loaded successfully.")

    return _resources_loaded


# --- Core NLP Functions ---

def detect_language(text: str) -> Optional[str]:
    """Detects the language of the text using langdetect."""
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
    """
    Performs tokenization, stop-word removal, and lemmatization based on language.
    Returns a dictionary with 'tokens', 'tokens_processed', 'lemmas'.
    Assumes ensure_nlp_resources() has been called successfully.
    """
    # Check if resources loaded successfully before proceeding
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
        # --- English Processing (using spaCy) ---
        if lang == 'en' and _nlp_spacy_en and _nltk_en_stopwords and _lemmatizer_en:
            doc = _nlp_spacy_en(text.lower())
            tokens = [token.text for token in doc if token.is_alpha]
            tokens_processed = [token.text for token in doc if token.is_alpha and not token.is_stop]
            lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

        # --- French Processing (using spaCy) ---
        elif lang == 'fr' and _nlp_spacy_fr and _nltk_fr_stopwords:
            doc = _nlp_spacy_fr(text.lower())
            tokens = [token.text for token in doc if token.is_alpha]
            tokens_processed = [token.text for token in doc if token.is_alpha and not token.is_stop]
            lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

        # --- Arabic Processing (using Stanza) ---
        elif lang == 'ar' and STANZA_AVAILABLE and _nlp_stanza_ar and _nltk_ar_stopwords:
            # Check if Stanza model loaded successfully
            if not _nlp_stanza_ar:
                logger.error("Skipping Arabic NLP: Stanza model failed to load.")
                return {"tokens": [], "tokens_processed": [], "lemmas": []}
            try:
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
                       and tokens[i] is not None and tokens[i].isalpha()
                }
                lemmas = [lemmas_with_stops[i] for i in processed_indices if lemmas_with_stops[i] is not None]
                tokens_processed = [tokens[i] for i in processed_indices]
                logger.trace(f"Stanza processing completed for Arabic.")

            except Exception as stanza_err:
                logger.error(f"Error during Stanza processing for Arabic: {stanza_err}", exc_info=True)
                tokens, tokens_processed, lemmas = [], [], []

        # --- Handle Unsupported Languages Explicitly ---
        else:
            # If language is not one of the specifically handled ones (en, fr, ar)
            # Or if the required model for a supported language failed to load
            if lang not in ['en', 'fr', 'ar']:
                logger.warning(f"Skipping NLP processing for unsupported language: '{lang}'. Text snippet: {text[:100]}...")
            else:
                 logger.error(f"Skipping NLP processing for lang '{lang}' due to missing/failed model (_nlp_spacy_en={bool(_nlp_spacy_en)}, _nlp_spacy_fr={bool(_nlp_spacy_fr)}, _nlp_stanza_ar={bool(_nlp_stanza_ar)}).")
            # Return empty lists as no processing is done
            tokens, tokens_processed, lemmas = [], [], []

    except Exception as e:
        # Catch any unexpected errors during the processing for supported languages
        logger.error(f"Unexpected error during NLP processing pipeline for lang '{lang}': {e}", exc_info=True)
        # Ensure lists are empty on any top-level error
        return {"tokens": [], "tokens_processed": [], "lemmas": []}

    logger.trace(f"NLP results - Tokens: {len(tokens)}, Processed: {len(tokens_processed)}, Lemmas: {len(lemmas)}")
    return {
        "tokens": tokens,
        "tokens_processed": tokens_processed,
        "lemmas": lemmas
    }