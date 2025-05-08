# Dockerfile
# Generated based on request from 2025-05-07
# Dockerfile for minbar-services/data-preprocessor

# ---- Stage 1: Build ----
# Use a Python base image. 3.11-slim-bullseye is a good balance of size and features.
FROM python:3.11-slim-bullseye AS builder

# Set environment variables for the builder stage
ENV PYTHONUNBUFFERED=1 \
    # Do not use pip cache for downloads, keeps layers smaller
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Path for NLTK data. NLTK will be configured to find data here.
    NLTK_DATA=/usr/local/share/nltk_data \
    # Path for Stanza models, consistent with app/processing/nlp_tasks.py (STANZA_RESOURCES_DIR)
    STANZA_RESOURCES_DIR=/app/.stanza_resources

# Install system dependencies required for building Python packages
# build-essential for compiling some Python packages (e.g., dependencies of uvicorn[standard])
# libxml2-dev, libxslt1-dev for lxml
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
# This layer is cached if requirements.txt doesn't change.
# PIP_NO_CACHE_DIR=on is set, so pip install won't use its http cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download NLP models and resources
# NLTK: Use a script for robust downloading and verification
COPY download_nltk_data.py /tmp/download_nltk_data.py
RUN python /tmp/download_nltk_data.py ${NLTK_DATA}

# spaCy: Models used in app/processing/nlp_tasks.py (en_core_web_sm, fr_core_news_sm)
# These will be installed into the Python site-packages directory.
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm

# Stanza: Arabic model, downloaded to STANZA_RESOURCES_DIR
# The app/processing/nlp_tasks.py expects models in this directory.
RUN mkdir -p ${STANZA_RESOURCES_DIR} && \
    python -c "import stanza; stanza.download('ar', model_dir='${STANZA_RESOURCES_DIR}', processors='tokenize,lemma,mwt', logging_level='INFO', verbose=True)"


# ---- Stage 2: Final image ----
# Start from a clean Python slim image for the final stage
FROM python:3.11-slim-bullseye AS final

# Set environment variables for the final image
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    # Path where NLTK models are located, consistent with builder stage and app expectations
    NLTK_DATA=/usr/local/share/nltk_data \
    # Path where Stanza models are located, consistent with app/processing/nlp_tasks.py and builder stage
    STANZA_RESOURCES_DIR=/app/.stanza_resources

# Create a non-root user and group for security
# The user's home directory will be APP_HOME. No login shell is assigned.
RUN groupadd --system appuser && \
    useradd --system --gid appuser -d ${APP_HOME} -s /sbin/nologin appuser

# Install runtime system dependencies
# lxml wheels often statically link libxml2/libxslt, but including libxml2 runtime is safer for compatibility.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2 \
    # Add any other essential runtime libraries here if needed
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${APP_HOME}

# Copy installed Python packages (including spaCy models linked into site-packages)
# and Python executables (like uvicorn, fastapi) from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy NLTK resources from the builder stage
COPY --from=builder ${NLTK_DATA} ${NLTK_DATA}

# Copy Stanza resources from the builder stage
# These are located at /app/.stanza_resources, which is inside APP_HOME
COPY --from=builder ${STANZA_RESOURCES_DIR} ${STANZA_RESOURCES_DIR}

# Copy application code from the build context (repository root) into APP_HOME/app directory
# This assumes your application code (main.py, core/, db/, etc.) is within an 'app' subfolder.
COPY --chown=appuser:appuser app/ ${APP_HOME}/app/

# Ensure APP_HOME and its contents (including STANZA_RESOURCES_DIR as it's /app/.stanza_resources)
# are owned by appuser. This allows the app to read its code and Stanza models.
# NLTK_DATA and site-packages are system paths, typically root-owned but should be world-readable.
RUN chown -R appuser:appuser ${APP_HOME}

# Switch to the non-root user
USER appuser

# Expose the port the application listens on
# This should match the port defined in your cloudbuild.yaml (--port=8002) and the Uvicorn command.
EXPOSE 8002

# Command to run the application
# This tells Uvicorn to run the FastAPI application instance named `app`
# located in the `app.main` module (i.e., app/main.py).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]