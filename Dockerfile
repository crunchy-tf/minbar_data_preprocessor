# FILE: services/data_preprocessor/Dockerfile (Optimized for Rebuild Speed)

# Stage 1: Base Image
FROM python:3.10-slim-bullseye AS base

# --- Environment Variables ---
# Set these early as they rarely change
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Define NLTK data path EARLY and use it consistently
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV STANZA_RESOURCES_DIR=/app/.stanza_resources
# NLTK will use the NLTK_DATA env var

# --- System Dependencies ---
# This layer is still prone to cache invalidation due to 'apt-get update',
# but we run it early. Accept that it might rerun on different days.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev && \
    # Create NLTK data directory early (permissions handled by root initially)
    mkdir -p $NLTK_DATA && \
    chmod -R 755 $NLTK_DATA && \
    # Clean up APT caches
    rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# Set working directory
WORKDIR /app
# Copy requirements first. This layer cache is invalidated only if requirements.txt changes.
COPY requirements.txt .
# Install Python dependencies. This layer is invalidated if requirements.txt changes
# OR if the preceding apt-get layer cache was missed. This is often the longest step.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Download Foundational NLP Resources (System-wide) ---
# These layers are invalidated only if the pip install layer above was rebuilt.
# Run as root to install to system paths defined by ENV vars.
# NLTK - Download using the NLTK_DATA env var. The -d flag is now redundant but harmless.
RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords wordnet omw-1.4 punkt_tab || \
    (echo "NLTK download command failed!" && exit 1) && \
    # Verification uses the standard NLTK mechanisms which should respect NLTK_DATA
    python -c "import nltk; print('Verifying NLTK resources...'); \
    nltk.data.find('tokenizers/punkt'); \
    nltk.data.find('corpora/stopwords'); \
    nltk.data.find('corpora/wordnet'); \
    nltk.data.find('corpora/omw-1.4'); \
    nltk.data.find('tokenizers/punkt_tab'); \
    print('NLTK resource verification successful.')" || \
    (echo "NLTK resource verification failed after download!" && exit 1)

# spaCy
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm

# --- Application Setup (User, Dirs, Stanza Model) ---
# Create user and necessary directories owned by the user *before* switching.
# Ensure appuser owns the Stanza dir. NLTK dir was created/populated by root.
RUN useradd -m appuser && \
    mkdir -p $STANZA_RESOURCES_DIR && \
    chown -R appuser:appuser $STANZA_RESOURCES_DIR && \
    chown -R appuser:appuser /app
    # Note: If appuser ever needs to *modify* NLTK data (unlikely), you'd chown $NLTK_DATA too. Read access is usually sufficient.

# Switch to the non-root user
USER appuser
# WORKDIR is already /app

# Download Stanza model *as appuser* to the designated directory.
# This layer is invalidated only if the user/dir setup layer above was rebuilt.
RUN python -c "import stanza; stanza.download('ar', model_dir='$STANZA_RESOURCES_DIR', verbose=True)" || \
    (echo "Stanza download command failed!" && exit 1)

# --- Application Code Copy ---
# This is the most frequently changing part. Copy it LAST.
# If only your app code changes, only this layer will rebuild.
COPY --chown=appuser:appuser ./app /app/app

# --- Expose Port ---
EXPOSE 8002

# --- Run Application ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]