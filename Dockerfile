# FILE: Dockerfile (Partial - NLTK Download Section Updated)

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

# NLTK - Download each resource individually for better error isolation
RUN echo "Downloading NLTK punkt..." && \
    python -m nltk.downloader -d $NLTK_DATA punkt || (echo "Failed to download NLTK punkt" && exit 1) && \
    echo "Downloading NLTK stopwords..." && \
    python -m nltk.downloader -d $NLTK_DATA stopwords || (echo "Failed to download NLTK stopwords" && exit 1) && \
    echo "Downloading NLTK wordnet..." && \
    python -m nltk.downloader -d $NLTK_DATA wordnet || (echo "Failed to download NLTK wordnet" && exit 1) && \
    echo "Downloading NLTK omw-1.4..." && \
    python -m nltk.downloader -d $NLTK_DATA omw-1.4 || (echo "Failed to download NLTK omw-1.4" && exit 1) && \
    echo "Downloading NLTK punkt_tab..." && \
    python -m nltk.downloader -d $NLTK_DATA punkt_tab || (echo "Failed to download NLTK punkt_tab" && exit 1) && \
    # Verification (keep this as it's a good check)
    echo "Verifying NLTK resources..." && \
    python -c "import nltk; \
    nltk.data.path.append('$NLTK_DATA'); \
    print(f'NLTK data paths: {nltk.data.path}'); \
    nltk.data.find('tokenizers/punkt'); print('punkt OK'); \
    nltk.data.find('corpora/stopwords'); print('stopwords OK'); \
    nltk.data.find('corpora/wordnet'); print('wordnet OK'); \
    nltk.data.find('corpora/omw-1.4'); print('omw-1.4 OK'); \
    nltk.data.find('tokenizers/punkt_tab'); print('punkt_tab OK'); \
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