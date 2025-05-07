# Minbar - Data Preprocessor Service

This microservice acts as an automated cleaning and organizing crew for raw text data (e.g., from Facebook posts). It reads raw data, performs essential text preprocessing using NLP techniques, and stores the structured results in a PostgreSQL database (the `Processed Data Lake`). It now runs periodically using an internal scheduler.

## Functionality

1.  **Starts Up:** Initializes a minimal FastAPI web server.
2.  **Connects:** Establishes connections to the source MongoDB and target PostgreSQL databases using asynchronous drivers (`motor`, `asyncpg`) and creates connection pools.
3.  **Checks/Creates DB Table:** Ensures the target PostgreSQL table (`processed_documents` by default) and necessary indexes exist within the specified database.
4.  **Loads NLP Resources:** Pre-loads necessary NLP models (spaCy models for EN/FR, NLTK data, Stanza for AR) for efficient processing. Resource loading happens during container startup.
5.  **Starts Scheduler:** Initializes and starts an APScheduler instance.
6.  **Scheduled Job (Runs Periodically):**
    *   Fetches batches of unprocessed documents from MongoDB (identified by the `PROCESSED_STATUS_FIELD`).
    *   For each document:
        *   Extracts text and relevant metadata (`_id`, `source`, `keyword_concept_id`, `original_timestamp`, `retrieved_by_keyword`, `keyword_language`, `original_url`) based on the expected schema from `social_media_ingester`.
        *   Cleans the text (HTML, URLs, emails, mentions, hashtags, basic punctuation, normalizes whitespace).
        *   Detects the language of the cleaned text (using `langdetect`).
        *   Performs Language-Specific NLP (if text and supported language [EN, FR, AR] are valid):
            *   **Tokenization:** Splits text into words/tokens (using spaCy for EN/FR, Stanza for AR).
            *   **Stop-Word Removal:** Removes common, low-information words (using spaCy lists for EN/FR, NLTK list for AR).
            *   **Lemmatization:** Reduces words to their base/dictionary form (using spaCy for EN/FR, Stanza for AR).
        *   Structures the results using a Pydantic model (`ProcessedDocument`).
    *   Inserts the structured data in batches into PostgreSQL (`ON CONFLICT DO NOTHING` based on `raw_mongo_id`).
    *   Optionally marks the source documents in MongoDB as processed (controlled by `.env` setting `MARK_AS_PROCESSED_IN_MONGO`).
    *   Repeats fetching/processing batches until no unprocessed documents remain for that run.
7.  **Provides Health Check:** Offers a `/health` API endpoint to check the status of database connections and the scheduler.
8.  **Allows Manual Trigger:** Includes a `/trigger-processing` endpoint to run the processing job on demand.

## Setup (using Docker is recommended)

1.  **Prerequisites:**
    *   Docker and Docker Compose (or `docker compose`) installed.
    *   Running MongoDB instance (accessible via `MONGO_URI` in `.env`).
    *   Running PostgreSQL instance (accessible via connection details in `.env`).
    *   **Create the target PostgreSQL database** specified in `.env` (e.g., `minbar_processed_data`) manually beforehand. The service will create the necessary *table* within this database.
2.  **Configure Environment:**
    *   Create a `.env` file in the `services/data_preprocessor/` directory (if not already present).
    *   Fill in the necessary MongoDB and PostgreSQL connection details (`MONGO_URI`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`). **Ensure these match your external database setup.** Use `host.docker.internal` for `MONGO_URI` and `POSTGRES_HOST` to connect from the container to your host machine.
    *   Set `SCHEDULER_INTERVAL_MINUTES` to the desired processing frequency (e.g., `60` for hourly).
    *   Set `MARK_AS_PROCESSED_IN_MONGO` (`True`/`False`). Set to `False` for testing runs that shouldn't modify the source data.
    *   Verify `PROCESSED_STATUS_FIELD` (e.g., `processor_v1_processed_status`).
    *   Verify `MONGO_RAW_COLLECTION` (e.g., `facebook_posts`).
    *   Verify `POSTGRES_TABLE` (e.g., `processed_documents`).
3.  **NLP Data:**
    *   The `Dockerfile` handles downloading the required NLTK data (`punkt`, `stopwords`, `wordnet`, `omw-1.4`, `punkt_tab`), spaCy models (`en_core_web_sm`, `fr_core_news_sm`), and the Stanza Arabic model during the `docker compose build` process. No manual download commands are typically needed when using Docker.

## Running the Service (with Docker Compose)

1.  **Build the Image:**
    ```bash
    # From the Minbar/ root directory:
    docker compose build data-preprocessor
    ```
2.  **Run the Container:**
    ```bash
    # From the Minbar/ root directory:
    docker compose up data-preprocessor
    # Add -d to run in detached mode: docker compose up -d data-preprocessor
    ```
    *   The service will be accessible on `http://localhost:8002`.

## API Endpoints

*   **`GET /health`**: Checks DB connections and scheduler status. Returns 200 OK or 503 Service Unavailable.
*   **`POST /trigger-processing`**: Manually starts one background processing run. Returns 202 Accepted.