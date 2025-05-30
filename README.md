# Minbar Data Preprocessor

This microservice preprocesses raw text data fetched from MongoDB, performing cleaning and NLP tasks (language detection, tokenization, lemmatization). The processed data is then loaded into a PostgreSQL data lake.

## API Endpoints

---

### Status Endpoints

#### `GET /`
-   **Description**: Retrieves a welcome message indicating the service is running.
-   **Sample Request**:
    ```http
    GET /
    ```

#### `GET /health`
-   **Description**: Performs health checks on the service, including its connections to MongoDB and PostgreSQL, and the status of its internal scheduler.
-   **Sample Request**:
    ```http
    GET /health
    ```

---

### Actions

#### `POST /trigger-processing`
-   **Description**: Manually triggers one background data processing job. This job will:
    1.  Fetch unprocessed documents from the source MongoDB collection.
    2.  Clean and perform NLP processing on the text content of these documents.
    3.  Insert the processed data into the target PostgreSQL table.
    4.  Mark the documents as processed in MongoDB (if configured).
    The API call returns immediately with a 202 Accepted status, and the processing job runs in the background.
-   **Request Body**: None
-   **Sample Request**:
    ```http
    POST /trigger-processing
    ```

---