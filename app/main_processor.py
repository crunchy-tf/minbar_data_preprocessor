# services/data_preprocessor/app/main_processor.py
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from bson import ObjectId

# Use shared logger and settings
from app.core.config import settings, logger

# Import DB functions (now async)
from app.db.mongo_db import fetch_unprocessed_documents, mark_documents_as_processed
from app.db.postgres_db import insert_processed_data_batch
# Import processing functions
from app.processing.cleaning import basic_text_clean
from app.processing.nlp_tasks import detect_language, process_text_nlp, ensure_nlp_resources
# Import model and utilities
from app.models.data_models import ProcessedDocument
import dateutil.parser
import dateutil.tz

# --- Helper function ---
def extract_text_and_metadata(raw_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts relevant text and metadata from the raw MongoDB document structure.
    Returns None if essential data (_id) is missing.
    """
    doc_id = raw_doc.get("_id")
    if not doc_id or not isinstance(doc_id, ObjectId):
        logger.warning(f"Skipping document with invalid or missing _id: {raw_doc.get('_id', 'MISSING')}")
        return None

    metadata = {
        "raw_mongo_id": doc_id,
        "source": raw_doc.get("data_type", "unknown"),
        "keyword_concept_id": raw_doc.get("keyword_concept_id"),
        "original_timestamp": None,
        "text": None,
        "retrieved_by_keyword": raw_doc.get("retrieved_by_keyword"),
        "keyword_language": raw_doc.get("keyword_language"),
        "original_url": None
    }
    original_data = raw_doc.get("original_post_data", {})
    if not isinstance(original_data, dict):
        logger.warning(f"original_post_data is not a dict for raw_mongo_id: {metadata['raw_mongo_id']}")
        original_data = {}

    metadata["text"] = original_data.get("text")
    ts_str = original_data.get("created_time")
    metadata["original_url"] = original_data.get("attached_link")

    if ts_str:
        try:
            metadata["original_timestamp"] = dateutil.parser.isoparse(ts_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse timestamp '{ts_str}' for doc {metadata['raw_mongo_id']}: {e}")

    if not isinstance(metadata["text"], str):
         metadata["text"] = ""

    logger.trace(f"Extracted metadata for {metadata['raw_mongo_id']}")
    return metadata

# --- Core Processing Function (now separate) ---
async def process_single_document(raw_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Processes a single raw document. Returns dict for insertion or None on error."""
    doc_id = raw_doc.get("_id", "N/A")
    try:
        logger.trace(f"Processing document: {doc_id}")
        extracted = extract_text_and_metadata(raw_doc)
        if not extracted: return None

        cleaned_text = basic_text_clean(extracted["text"])
        detected_lang = detect_language(cleaned_text) if cleaned_text else None
        nlp_results = {"tokens": [], "tokens_processed": [], "lemmas": []}
        if cleaned_text and detected_lang:
            nlp_results = process_text_nlp(cleaned_text, detected_lang)

        processed = ProcessedDocument(
            raw_mongo_id=extracted["raw_mongo_id"],
            source=extracted["source"],
            keyword_concept_id=extracted["keyword_concept_id"],
            original_timestamp=extracted["original_timestamp"],
            retrieved_by_keyword=extracted["retrieved_by_keyword"],
            keyword_language=extracted["keyword_language"],
            detected_language=detected_lang,
            cleaned_text=cleaned_text,
            tokens=nlp_results["tokens"],
            tokens_processed=nlp_results["tokens_processed"],
            lemmas=nlp_results["lemmas"],
            original_url=extracted["original_url"]
        )
        logger.trace(f"Successfully processed document {doc_id}")
        return processed.model_dump()

    except Exception as e:
        logger.error(f"Failed to process document {doc_id}: {e}", exc_info=True)
        return None

# --- Function Called by Scheduler ---
async def scheduled_processing_job():
    """
    The main logic for a single processing run, designed to be called by APScheduler.
    Ensures NLP resources are loaded and handles the fetch-process-store loop.
    Database connections are managed by the main application lifespan.
    """
    job_start_time = time.time()
    logger.info("--- Scheduled Processing Job Starting ---")
    processed_count_job = 0
    inserted_count_job = 0
    processed_ids_mongo_job = [] # Store ObjectIds of docs attempted insert

    # Ensure NLP resources are loaded (idempotent check)
    if not ensure_nlp_resources():
        logger.error("Cannot run processing job: Failed to load essential NLP resources.")
        return # Don't run if resources are missing

    try:
        total_docs_in_run = 0
        while True: # Loop to process all available data in batches
            batch_start_time = time.time()
            logger.debug(f"Fetching batch of up to {settings.batch_size} documents...")
            raw_docs = await fetch_unprocessed_documents(settings.batch_size)
            if not raw_docs:
                if total_docs_in_run == 0: logger.info("No unprocessed documents found in MongoDB for this run.")
                else: logger.info("Finished processing all available documents for this run.")
                break # Exit loop

            total_docs_in_run += len(raw_docs)
            fetch_duration = time.time() - batch_start_time
            logger.info(f"Fetched {len(raw_docs)} docs in {fetch_duration:.2f}s.")

            processed_batch_data: List[Dict[str, Any]] = []
            # --- Process documents in parallel ---
            process_tasks = [process_single_document(doc) for doc in raw_docs]
            results = await asyncio.gather(*process_tasks, return_exceptions=True)

            current_batch_ids_processed = [] # Track IDs successfully processed in this batch
            for i, result in enumerate(results):
                 raw_doc_id = raw_docs[i].get("_id")
                 if isinstance(result, Exception):
                      logger.error(f"Error processing document {raw_doc_id}: {result}")
                 elif result is not None:
                      processed_batch_data.append(result)
                      if raw_doc_id and isinstance(raw_doc_id, ObjectId):
                           current_batch_ids_processed.append(raw_doc_id)

            processing_duration = time.time() - batch_start_time - fetch_duration
            logger.info(f"Processed {len(processed_batch_data)} valid docs from batch in {processing_duration:.2f}s.")

            # --- Insert batch into PostgreSQL ---
            if processed_batch_data:
                insert_start_time = time.time()
                inserted_count_batch = await insert_processed_data_batch(processed_batch_data) # Now async
                insert_duration = time.time() - insert_start_time
                # We count successful processing attempts, insert_processed_data_batch returns attempt count
                inserted_count_job += inserted_count_batch
                processed_count_job += len(processed_batch_data)

                # Add IDs of docs *attempted* for insert to the list for MongoDB update
                # This assumes an insert attempt implies successful processing
                processed_ids_mongo_job.extend(current_batch_ids_processed)
                logger.info(f"PostgreSQL insert attempt for {inserted_count_batch} docs took {insert_duration:.2f}s.")
            elif len(raw_docs) > 0 :
                 # If docs were fetched but none were processed successfully
                 logger.warning(f"No valid documents processed in this batch ({len(raw_docs)} fetched) to insert.")
            else: # Should not happen if raw_docs check passed
                 logger.debug("No documents processed in this batch.")


            batch_duration = time.time() - batch_start_time
            logger.debug(f"Batch finished in {batch_duration:.2f}s.")

            # Safety break removed for production, scheduler interval controls frequency


    except ConnectionError as ce:
         logger.error(f"Database connection error during job: {ce}. Job will retry on next schedule.")
    except Exception as e:
        logger.exception(f"Critical unexpected error during scheduled processing job: {e}")
    finally:
        # Mark documents AFTER the loop finishes or if an error occurs mid-loop
        if processed_ids_mongo_job:
            mark_start_time = time.time()
            # Pass only unique IDs
            unique_ids_to_mark = list(set(processed_ids_mongo_job))
            modified_mongo_count = await mark_documents_as_processed(unique_ids_to_mark)
            mark_duration = time.time() - mark_start_time
            logger.info(f"MongoDB marking attempt for {len(unique_ids_to_mark)} unique docs took {mark_duration:.2f}s. Modified: {modified_mongo_count}")
        elif total_docs_in_run > 0 : # Log if processing happened but nothing to mark
             logger.info("No documents were successfully processed to mark in MongoDB.")

        job_duration = time.time() - job_start_time
        logger.info(f"--- Scheduled Processing Job Finished. Duration: {job_duration:.2f}s ---")
        logger.info(f"Job Summary - Docs Processed: {processed_count_job}, Insert Attempts: {inserted_count_job}, Marked in Mongo: {len(set(processed_ids_mongo_job)) if processed_ids_mongo_job else 0}")