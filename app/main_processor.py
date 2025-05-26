# services/data_preprocessor/app/main_processor.py
import asyncio
import time
from datetime import datetime # Ensure datetime is imported for timestamping
from typing import Dict, Any, List, Optional
from bson import ObjectId

# Use shared logger and settings
from app.core.config import settings, logger

# Import DB functions (now async)
from app.db.mongo_db import fetch_unprocessed_documents, mark_documents_as_processed, mongo_reader
from app.db.postgres_db import insert_processed_data_batch
# Import processing functions
from app.processing.cleaning import basic_text_clean
from app.processing.nlp_tasks import detect_language, process_text_nlp, ensure_nlp_resources
# Import model and utilities
from app.models.data_models import ProcessedDocument
import dateutil.parser
# import dateutil.tz # Not explicitly used, can be removed if not needed elsewhere

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
    ts_str = original_data.get("created_time") # Assuming 'created_time' is the field from Data365
    metadata["original_url"] = original_data.get("attached_link") # Example, adjust if different

    if ts_str:
        try:
            # Ensure created_time is parsed correctly. Data365 might provide Unix timestamp or ISO string.
            # If it's a Unix timestamp (integer/float):
            # metadata["original_timestamp"] = datetime.fromtimestamp(ts_str, tz=timezone.utc)
            # If it's an ISO string (as assumed by dateutil.parser.isoparse):
            metadata["original_timestamp"] = dateutil.parser.isoparse(ts_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse timestamp '{ts_str}' for doc {metadata['raw_mongo_id']}: {e}")

    if not isinstance(metadata["text"], str):
         metadata["text"] = "" # Ensure text is always a string

    logger.trace(f"Extracted metadata for {metadata['raw_mongo_id']}")
    return metadata

# --- Core Processing Function (now separate) ---
async def process_single_document(raw_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Processes a single raw document. Returns dict for insertion or None on error."""
    doc_id = raw_doc.get("_id", "N/A") # Get ID for logging before potential failure
    try:
        logger.trace(f"Processing document: {doc_id}")
        extracted = extract_text_and_metadata(raw_doc)
        if not extracted:
            logger.warning(f"Extraction failed for document, skipping: {doc_id}")
            return None

        cleaned_text = basic_text_clean(extracted["text"])
        detected_lang = detect_language(cleaned_text) if cleaned_text else None # Only detect if there's text
        
        nlp_results = {"tokens": [], "tokens_processed": [], "lemmas": []}
        # Only run full NLP if text exists AND language is detected AND supported
        if cleaned_text and detected_lang and detected_lang in ['en', 'fr', 'ar']: # Check for supported lang here
            nlp_results = process_text_nlp(cleaned_text, detected_lang)
        elif cleaned_text and detected_lang: # Language detected but not supported for full NLP
             logger.debug(f"Language '{detected_lang}' detected for doc {doc_id} but not supported for full NLP. Basic processing only.")


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
            # The new 'nlp_analyzer_v1_status' field will be added by the DB schema default or ORM
        )
        logger.trace(f"Successfully processed document {doc_id}")
        return processed.model_dump() # Use model_dump() for Pydantic v2

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
    total_processed_in_job = 0
    total_inserted_in_job = 0
    total_marked_in_mongo_job = 0

    # Ensure NLP resources are loaded (idempotent check)
    if not ensure_nlp_resources(): # This is called at app startup but good to have a check here too
        logger.error("Cannot run processing job: Failed to load essential NLP resources.")
        return 

    try:
        current_run_processed_doc_count = 0 # Count docs processed in this specific trigger/run
        while True: # Loop to process all available data in batches
            batch_start_time = time.time()
            logger.debug(f"Fetching batch of up to {settings.batch_size} documents...")
            
            # Ensure mongo_reader and collection are available
            if mongo_reader.collection is None: # <<< CORRECTED CHECK
                logger.error("MongoDB collection is not available. Cannot fetch documents. Aborting job.")
                break # Exit the while loop if collection is None
            
            raw_docs = await fetch_unprocessed_documents(settings.batch_size)
            
            if not raw_docs:
                if current_run_processed_doc_count == 0: # First check in this run
                    logger.info("No unprocessed documents found in MongoDB for this run.")
                else: # Subsequent checks in this run
                    logger.info("Finished processing all available documents for this run.")
                break # Exit while loop if no documents are found

            current_run_processed_doc_count += len(raw_docs)
            fetch_duration = time.time() - batch_start_time
            logger.info(f"Fetched {len(raw_docs)} docs in {fetch_duration:.2f}s.")

            processed_batch_data: List[Dict[str, Any]] = []
            # --- Process documents in parallel ---
            process_tasks = [process_single_document(doc) for doc in raw_docs]
            results = await asyncio.gather(*process_tasks, return_exceptions=True)

            # Collect IDs of documents that were successfully processed and are candidates for PG insertion
            mongo_ids_for_pg_insert_and_marking: List[ObjectId] = []
            for i, result in enumerate(results):
                 raw_doc_id = raw_docs[i].get("_id") # Should always be an ObjectId if fetched
                 if isinstance(result, Exception):
                      logger.error(f"Error processing document {raw_doc_id}: {result}", exc_info=result)
                 elif result is not None: # Successfully processed
                      processed_batch_data.append(result)
                      if raw_doc_id and isinstance(raw_doc_id, ObjectId):
                           mongo_ids_for_pg_insert_and_marking.append(raw_doc_id)
            
            processing_end_time = time.time()
            processing_duration = processing_end_time - batch_start_time - fetch_duration
            logger.info(f"Processed {len(processed_batch_data)} valid docs from batch of {len(raw_docs)} in {processing_duration:.2f}s.")

            # --- Insert batch into PostgreSQL ---
            if processed_batch_data:
                insert_start_time = time.time()
                # Pass only the successfully processed data to insert_processed_data_batch
                inserted_count_batch = await insert_processed_data_batch(processed_batch_data)
                insert_duration = time.time() - insert_start_time
                
                if inserted_count_batch > 0:
                    logger.info(f"PostgreSQL insert attempt for {len(processed_batch_data)} docs resulted in {inserted_count_batch} actual inserts/updates, took {insert_duration:.2f}s.")
                    total_inserted_in_job += inserted_count_batch
                    
                    # --- MARK DOCUMENTS AS PROCESSED FOR THIS BATCH IN MONGO ---
                    # Only mark documents that were successfully prepared and sent for PG insertion attempt
                    if mongo_ids_for_pg_insert_and_marking:
                        mark_start_time_batch = time.time()
                        modified_mongo_count_batch = await mark_documents_as_processed(mongo_ids_for_pg_insert_and_marking)
                        mark_duration_batch = time.time() - mark_start_time_batch
                        logger.info(f"MongoDB marking for batch ({len(mongo_ids_for_pg_insert_and_marking)} docs) took {mark_duration_batch:.2f}s. Successfully marked: {modified_mongo_count_batch}")
                        total_marked_in_mongo_job += modified_mongo_count_batch
                    else:
                        logger.info("No document IDs from this batch were eligible for MongoDB marking.")
                else:
                    logger.warning(f"No documents were inserted into PostgreSQL for this batch of {len(processed_batch_data)} processed items. Not marking in MongoDB.")

                total_processed_in_job += len(processed_batch_data) # Count successfully processed items

            elif len(raw_docs) > 0 :
                 logger.warning(f"No valid documents resulted from processing this batch of {len(raw_docs)} fetched raw documents.")
            
            batch_duration = time.time() - batch_start_time
            logger.debug(f"Batch finished in {batch_duration:.2f}s.")
            
            # Optional: Add a small sleep if you want to be very cautious about hammering DBs,
            # but usually not needed if batch_size and scheduler interval are reasonable.
            # await asyncio.sleep(0.1) # e.g., 100ms delay between batches

    except ConnectionError as ce: # Catch connection errors from DB interactions
         logger.error(f"Database connection error during job: {ce}. Job will retry on next schedule if applicable.", exc_info=True)
    except Exception as e:
        # Catch any other unexpected error during the job
        logger.exception(f"Critical unexpected error during scheduled processing job: {e}")
    finally:
        job_duration = time.time() - job_start_time
        logger.info(f"--- Scheduled Processing Job Finished (or errored). Duration: {job_duration:.2f}s ---")
        logger.info(f"Job Summary - Total Docs Successfully Processed: {total_processed_in_job}, Total Docs Inserted/Updated in PG: {total_inserted_in_job}, Total Docs Marked in Mongo: {total_marked_in_mongo_job}")