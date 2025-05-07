# services/data_preprocessor/app/db/mongo_db.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from app.core.config import settings, logger # Use shared logger
from typing import List, Dict, Any, Optional
from bson import ObjectId
from datetime import datetime # Import datetime


class MongoReader:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None
    collection: AsyncIOMotorCollection | None = None

mongo_reader = MongoReader()

async def connect_mongo():
    """Establishes connection to the source MongoDB."""
    if mongo_reader.client:
        logger.debug("MongoDB connection already established.")
        return
    logger.info("Connecting to source MongoDB...")
    try:
        mongo_reader.client = AsyncIOMotorClient(str(settings.mongo_uri))
        mongo_reader.db = mongo_reader.client[settings.mongo_db_name]
        mongo_reader.collection = mongo_reader.db[settings.mongo_raw_collection]
        await mongo_reader.client.admin.command('ping') # Verify connection
        logger.success(f"Successfully connected to MongoDB: {settings.mongo_db_name}/{settings.mongo_raw_collection}")
    except Exception as e:
        logger.critical(f"Failed to connect to source MongoDB: {e}", exc_info=True)
        mongo_reader.client = None
        mongo_reader.db = None
        mongo_reader.collection = None
        raise ConnectionError("Could not connect to source MongoDB") from e

async def close_mongo():
    """Closes the source MongoDB connection."""
    if mongo_reader.client:
        logger.info("Closing source MongoDB connection...")
        mongo_reader.client.close()
        mongo_reader.client = None
        mongo_reader.db = None
        mongo_reader.collection = None
        logger.success("Source MongoDB connection closed.")
    else:
        logger.debug("MongoDB connection already closed or not established.")


async def fetch_unprocessed_documents(batch_size: int) -> List[Dict[str, Any]]:
    """Fetches a batch of documents that haven't been processed yet."""
    # --- CORRECTED CHECK ---
    if mongo_reader.collection is None:
        logger.error("MongoDB connection not established. Cannot fetch.")
        return []
    # --- END CORRECTION ---

    query = {
        settings.processed_status_field: {"$ne": True}
    }
    logger.debug(f"Fetching up to {batch_size} docs with query: {query}")
    try:
        cursor = mongo_reader.collection.find(query).limit(batch_size)
        documents = await cursor.to_list(length=batch_size)
        logger.info(f"Fetched {len(documents)} unprocessed documents from MongoDB.")
        return documents
    except Exception as e:
        logger.error(f"Error fetching unprocessed documents from MongoDB: {e}", exc_info=True)
        return []

async def mark_documents_as_processed(document_ids: List[ObjectId]) -> int:
    """Updates documents in MongoDB to mark them as processed."""
    # --- CORRECTED CHECK ---
    if mongo_reader.collection is None:
        logger.warning("Cannot mark documents as processed, MongoDB connection not available.")
        return 0
    # --- END CORRECTION ---

    if not settings.mark_as_processed_in_mongo:
        logger.debug("Skipping marking documents in MongoDB as per configuration.")
        return 0
    if not document_ids:
        logger.debug("No document IDs provided to mark as processed.")
        return 0

    logger.info(f"Attempting to mark {len(document_ids)} documents as processed in MongoDB with field '{settings.processed_status_field}'.")
    try:
        # Add timestamp along with the status field
        update_payload = {
            "$set": {
                settings.processed_status_field: True,
                f"{settings.processed_status_field}_timestamp": datetime.utcnow()
            }
        }
        result = await mongo_reader.collection.update_many(
            {"_id": {"$in": document_ids}},
            update_payload
        )
        logger.success(f"MongoDB update result: Matched={result.matched_count}, Modified={result.modified_count}")
        if result.matched_count != len(document_ids):
             logger.warning(f"Attempted to mark {len(document_ids)} docs, but only matched {result.matched_count}. Some IDs might be invalid or already marked.")
        return result.modified_count
    except Exception as e:
        logger.error(f"Error marking documents as processed in MongoDB: {e}", exc_info=True)
        return 0