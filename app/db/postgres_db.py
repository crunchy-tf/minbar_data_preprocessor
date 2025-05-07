# services/data_preprocessor/app/db/postgres_db.py
import asyncpg # Use asyncpg for asyncio compatibility
from app.core.config import settings, logger
from typing import List, Dict, Any, Optional
import json
from bson import ObjectId

# --- Asyncpg Connection Pool ---
_pool: Optional[asyncpg.Pool] = None
_is_table_checked = False

async def connect_postgres():
    """Creates an asyncpg connection pool."""
    global _pool
    if _pool:
        logger.debug("Asyncpg connection pool already exists.")
        return
    logger.info("Creating asyncpg connection pool...")
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn_asyncpg,
            min_size=2,  # Minimum number of connections in the pool
            max_size=10, # Maximum number of connections
            # Add command_timeout if needed
            # command_timeout=60
        )
        logger.success(f"Asyncpg connection pool created for DB: {settings.postgres_db}")
        # Check/create table after pool is created
        await create_table_if_not_exists()
    except Exception as e:
        logger.critical(f"Failed to create asyncpg connection pool: {e}", exc_info=True)
        _pool = None
        raise ConnectionError("Could not connect to PostgreSQL") from e

async def close_postgres():
    """Closes the asyncpg connection pool."""
    global _pool, _is_table_checked
    if _pool:
        logger.info("Closing asyncpg connection pool...")
        await _pool.close()
        _pool = None
        _is_table_checked = False
        logger.success("Asyncpg connection pool closed.")
    else:
        logger.debug("Asyncpg connection pool already closed or not established.")

async def get_pool() -> asyncpg.Pool:
    """Returns the existing pool, ensuring it's initialized."""
    if _pool is None:
        logger.warning("Attempted to get pool before it was initialized.")
        await connect_postgres() # Try to connect if not already
        if _pool is None: # Check again after attempting connection
             raise ConnectionError("PostgreSQL pool is not available.")
    return _pool

async def create_table_if_not_exists():
    """Creates the target table and indexes in PostgreSQL using asyncpg."""
    global _is_table_checked
    if _is_table_checked:
        return

    pool = await get_pool() # Ensure pool exists
    logger.info(f"Checking/Creating PostgreSQL table '{settings.postgres_table}' and indexes...")

    # Use triple quotes for multi-line SQL
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {settings.postgres_table} (
        id SERIAL PRIMARY KEY,
        raw_mongo_id VARCHAR(24) UNIQUE NOT NULL,
        source VARCHAR(50) NOT NULL,
        keyword_concept_id VARCHAR(24),
        original_timestamp TIMESTAMPTZ,
        processing_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        retrieved_by_keyword TEXT,
        keyword_language CHAR(2),
        detected_language CHAR(2),
        cleaned_text TEXT,
        tokens JSONB,
        tokens_processed JSONB,
        lemmas JSONB,
        original_url TEXT
    );
    """
    # Separate INDEX creation for better handling if table exists
    # Uses simplified CREATE INDEX IF NOT EXISTS (requires PostgreSQL 9.5+)
    create_indexes_sql = f"""
    CREATE INDEX IF NOT EXISTS idx_proc_timestamp ON {settings.postgres_table} (processing_timestamp);
    CREATE INDEX IF NOT EXISTS idx_proc_detected_lang ON {settings.postgres_table} (detected_language);
    CREATE INDEX IF NOT EXISTS idx_proc_source ON {settings.postgres_table} (source);
    CREATE INDEX IF NOT EXISTS idx_proc_keyword_concept ON {settings.postgres_table} (keyword_concept_id);
    CREATE INDEX IF NOT EXISTS idx_proc_orig_ts ON {settings.postgres_table} (original_timestamp);
    """

    async with pool.acquire() as conn:
        async with conn.transaction(): # Use a transaction
            try:
                await conn.execute(create_table_sql)
                await conn.execute(create_indexes_sql)
                logger.success(f"Table '{settings.postgres_table}' and indexes checked/created successfully.")
                _is_table_checked = True
            except Exception as e:
                logger.error(f"Error creating/checking table '{settings.postgres_table}' or indexes: {e}", exc_info=True)
                # Transaction automatically rolls back on exception with async with
                raise # Re-raise to indicate failure


async def insert_processed_data_batch(data: List[Dict[str, Any]]) -> int:
    """Inserts a batch of processed data into PostgreSQL using asyncpg."""
    if not data:
        logger.debug("No data provided for batch insert.")
        return 0

    pool = await get_pool() # Get connection pool
    inserted_count = 0
    skipped_count = 0
    batch_args = [] # List of tuples for executemany

    # Prepare data into tuples matching column order
    column_order = [
        'raw_mongo_id', 'source', 'keyword_concept_id', 'original_timestamp',
        'retrieved_by_keyword', 'keyword_language', 'detected_language',
        'cleaned_text', 'tokens', 'tokens_processed', 'lemmas', 'original_url'
    ]
    placeholders = ', '.join(f'${i+1}' for i in range(len(column_order)))
    insert_sql = f"""
    INSERT INTO {settings.postgres_table} (
        {', '.join(column_order)}
    ) VALUES ({placeholders})
    ON CONFLICT (raw_mongo_id) DO NOTHING;
    """

    for record in data:
        try:
            # Validate and prepare raw_mongo_id
            raw_id = record.get('raw_mongo_id')
            if isinstance(raw_id, ObjectId):
                mongo_id_str = str(raw_id)
            elif isinstance(raw_id, str) and len(raw_id) == 24:
                mongo_id_str = raw_id
            else:
                logger.warning(f"Record missing or has invalid raw_mongo_id, skipping: {raw_id}")
                skipped_count += 1
                continue

            # Prepare tuple in the correct order, serializing JSON
            record_tuple = (
                mongo_id_str,
                record.get('source', 'unknown'),
                record.get('keyword_concept_id'),
                record.get('original_timestamp'),
                record.get('retrieved_by_keyword'),
                record.get('keyword_language'),
                record.get('detected_language'),
                record.get('cleaned_text'),
                json.dumps(record.get('tokens')) if record.get('tokens') is not None else None,
                json.dumps(record.get('tokens_processed')) if record.get('tokens_processed') is not None else None,
                json.dumps(record.get('lemmas')) if record.get('lemmas') is not None else None,
                record.get('original_url')
            )
            batch_args.append(record_tuple)
        except Exception as prep_err:
            logger.error(f"Error preparing record for batch insert (ID: {record.get('raw_mongo_id', 'N/A')}): {prep_err}", exc_info=True)
            skipped_count += 1

    if not batch_args:
        logger.warning(f"No valid records remaining to insert after preparation (Skipped: {skipped_count}).")
        return 0

    logger.info(f"Attempting asyncpg batch insert of {len(batch_args)} records (Skipped: {skipped_count}).")
    async with pool.acquire() as conn:
        # Use a transaction for batch insert
        async with conn.transaction():
            try:
                # executemany is generally efficient for batch inserts
                status = await conn.executemany(insert_sql, batch_args)
                inserted_count = len(batch_args) # Count attempts
                logger.success(f"Asyncpg executemany completed. Status: {status}. Attempted: {inserted_count}")
            except Exception as e:
                logger.error(f"Error during asyncpg batch insert: {e}", exc_info=True)
                # Transaction rolls back automatically
                inserted_count = 0 # Mark as failed

    return inserted_count