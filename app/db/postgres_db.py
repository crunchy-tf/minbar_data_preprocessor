# services/data_preprocessor/app/db/postgres_db.py
import asyncpg # Use asyncpg for asyncio compatibility
from app.core.config import settings, logger
from typing import List, Dict, Any, Optional
import json
from bson import ObjectId

# --- Asyncpg Connection Pool ---
_pool: Optional[asyncpg.Pool] = None
_is_table_checked = False # Flag to indicate if create_table_if_not_exists has run successfully

async def connect_postgres():
    """Creates an asyncpg connection pool."""
    global _pool, _is_table_checked # Ensure _is_table_checked is global here
    if _pool and not _pool._closed: # Check if pool exists and is not closed
        logger.debug("Asyncpg connection pool already exists and is open.")
        # If pool exists but table wasn't checked (e.g., previous attempt failed before table check)
        if not _is_table_checked:
            logger.info("Pool exists, but table status unknown. Attempting table check/creation.")
            try:
                await create_table_if_not_exists()
            except Exception as table_err:
                logger.error(f"Failed to check/create table on existing pool: {table_err}")
                # Potentially close pool if table check is critical and fails, or let it be handled by health checks
        return
    
    if _pool and _pool._closed:
        logger.warning("Asyncpg connection pool exists but was closed. Recreating...")
        _pool = None # Reset to allow recreation
        _is_table_checked = False # Reset table check status

    logger.info("Creating asyncpg connection pool...")
    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.postgres_dsn_asyncpg,
            min_size=2,  # Minimum number of connections in the pool
            max_size=10, # Maximum number of connections
            # command_timeout=60 # Optional: if queries might take longer
        )
        logger.success(f"Asyncpg connection pool created for DB: {settings.postgres_db}")
        # Check/create table after pool is successfully created
        await create_table_if_not_exists()
    except Exception as e:
        logger.critical(f"Failed to create asyncpg connection pool: {e}", exc_info=True)
        _pool = None # Ensure pool is None on failure
        _is_table_checked = False # Ensure table check status is reset
        raise ConnectionError(f"Could not connect to PostgreSQL: {e}") from e

async def close_postgres():
    """Closes the asyncpg connection pool."""
    global _pool, _is_table_checked
    if _pool:
        logger.info("Closing asyncpg connection pool...")
        await _pool.close()
        _pool = None
        _is_table_checked = False # Reset table check status
        logger.success("Asyncpg connection pool closed.")
    else:
        logger.debug("Asyncpg connection pool already closed or not established.")

async def get_pool() -> asyncpg.Pool:
    """
    Returns the existing pool.
    Raises ConnectionError if the pool is not initialized, closed, or unusable.
    Attempts to re-initialize if pool is None (e.g., if startup failed).
    """
    global _pool # Ensure we're working with the global _pool

    if _pool is None:
        logger.warning("PostgreSQL pool is None. Attempting to (re)initialize...")
        try:
            await connect_postgres() # This will attempt to create the pool and check table
        except Exception as e: # Catch any exception from connect_postgres
            # connect_postgres logs critical error already
            raise ConnectionError(f"Failed to (re)initialize PostgreSQL pool: {e}") from e
        
        # After attempting to connect, check _pool again
        if _pool is None:
             logger.error("PostgreSQL pool is still None after (re)initialization attempt.")
             raise ConnectionError("PostgreSQL pool not initialized (initialization attempt failed).")

    # Check if pool was closed explicitly (e.g., by shutdown or error)
    if hasattr(_pool, '_closed') and _pool._closed:
        logger.error("PostgreSQL pool is marked as closed. Attempting to (re)initialize...")
        try:
            await connect_postgres() # This will try to recreate the pool
        except Exception as e:
            raise ConnectionError(f"Failed to (re)initialize closed PostgreSQL pool: {e}") from e
        
        if _pool is None or (hasattr(_pool, '_closed') and _pool._closed): # Check again
            logger.error("PostgreSQL pool remains closed or None after re-initialization attempt.")
            raise ConnectionError("PostgreSQL pool is closed (re-initialization attempt failed).")


    # Simple check to see if the pool is "acquirable"
    # This doesn't guarantee a connection from the pool will work, but that the pool object itself is usable.
    try:
        conn = await _pool.acquire()
        await _pool.release(conn)
    except Exception as e: # Could be `asyncpg.exceptions.PoolConnectionError` or others
        logger.error(f"Failed to acquire and release a connection from the pool: {e}")
        raise ConnectionError(f"PostgreSQL pool is unhealthy or unusable: {e}")

    return _pool

async def create_table_if_not_exists():
    """Creates the target table and indexes in PostgreSQL using asyncpg."""
    global _is_table_checked
    if _is_table_checked:
        logger.debug(f"Table '{settings.postgres_table}' already checked/created in this session.")
        return

    try:
        pool = await get_pool() # Ensures pool exists and is usable (will attempt reconnect if needed)
    except ConnectionError as e:
        logger.error(f"Cannot check/create table, PostgreSQL pool not available: {e}")
        _is_table_checked = False # Ensure this remains false
        # Re-raise to signal that a critical setup step failed
        raise ConnectionError(f"Failed to get PostgreSQL pool for table creation: {e}") from e


    logger.info(f"Checking/Creating PostgreSQL table '{settings.postgres_table}' and indexes...")
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
    create_indexes_sql = f"""
    CREATE INDEX IF NOT EXISTS idx_proc_timestamp ON {settings.postgres_table} (processing_timestamp);
    CREATE INDEX IF NOT EXISTS idx_proc_detected_lang ON {settings.postgres_table} (detected_language);
    CREATE INDEX IF NOT EXISTS idx_proc_source ON {settings.postgres_table} (source);
    CREATE INDEX IF NOT EXISTS idx_proc_keyword_concept ON {settings.postgres_table} (keyword_concept_id);
    CREATE INDEX IF NOT EXISTS idx_proc_orig_ts ON {settings.postgres_table} (original_timestamp);
    """

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(create_table_sql)
                await conn.execute(create_indexes_sql)
        logger.success(f"Table '{settings.postgres_table}' and indexes checked/created successfully.")
        _is_table_checked = True
    except Exception as e:
        logger.error(f"Error creating/checking table '{settings.postgres_table}' or indexes: {e}", exc_info=True)
        _is_table_checked = False # Explicitly set to false on error
        raise # Re-raise to indicate failure, transaction will roll back


async def insert_processed_data_batch(data: List[Dict[str, Any]]) -> int:
    """Inserts a batch of processed data into PostgreSQL using asyncpg."""
    if not data:
        logger.debug("No data provided for batch insert.")
        return 0

    try:
        pool = await get_pool() # Get connection pool (will attempt reconnect if needed)
    except ConnectionError as e:
        logger.error(f"Cannot insert data, PostgreSQL pool not available: {e}")
        return 0 # Indicate no records inserted due to pool issue
        
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
    try:
        async with pool.acquire() as conn:
            # Use a transaction for batch insert
            async with conn.transaction():
                # executemany is generally efficient for batch inserts
                status = await conn.executemany(insert_sql, batch_args)
                # `status` for `executemany` is like "INSERT 0 N" where N is number of rows affected by `DO NOTHING`.
                # We count attempted inserts.
                inserted_count = len(batch_args)
                logger.success(f"Asyncpg executemany completed. Status: {status}. Attempted to insert: {inserted_count}")
    except asyncpg.PostgresError as e: # Specific PG errors
        logger.error(f"Error during asyncpg batch insert (PostgresError): {e}", exc_info=True)
        inserted_count = 0 # Mark as failed if PG error occurs
    except Exception as e: # Other errors (e.g., from pool acquire if it fails after initial check)
        logger.error(f"Unexpected error during asyncpg batch insert: {e}", exc_info=True)
        inserted_count = 0 # Mark as failed

    return inserted_count