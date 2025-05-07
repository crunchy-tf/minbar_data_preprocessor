# services/data_preprocessor/app/main.py
from fastapi import FastAPI, HTTPException, status, Response
from contextlib import asynccontextmanager
import asyncio
import json # <--- THIS LINE WAS ADDED

# Use shared logger and settings
from app.core.config import settings, logger

# Import DB connection handlers (now async)
from app.db.mongo_db import connect_mongo, close_mongo, mongo_reader # Import reader instance
# Import get_pool to check PG status and _pool for the trigger check (if needed, though get_pool is better)
from app.db.postgres_db import connect_postgres, close_postgres, get_pool, _pool

# Import scheduler functions
from app.services.scheduler_service import start_scheduler, stop_scheduler, scheduler

# Import the job function for manual trigger
from app.main_processor import scheduled_processing_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info(f"Starting up {settings.SERVICE_NAME}...")

    # Load NLP resources first (can take time)
    try:
        from app.processing.nlp_tasks import ensure_nlp_resources
        ensure_nlp_resources()
    except Exception as nlp_err:
        logger.critical(f"Failed to load NLP resources during startup: {nlp_err}", exc_info=True)
        # Decide if you want to prevent startup if NLP fails
        # raise RuntimeError("NLP Resource Loading Failed") from nlp_err

    # Connect to databases
    db_connect_tasks = [connect_mongo(), connect_postgres()]
    results = await asyncio.gather(*db_connect_tasks, return_exceptions=True)

    mongo_conn_ok = not isinstance(results[0], Exception)
    pg_conn_ok = not isinstance(results[1], Exception)

    if not mongo_conn_ok: logger.critical(f"MongoDB connection failed: {results[0]}")
    if not pg_conn_ok: logger.critical(f"PostgreSQL connection failed: {results[1]}")

    # Proceed only if essential connections are up
    if mongo_conn_ok and pg_conn_ok:
        logger.info("Database connections established.")
        await start_scheduler() # Start scheduler after successful DB connections
    else:
        logger.critical("Essential database connections failed. Service will not start scheduler.")

    yield # Application runs here

    # --- Shutdown ---
    logger.info(f"Shutting down {settings.SERVICE_NAME}...")
    await stop_scheduler() # Stop scheduler first
    await close_postgres()
    await close_mongo()
    logger.info("Shutdown complete.")


# Create FastAPI app instance
app = FastAPI(
    title=settings.SERVICE_NAME,
    version="1.0.0",
    lifespan=lifespan,
    description="Microservice to preprocess raw text data and load it into a processed data lake."
)

# --- API Endpoints ---

@app.get("/", tags=["Status"], include_in_schema=False)
async def read_root():
    return {"message": f"{settings.SERVICE_NAME} is running."}

@app.get("/health", tags=["Health"], summary="Service Health Check")
async def health_check():
    """Performs health checks on database connections and scheduler."""
    db_mongo_status = "disconnected"
    db_postgres_status = "disconnected"
    scheduler_status = "stopped"
    overall_status = "error"
    http_status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    # Check MongoDB
    if mongo_reader.client:
        try:
            await mongo_reader.client.admin.command('ping')
            db_mongo_status = "connected"
        except Exception as e:
            db_mongo_status = f"error (ping failed: {e})"
    else:
        db_mongo_status = "error (no client)"

    # Check PostgreSQL Pool
    try:
        # Check if the pool object exists and is not closed
        pool = await get_pool() # Ensures pool is attempted to be initialized if None
        async with pool.acquire() as conn:
           await conn.execute("SELECT 1")
        db_postgres_status = "connected"
    except ConnectionError as ce: # Catch specific pool/connection errors
         logger.warning(f"Health check PG connection error: {ce}")
         db_postgres_status = "error (pool unavailable or connection failed)"
    except Exception as e:
        logger.error(f"Health check PG query failed: {e}", exc_info=True) # Log unexpected errors
        db_postgres_status = f"error (query failed: {e})"


    # Check Scheduler
    if scheduler and scheduler.running:
        scheduler_status = "running"

    # Determine overall status
    is_healthy = db_mongo_status == "connected" and db_postgres_status == "connected" and scheduler_status == "running"
    if is_healthy:
        overall_status = "ok"
        http_status_code = status.HTTP_200_OK

    return Response(
        status_code=http_status_code,
        content=json.dumps({ # Use json.dumps for proper JSON response
            "status": overall_status,
            "details": {
                "mongodb": db_mongo_status,
                "postgresql": db_postgres_status,
                "scheduler": scheduler_status
            }
        }),
        media_type="application/json"
    )


@app.post("/trigger-processing", tags=["Actions"], status_code=status.HTTP_202_ACCEPTED)
async def trigger_manual_processing():
    """Manually triggers one background data processing job."""
    logger.warning("Manual processing job triggered via API.")

    # Ensure the job exists before triggering
    job = scheduler.get_job("data_processing_cycle_job")
    if not job:
        logger.error("Processing job 'data_processing_cycle_job' not found in scheduler.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scheduled job not found.")

    # Improved Check: Ensure DB clients/pools seem available before triggering
    mongo_ok = bool(mongo_reader.client)
    pg_ok = False
    try:
        # Attempt to get the pool; might raise ConnectionError if init failed badly
        pool = await get_pool()
        # Quick check if pool object exists and seems usable (might not catch all edge cases)
        pg_ok = bool(pool and not pool.is_closing())
    except ConnectionError:
        pg_ok = False # If get_pool fails, PG is not OK

    if not (mongo_ok and pg_ok):
         logger.error(f"Cannot trigger job: Database connections are not healthy (MongoOK: {mongo_ok}, PGOK: {pg_ok}).")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connections not ready.")


    try:
        # Run the job function directly in a background task
        asyncio.create_task(scheduled_processing_job())
        logger.info("Manual processing job initiated in the background.")
        return {"message": "Manual processing job initiated in the background."}
    except Exception as e:
         logger.error(f"Failed to initiate manual processing job: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start processing job.")