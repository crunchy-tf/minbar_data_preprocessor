# services/data_preprocessor/app/services/scheduler_service.py
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.core.config import settings, logger
# Import the main processing function that the scheduler will run
from app.main_processor import scheduled_processing_job # Corrected import path

scheduler = AsyncIOScheduler(timezone="UTC") # Use UTC or preferred timezone

# Flag to prevent scheduler starting multiple times in some environments
_scheduler_started = False

async def start_scheduler():
    """Adds the processing job and starts the scheduler."""
    global _scheduler_started
    if scheduler.running or _scheduler_started:
        logger.info("Scheduler already running or start initiated.")
        return

    try:
        job_interval_minutes = settings.scheduler_interval_minutes
        logger.info(f"Adding scheduled processing job with interval: {job_interval_minutes} minutes.")
        scheduler.add_job(
            scheduled_processing_job, # Function to run
            trigger=IntervalTrigger(minutes=job_interval_minutes, jitter=60), # Run periodically
            id="data_processing_cycle_job",
            name="Run Data Preprocessing Cycle",
            replace_existing=True,
            max_instances=1, # Ensure only one instance runs at a time
            misfire_grace_time=300 # Allow 5 minutes delay before misfire
        )
        scheduler.start()
        _scheduler_started = True
        logger.success(f"Scheduler started. Processing job scheduled every {job_interval_minutes} minutes.")
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}", exc_info=True)
        _scheduler_started = False # Ensure flag is reset on failure

async def stop_scheduler():
    """Stops the scheduler gracefully."""
    global _scheduler_started
    if scheduler.running:
        logger.info("Stopping scheduler...")
        try:
            scheduler.shutdown(wait=False) # Don't wait for current job to finish
            _scheduler_started = False
            logger.success("Scheduler stopped.")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}", exc_info=True)
    else:
        logger.info("Scheduler was not running.")