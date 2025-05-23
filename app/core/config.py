# services/data_preprocessor/app/core/config.py
import os
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict # For Pydantic V2
from pydantic import Field # For Pydantic V2
from typing import Optional

# Use Loguru if installed, otherwise fallback to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__) # Get logger for this module

class Settings(BaseSettings):
    """ Application settings for Data Preprocessor """
    SERVICE_NAME: str = "Minbar Data Preprocessor"
    LOG_LEVEL: str = Field(default="INFO", validation_alias='LOG_LEVEL')

    # MongoDB Source Config
    mongo_uri: str = Field(..., validation_alias='MONGO_URI')
    mongo_db_name: str = Field(..., validation_alias='MONGO_DB_NAME')
    mongo_raw_collection: str = Field(..., validation_alias='MONGO_RAW_COLLECTION')
    batch_size: int = Field(default=100, gt=0, validation_alias='BATCH_SIZE')
    processed_status_field: str = Field(default="processor_v1_processed_status", validation_alias='PROCESSED_STATUS_FIELD')

    # PostgreSQL Target Config (where this service writes its output)
    postgres_user: str = Field(..., validation_alias='POSTGRES_USER')
    postgres_password: str = Field(..., validation_alias='POSTGRES_PASSWORD')
    postgres_host: str = Field(..., validation_alias='POSTGRES_HOST')
    postgres_port: int = Field(default=5432, validation_alias='POSTGRES_PORT') # Changed to int for direct use
    postgres_db: str = Field(..., validation_alias='POSTGRES_DB')
    postgres_table: str = Field(default="processed_documents", validation_alias='POSTGRES_TABLE')

    # --- NEW: Configuration for the status field this service creates for the NLP Analyzer ---
    DOWNSTREAM_NLP_ANALYZER_STATUS_FIELD: str = Field(
        default="nlp_analyzer_v1_status", 
        validation_alias='NLP_ANALYZER_STATUS_FIELD_TO_CREATE' # .env variable name
    )
    # --- END NEW ---

    mark_as_processed_in_mongo: bool = Field(default=True, validation_alias='MARK_AS_PROCESSED_IN_MONGO')

    # Scheduler Config
    scheduler_interval_minutes: int = Field(default=60, gt=0, validation_alias='SCHEDULER_INTERVAL_MINUTES')

    # Derived PostgreSQL DSN for asyncpg
    @property
    def postgres_dsn_asyncpg(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

settings = Settings()

_log_level_to_use = settings.LOG_LEVEL.upper()
try: # For Loguru
    logger.remove() # Remove default handler if any was added by Loguru itself
    log_format = ( # Define format here if not globally defined
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(lambda msg: print(msg, end=""), format=log_format, level=_log_level_to_use)
    _is_loguru = True
except NameError: # Fallback to standard logging if Loguru not imported/found
    logging.basicConfig(
        level=_log_level_to_use,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True # Ensure this config takes effect even if basicConfig was called before
    )
    logger = logging.getLogger(__name__) # Re-get logger for this module after basicConfig
    _is_loguru = False

logging.getLogger("nltk").setLevel(logging.WARNING)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("langdetect").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING) # Often noisy with debug tasks
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING)
logging.getLogger("motor").setLevel(logging.INFO) # Motor can be verbose on DEBUG
logging.getLogger("asyncpg").setLevel(logging.WARNING) # Asyncpg connection logs

logger.info(f"Configuration loaded for {settings.SERVICE_NAME}. Log level: {_log_level_to_use}. Using {'Loguru' if _is_loguru else 'standard logging'}.")
logger.debug(f"MongoDB Source: {settings.mongo_db_name}/{settings.mongo_raw_collection}")
logger.debug(f"PostgreSQL Target: {settings.postgres_db}/{settings.postgres_table}")
logger.debug(f"Downstream NLP Analyzer status field to create: {settings.DOWNSTREAM_NLP_ANALYZER_STATUS_FIELD}")
logger.debug(f"Mark as processed in Mongo: {settings.mark_as_processed_in_mongo}")
logger.debug(f"Scheduler Interval: {settings.scheduler_interval_minutes} minutes")