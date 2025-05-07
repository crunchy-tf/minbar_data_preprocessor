# services/data_preprocessor/app/core/config.py
import os
import logging
from pydantic_settings import BaseSettings
from pydantic import Field, MongoDsn
from typing import Optional

# Use Loguru if installed, otherwise fallback to standard logging
try:
    from loguru import logger
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    # Keep console logging at INFO by default unless overridden by LOG_LEVEL
    logger.add(lambda msg: print(msg, end=""), format=log_format, level="INFO")
except ImportError:
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Loguru not found, using standard logging.")


class Settings(BaseSettings):
    """ Application settings for Data Preprocessor """
    SERVICE_NAME: str = "Minbar Data Preprocessor"
    LOG_LEVEL: str = Field("INFO", validation_alias='LOG_LEVEL')

    # MongoDB Source Config
    mongo_uri: MongoDsn = Field(..., validation_alias='MONGO_URI')
    mongo_db_name: str = Field(..., validation_alias='MONGO_DB_NAME')
    mongo_raw_collection: str = Field(..., validation_alias='MONGO_RAW_COLLECTION')
    batch_size: int = Field(100, gt=0, validation_alias='BATCH_SIZE')
    processed_status_field: str = Field("processor_v1_processed_status", validation_alias='PROCESSED_STATUS_FIELD')

    # PostgreSQL Target Config
    postgres_user: str = Field(..., validation_alias='POSTGRES_USER')
    postgres_password: str = Field(..., validation_alias='POSTGRES_PASSWORD')
    postgres_host: str = Field(..., validation_alias='POSTGRES_HOST')
    postgres_port: str = Field(..., validation_alias='POSTGRES_PORT') # Keep as string or int
    postgres_db: str = Field(..., validation_alias='POSTGRES_DB')
    postgres_table: str = Field("processed_documents", validation_alias='POSTGRES_TABLE')

    # NLP Config - Removed camel_tools specific setting
    # camel_tools_mle_disambiguator: str = Field("calima-msa-r13", validation_alias='CAMEL_TOOLS_MLE_DISAMBIGUATOR')
    mark_as_processed_in_mongo: bool = Field(True, validation_alias='MARK_AS_PROCESSED_IN_MONGO')

    # Scheduler Config
    scheduler_interval_minutes: int = Field(60, gt=0, validation_alias='SCHEDULER_INTERVAL_MINUTES')

    # Derived PostgreSQL DSN for asyncpg
    @property
    def postgres_dsn_asyncpg(self) -> str:
        # asyncpg uses a DSN format like postgresql://user:password@host:port/database
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'


settings = Settings()

# Update logging level based on settings
try: # For Loguru
    logger.remove() # Remove default INFO handler
    logger.add(lambda msg: print(msg, end=""), format=log_format, level=settings.LOG_LEVEL.upper()) # Add handler with level from settings
except NameError: # For standard logging
    logging.getLogger().setLevel(settings.LOG_LEVEL.upper())

# Suppress overly verbose logs from dependencies
# logging.getLogger("camel_tools").setLevel(logging.WARNING) # Removed
logging.getLogger("nltk").setLevel(logging.WARNING)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("langdetect").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING) # <-- ADDED

logger.info(f"Configuration loaded for {settings.SERVICE_NAME}. Log level: {settings.LOG_LEVEL.upper()}")
logger.debug(f"MongoDB Target: {settings.mongo_db_name}/{settings.mongo_raw_collection}")
logger.debug(f"PostgreSQL Target: {settings.postgres_db}/{settings.postgres_table}")
logger.debug(f"Mark as processed in Mongo: {settings.mark_as_processed_in_mongo}")
logger.debug(f"Scheduler Interval: {settings.scheduler_interval_minutes} minutes")