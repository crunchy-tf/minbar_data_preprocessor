# services/data_preprocessor/app/models/data_models.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Any
from datetime import datetime
from bson import ObjectId # Import ObjectId

class ProcessedDocument(BaseModel):
    """ Pydantic model representing the structure of data to be stored in PostgreSQL. """
    raw_mongo_id: str | ObjectId # Allow ObjectId during processing, convert before PG insert
    source: str # e.g., 'post', 'comment'
    keyword_concept_id: Optional[str] = None
    original_timestamp: Optional[datetime] = None
    retrieved_by_keyword: Optional[str] = None
    keyword_language: Optional[str] = None # 'en', 'fr', 'ar'
    detected_language: Optional[str] = None # 'en', 'fr', 'ar', etc.
    cleaned_text: Optional[str] = None
    tokens: Optional[List[str]] = Field(default_factory=list) # Ensure default is list
    tokens_processed: Optional[List[str]] = Field(default_factory=list) # Ensure default is list
    lemmas: Optional[List[str]] = Field(default_factory=list) # Ensure default is list
    original_url: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True # Allow ObjectId type
    )

    @field_validator('keyword_language', 'detected_language', mode='before')
    @classmethod
    def check_lang_code(cls, v: Optional[str]) -> Optional[str]:
        """ Basic validation/cleanup for language codes. """
        if v is not None:
            v = v.lower().strip()
            # Allow only 2-char codes or None after validation
            if len(v) != 2:
                 # Log the original value if it's unexpected before setting to None
                 # logger.warning(f"Invalid language code detected: '{v}'. Setting to None.")
                 return None
        return v