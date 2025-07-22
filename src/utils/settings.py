from typing import Optional
from pydantic_settings import BaseSettings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class Settings(BaseSettings):

    azure_openai_api_key: Optional[str] = None 
    azure_openai_endpoint: Optional[str] = None
    llm_deployment_model: Optional[str] = None
    embedding_deployment_model: Optional[str] = None
    llm_api_version: Optional[str] = None
    embedding_api_version: Optional[str] = None