import os
from typing import Optional
from pydantic_settings import BaseSettings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def find_env_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        env_file = os.path.join(current_dir, '.env')
        if os.path.exists(env_file):
            logger.info(f".env founded in: {env_file}")
            return env_file
        current_dir = os.path.dirname(current_dir)

    env_file = '.env'
    if os.path.exists(env_file):
        logger.info(f".env founded in: {os.path.abspath(env_file)}")
        return env_file

    logger.warning(".env not found!")
    return '.env'  # Fallback

class Settings(BaseSettings):
    
    azure_openai_api_key: Optional[str] = None 
    azure_openai_endpoint: Optional[str] = None
    llm_deployment_model: Optional[str] = None
    embedding_deployment_model: Optional[str] = None
    llm_api_version: Optional[str] = None
    embedding_api_version: Optional[str] = None
    groq_api_key: Optional[str] = None

    class Config:
        env_file = find_env_file()
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._debug_settings()
        
    
    def _debug_settings(self):
        logger.info("================================")
        for field_name, field_info in self.__fields__.items():
            value = getattr(self, field_name)
            if value is not None:
                logger.info(f"{field_name}: OK")
            else:
                logger.warning(f"{field_name}: not defined!")
        logger.info("================================")