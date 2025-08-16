from langchain_groq import ChatGroq
from src.utils import Settings


class GroqCloudService:
    def __init__(self, 
                 sets: Settings,
                 llm_model:str):
        self.llm_model = llm_model
        self.GROQ_API_KEY = sets.groq_api_key

    def get_llm(self, temperature: float = 0,
                      max_tokens: int = 1000) -> ChatGroq:
        
        return ChatGroq(
            model=self.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )