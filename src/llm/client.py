"""LLM client module providing both legacy LLM clients and LangChain-compatible wrappers for use with the agentic workflow."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Union
import os
import google.generativeai as genai
from openai import OpenAI
from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
else:
    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError:
        BaseChatModel = Any

logger = get_logger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from the LLM. Args: prompt (str): User prompt. system_prompt (Optional[str]): System prompt. Returns: str: Generated text."""
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview") -> None:
        self.api_key = api_key or get_env_var("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro") -> None:
        self.api_key = api_key or get_env_var("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Gemini. Args: prompt (str): User prompt. system_prompt (Optional[str]): System prompt. Returns: str: Generated text."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"
            
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class DebugClient(LLMClient):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate mock text for debugging. Args: prompt (str): User prompt. system_prompt (Optional[str]): System prompt. Returns: str: Mock response."""
        logger.info("DebugClient: Generating mock response.")
        return "MOCK_RESPONSE"


def get_llm_client(provider: str = "openai") -> LLMClient:
    """Gets a legacy LLM client instance. Args: provider (str): Provider name ("openai", "gemini", or "debug"). Returns: LLMClient: LLMClient instance."""
    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "gemini":
        return GeminiClient()
    elif provider.lower() == "debug":
        return DebugClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_langchain_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """Get a LangChain-compatible LLM instance. Args: provider (str): Provider name. model (Optional[str]): Model name override. temperature (float): Generation temperature. **kwargs: Additional arguments. Returns: BaseChatModel: LangChain BaseChatModel instance."""
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return _get_langchain_openai(model, temperature, **kwargs)
    elif provider_lower == "gemini":
        return _get_langchain_gemini(model, temperature, **kwargs)
    else:
        raise ValueError(f"Unknown LangChain provider: {provider}. Supported: openai, gemini")


def _get_langchain_openai(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> "BaseChatModel":
    """Get a LangChain ChatOpenAI instance. Args: model (Optional[str]): Model name. temperature (float): Generation temperature. **kwargs: Additional arguments. Returns: BaseChatModel: ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for OpenAI LangChain support. "
            "Install with: pip install langchain-openai"
        )
    
    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    
    model_name = model or "gpt-4-turbo-preview"
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def _get_langchain_gemini(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> "BaseChatModel":
    """Get a LangChain ChatGoogleGenerativeAI instance. Args: model (Optional[str]): Model name. temperature (float): Generation temperature. **kwargs: Additional arguments. Returns: BaseChatModel: ChatGoogleGenerativeAI instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for Gemini LangChain support. "
            "Install with: pip install langchain-google-genai"
        )
    
    api_key = get_env_var("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    
    model_name = model or "gemini-1.5-pro"
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs
    )


def get_available_providers() -> List[str]:
    """Get list of available LLM providers. Returns: List[str]: Available provider names."""
    available = []
    
    try:
        from langchain_openai import ChatOpenAI
        if get_env_var("OPENAI_API_KEY"):
            available.append("openai")
    except ImportError:
        pass
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if get_env_var("GEMINI_API_KEY"):
            available.append("gemini")
    except ImportError:
        pass
    
    return available


def validate_provider(provider: str) -> bool:
    """Check if a provider is available and properly configured. Args: provider (str): Provider name. Returns: bool: True if available, False otherwise."""
    try:
        llm = get_langchain_llm(provider)
        return llm is not None
    except (ValueError, ImportError):
        return False
