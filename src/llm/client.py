"""LLM client module providing both legacy LLM clients and LangChain-compatible wrappers for use with the agentic workflow."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import google.generativeai as genai
from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError

from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger
from src.utils.performance import LLMCallTracker

MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0
BACKOFF_MULTIPLIER = 2.0

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
        """Generate text from the LLM.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.
        """
        pass

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate text from the LLM. Default implementation uses sync method in executor.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system_prompt)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview") -> None:
        self.api_key = api_key or get_env_var("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client: Optional[AsyncOpenAI] = None
        self.model = model

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client.

        Returns:
            AsyncOpenAI: Async client instance.
        """
        if self.async_client is None:
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        return self.async_client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from OpenAI with retry logic.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.

        Raises:
            Exception: If all retries fail.
        """
        from openai.types.chat import (
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
        )

        messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = []
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        backoff = INITIAL_BACKOFF
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0
                )
                latency = time.time() - start_time

                if attempt > 0:
                    logger.info(f"OpenAI request succeeded on attempt {attempt + 1}")

                usage = getattr(response, "usage", None)
                if usage:
                    from src.utils.performance import get_global_llm_tracker

                    tracker = LLMCallTracker(
                        model=self.model, provider="openai", global_tracking=True
                    )
                    tracker.record(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                        latency=latency,
                    )
                    global_tracker = get_global_llm_tracker()
                    if global_tracker:
                        with global_tracker.lock:
                            global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})
                    logger.debug(
                        f"[PERF] OpenAI call: {getattr(usage, 'total_tokens', 0)} tokens, "
                        f"cost={tracker.total_cost:.6f}, latency={latency:.3f}s"
                    )

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI returned empty response content")
                return str(content)
            except (RateLimitError, APIError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(backoff, MAX_BACKOFF)
                    logger.warning(
                        f"OpenAI API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    logger.error(f"OpenAI generation failed after {MAX_RETRIES} attempts: {e}")
            except Exception as e:
                logger.error(f"OpenAI generation failed with unexpected error: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("OpenAI generation failed: unknown error")

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate text from OpenAI with retry logic.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.

        Raises:
            Exception: If all retries fail.
        """
        from openai.types.chat import (
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
        )

        messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = []
        if system_prompt:
            messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))
        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        backoff = INITIAL_BACKOFF
        last_exception = None
        async_client = self._get_async_client()

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = await async_client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0
                )
                latency = time.time() - start_time

                if attempt > 0:
                    logger.info(f"OpenAI async request succeeded on attempt {attempt + 1}")

                usage = getattr(response, "usage", None)
                if usage:
                    from src.utils.performance import get_global_llm_tracker

                    tracker = LLMCallTracker(
                        model=self.model, provider="openai", global_tracking=True
                    )
                    tracker.record(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                        latency=latency,
                    )
                    global_tracker = get_global_llm_tracker()
                    if global_tracker:
                        with global_tracker.lock:
                            global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})
                    logger.debug(
                        f"[PERF] OpenAI async call: {getattr(usage, 'total_tokens', 0)} tokens, "
                        f"cost={tracker.total_cost:.6f}, latency={latency:.3f}s"
                    )

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI returned empty response content")
                return str(content)
            except (RateLimitError, APIError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(backoff, MAX_BACKOFF)
                    logger.warning(
                        f"OpenAI async API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    backoff *= BACKOFF_MULTIPLIER
                else:
                    logger.error(
                        f"OpenAI async generation failed after {MAX_RETRIES} attempts: {e}"
                    )
            except Exception as e:
                logger.error(f"OpenAI async generation failed with unexpected error: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("OpenAI async generation failed: unknown error")


class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro") -> None:
        self.api_key = api_key or get_env_var("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Gemini with retry logic.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.

        Raises:
            Exception: If all retries fail.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        backoff = INITIAL_BACKOFF
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                response = self.model.generate_content(full_prompt)
                latency = time.time() - start_time

                if attempt > 0:
                    logger.info(f"Gemini request succeeded on attempt {attempt + 1}")

                usage_metadata = getattr(response, "usage_metadata", None)
                if usage_metadata:
                    from src.utils.performance import get_global_llm_tracker

                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                    total_tokens = getattr(usage_metadata, "total_token_count", 0)

                    tracker = LLMCallTracker(
                        model=self.model_name, provider="gemini", global_tracking=True
                    )
                    tracker.record(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency=latency,
                    )
                    global_tracker = get_global_llm_tracker()
                    if global_tracker:
                        with global_tracker.lock:
                            global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})
                    logger.debug(
                        f"[PERF] Gemini call: {total_tokens} tokens, "
                        f"cost={tracker.total_cost:.6f}, latency={latency:.3f}s"
                    )

                return str(response.text)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["rate", "quota", "503", "500", "429"]):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = min(backoff, MAX_BACKOFF)
                        logger.warning(
                            f"Gemini API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        backoff *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(f"Gemini generation failed after {MAX_RETRIES} attempts: {e}")
                else:
                    logger.error(f"Gemini generation failed with non-retryable error: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini generation failed: unknown error")

    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate text using Gemini with retry logic.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Generated text.

        Raises:
            Exception: If all retries fail.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        backoff = INITIAL_BACKOFF
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                # Run synchronous Gemini call in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(full_prompt)
                )
                latency = time.time() - start_time

                if attempt > 0:
                    logger.info(f"Gemini async request succeeded on attempt {attempt + 1}")

                usage_metadata = getattr(response, "usage_metadata", None)
                if usage_metadata:
                    from src.utils.performance import get_global_llm_tracker

                    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                    completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                    total_tokens = getattr(usage_metadata, "total_token_count", 0)

                    tracker = LLMCallTracker(
                        model=self.model_name, provider="gemini", global_tracking=True
                    )
                    tracker.record(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency=latency,
                    )
                    global_tracker = get_global_llm_tracker()
                    if global_tracker:
                        with global_tracker.lock:
                            global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})
                    logger.debug(
                        f"[PERF] Gemini async call: {total_tokens} tokens, "
                        f"cost={tracker.total_cost:.6f}, latency={latency:.3f}s"
                    )

                return str(response.text)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["rate", "quota", "503", "500", "429"]):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = min(backoff, MAX_BACKOFF)
                        logger.warning(
                            f"Gemini async API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                        backoff *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(
                            f"Gemini async generation failed after {MAX_RETRIES} attempts: {e}"
                        )
                else:
                    logger.error(f"Gemini async generation failed with non-retryable error: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Gemini async generation failed: unknown error")


class DebugClient(LLMClient):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate mock text for debugging.

        Args:
            prompt (str): User prompt.
            system_prompt (Optional[str]): System prompt.

        Returns:
            str: Mock response.
        """
        logger.info("DebugClient: Generating mock response.")
        return "MOCK_RESPONSE"


def get_llm_client(provider: str = "openai") -> LLMClient:
    """Gets a legacy LLM client instance.

    Args:
        provider (str): Provider name ("openai", "gemini", or "debug").

    Returns:
        LLMClient: LLMClient instance.

    Raises:
        ValueError: If provider is unknown.
    """
    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "gemini":
        return GeminiClient()
    elif provider.lower() == "debug":
        return DebugClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_langchain_llm(
    provider: str = "openai", model: Optional[str] = None, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    """Get a LangChain-compatible LLM instance.

    Args:
        provider (str): Provider name.
        model (Optional[str]): Model name override.
        temperature (float): Generation temperature.
        **kwargs: Additional arguments.

    Returns:
        BaseChatModel: LangChain BaseChatModel instance.

    Raises:
        ValueError: If provider is unknown.
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        return _get_langchain_openai(model, temperature, **kwargs)
    elif provider_lower == "gemini":
        return _get_langchain_gemini(model, temperature, **kwargs)
    else:
        raise ValueError(f"Unknown LangChain provider: {provider}. Supported: openai, gemini")


def _get_langchain_openai(
    model: Optional[str] = None, temperature: float = 0.0, **kwargs
) -> "BaseChatModel":
    """Get a LangChain ChatOpenAI instance.

    Args:
        model (Optional[str]): Model name.
        temperature (float): Generation temperature.
        **kwargs: Additional arguments.

    Returns:
        BaseChatModel: ChatOpenAI instance.

    Raises:
        ImportError: If langchain-openai is not installed.
        ValueError: If OPENAI_API_KEY is not found.
    """
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

    from pydantic import SecretStr

    return ChatOpenAI(
        model=model_name, temperature=temperature, api_key=SecretStr(api_key), **kwargs
    )


def _get_langchain_gemini(
    model: Optional[str] = None, temperature: float = 0.0, **kwargs
) -> "BaseChatModel":
    """Get a LangChain ChatGoogleGenerativeAI instance.

    Args:
        model (Optional[str]): Model name.
        temperature (float): Generation temperature.
        **kwargs: Additional arguments.

    Returns:
        BaseChatModel: ChatGoogleGenerativeAI instance.

    Raises:
        ImportError: If langchain-google-genai is not installed.
        ValueError: If GEMINI_API_KEY is not found.
    """
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
        model=model_name, temperature=temperature, google_api_key=api_key, **kwargs
    )


def get_available_providers() -> List[str]:
    """Get list of available LLM providers.

    Returns:
        List[str]: Available provider names.
    """
    available = []

    try:
        from langchain_openai import ChatOpenAI  # noqa: F401

        if get_env_var("OPENAI_API_KEY"):
            available.append("openai")
    except ImportError:
        pass

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401

        if get_env_var("GEMINI_API_KEY"):
            available.append("gemini")
    except ImportError:
        pass

    return available


def validate_provider(provider: str) -> bool:
    """Check if a provider is available and properly configured.

    Args:
        provider (str): Provider name.

    Returns:
        bool: True if available, False otherwise.
    """
    try:
        llm = get_langchain_llm(provider)
        return llm is not None
    except (ValueError, ImportError):
        return False
