"""Tests for LLM response caching."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.client import (
    CachedLangChainLLM,
    DebugClient,
    GeminiClient,
    OpenAIClient,
    _generate_cache_key,
)
from src.utils.cache_manager import get_cache_manager


class TestLLMCaching:
    """Test suite for LLM caching functionality."""

    def test_generate_cache_key(self) -> None:
        """Test cache key generation."""
        key1 = _generate_cache_key("prompt1", "system1", "model1")
        key2 = _generate_cache_key("prompt1", "system1", "model1")
        key3 = _generate_cache_key("prompt2", "system1", "model1")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64

    def test_openai_client_cache_hit(self) -> None:
        """Test OpenAI client returns cached response."""
        cache_manager = get_cache_manager()
        cache_manager.clear("llm")

        with patch("src.llm.client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "cached response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test_key", model="gpt-4")
            result1 = client.generate("test prompt", "system prompt")
            # Clear mock call count to isolate second call
            mock_client.chat.completions.create.reset_mock()
            result2 = client.generate("test prompt", "system prompt")

            assert result1 == "cached response"
            assert result2 == "cached response"
            # Second call should not make API call due to cache
            assert mock_client.chat.completions.create.call_count == 0

    def test_openai_client_cache_miss(self) -> None:
        """Test OpenAI client makes API call on cache miss."""
        with patch("src.llm.client.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test_key", model="gpt-4")
            result1 = client.generate("prompt1", "system")
            result2 = client.generate("prompt2", "system")

            assert result1 == "response"
            assert result2 == "response"
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_openai_client_async_cache_hit(self) -> None:
        """Test OpenAI async client returns cached response."""
        cache_manager = get_cache_manager()
        cache_manager.clear("llm")

        with patch("src.llm.client.AsyncOpenAI") as mock_async_openai_class:
            mock_async_client = MagicMock()
            mock_async_openai_class.return_value = mock_async_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "async cached"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAIClient(api_key="test_key", model="gpt-4")
            result1 = await client.agenerate("test prompt", "system")
            # Reset mock to check if second call uses cache
            mock_async_client.chat.completions.create.reset_mock()
            result2 = await client.agenerate("test prompt", "system")

            assert result1 == "async cached"
            assert result2 == "async cached"
            # Second call should not make API call due to cache
            assert mock_async_client.chat.completions.create.call_count == 0

    def test_gemini_client_cache_hit(self) -> None:
        """Test Gemini client returns cached response."""
        with (
            patch("src.llm.client.genai.configure"),
            patch("src.llm.client.genai.GenerativeModel") as mock_model_class,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            mock_response = MagicMock()
            mock_response.text = "gemini cached"
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 20
            mock_response.usage_metadata.total_token_count = 30

            mock_model.generate_content.return_value = mock_response

            client = GeminiClient(api_key="test_key", model="gemini-pro")
            result1 = client.generate("test prompt", "system")
            result2 = client.generate("test prompt", "system")

            assert result1 == "gemini cached"
            assert result2 == "gemini cached"
            assert mock_model.generate_content.call_count == 1

    def test_debug_client_no_caching(self) -> None:
        """Test DebugClient doesn't use caching."""
        client = DebugClient()
        result1 = client.generate("prompt1")
        result2 = client.generate("prompt2")

        assert result1 == "MOCK_RESPONSE"
        assert result2 == "MOCK_RESPONSE"

    def test_cache_key_includes_model(self) -> None:
        """Test cache key includes model name."""
        key1 = _generate_cache_key("prompt", "system", "model1")
        key2 = _generate_cache_key("prompt", "system", "model2")

        assert key1 != key2

    def test_cache_key_includes_system_prompt(self) -> None:
        """Test cache key includes system prompt."""
        key1 = _generate_cache_key("prompt", "system1", "model")
        key2 = _generate_cache_key("prompt", "system2", "model")

        assert key1 != key2

    def test_cache_key_handles_none_system_prompt(self) -> None:
        """Test cache key handles None system prompt (None and empty string produce same key)."""
        key1 = _generate_cache_key("prompt", None, "model")
        key2 = _generate_cache_key("prompt", "", "model")

        assert key1 == key2

    def test_cached_langchain_llm_cache_hit(self) -> None:
        """Test CachedLangChainLLM returns cached response."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_response = MagicMock()
        mock_response.content = "cached response"

        mock_llm.invoke.return_value = mock_response

        cached_llm = CachedLangChainLLM(mock_llm)

        message1 = MagicMock()
        message1.type = "human"
        message1.content = "test message"
        messages = [message1]

        result1 = cached_llm.invoke(messages)
        result2 = cached_llm.invoke(messages)

        assert result1.content == "cached response"
        assert result2.content == "cached response"
        assert mock_llm.invoke.call_count == 1

    @pytest.mark.asyncio
    async def test_cached_langchain_llm_async_cache_hit(self) -> None:
        """Test CachedLangChainLLM async returns cached response."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_response = MagicMock()
        mock_response.content = "async cached"

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        cached_llm = CachedLangChainLLM(mock_llm)

        message1 = MagicMock()
        message1.type = "human"
        message1.content = "test message"
        messages = [message1]

        result1 = await cached_llm.ainvoke(messages)
        result2 = await cached_llm.ainvoke(messages)

        assert result1.content == "async cached"
        assert result2.content == "async cached"
        assert mock_llm.ainvoke.call_count == 1

    def test_cached_langchain_llm_cache_miss(self) -> None:
        """Test CachedLangChainLLM makes API call on cache miss."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_response1 = MagicMock()
        mock_response1.content = "response1"
        mock_response2 = MagicMock()
        mock_response2.content = "response2"

        mock_llm.invoke.side_effect = [mock_response1, mock_response2]

        cached_llm = CachedLangChainLLM(mock_llm)

        message1 = MagicMock()
        message1.type = "human"
        message1.content = "message1"
        messages1 = [message1]

        message2 = MagicMock()
        message2.type = "human"
        message2.content = "message2"
        messages2 = [message2]

        result1 = cached_llm.invoke(messages1)
        result2 = cached_llm.invoke(messages2)

        assert result1.content == "response1"
        assert result2.content == "response2"
        assert mock_llm.invoke.call_count == 2

    def test_cached_langchain_llm_delegates_attributes(self) -> None:
        """Test CachedLangChainLLM delegates attribute access to wrapped LLM."""
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_llm.some_attribute = "test_value"

        cached_llm = CachedLangChainLLM(mock_llm)

        assert cached_llm.some_attribute == "test_value"
        assert cached_llm.model_name == "test-model"
