from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import (
    DebugClient,
    GeminiClient,
    OpenAIClient,
    get_available_providers,
    get_langchain_llm,
    get_llm_client,
    validate_provider,
)


@patch("src.llm.client.OpenAI")
@patch("src.llm.client.get_env_var")
def test_openai_client(mock_get_env, mock_openai):
    mock_get_env.return_value = "fake-key"
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Response"
    mock_openai.return_value.chat.completions.create.return_value = mock_completion

    client = OpenAIClient()
    response = client.generate("Hello")
    assert response == "Response"
    mock_openai.return_value.chat.completions.create.assert_called_once()


@patch("src.llm.client.genai")
@patch("src.llm.client.get_env_var")
def test_gemini_client(mock_get_env, mock_genai):
    mock_get_env.return_value = "fake-key"
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Response"
    mock_genai.GenerativeModel.return_value = mock_model

    client = GeminiClient()
    response = client.generate("Hello")
    assert response == "Response"
    mock_model.generate_content.assert_called_once()


def test_debug_client():
    client = DebugClient()
    assert client.generate("test") == "MOCK_RESPONSE"


def test_get_llm_client():
    assert isinstance(get_llm_client("debug"), DebugClient)

    with patch("src.llm.client.get_env_var", return_value="key"):
        with patch("src.llm.client.OpenAI"):
            assert isinstance(get_llm_client("openai"), OpenAIClient)


@patch("src.llm.client.get_env_var")
def test_get_langchain_openai_valid_key(mock_get_env):
    mock_get_env.return_value = "valid-key"
    mock_llm = MagicMock()

    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_chat_openai:
        from src.llm.client import _get_langchain_openai

        result = _get_langchain_openai()

        mock_chat_openai.assert_called_once()
        assert result == mock_llm


@patch("src.llm.client.get_env_var")
def test_get_langchain_openai_missing_key(mock_get_env):
    mock_get_env.return_value = None

    from src.llm.client import _get_langchain_openai

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        _get_langchain_openai()


@patch("src.llm.client.get_env_var")
def test_get_langchain_openai_custom_model(mock_get_env):
    mock_get_env.return_value = "valid-key"
    mock_llm = MagicMock()

    with patch("langchain_openai.ChatOpenAI", return_value=mock_llm) as mock_chat_openai:
        from src.llm.client import _get_langchain_openai

        result = _get_langchain_openai(model="gpt-4", temperature=0.5)

        assert mock_chat_openai.called
        call_kwargs = mock_chat_openai.call_args[1] if mock_chat_openai.call_args else {}
        if "model" in call_kwargs:
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 0.5


@patch("src.llm.client.get_env_var")
def test_get_langchain_openai_import_error(mock_get_env):
    mock_get_env.return_value = "valid-key"

    with patch("langchain_openai.ChatOpenAI", side_effect=ImportError("No module")):
        from src.llm.client import _get_langchain_openai

        with pytest.raises(ImportError):
            _get_langchain_openai()


@patch("src.llm.client.get_env_var")
def test_get_langchain_gemini_valid_key(mock_get_env):
    mock_get_env.return_value = "valid-key"
    mock_llm = MagicMock()

    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm
    ) as mock_chat_gemini:
        from src.llm.client import _get_langchain_gemini

        result = _get_langchain_gemini()

        mock_chat_gemini.assert_called_once()
        assert result == mock_llm


@patch("src.llm.client.get_env_var")
def test_get_langchain_gemini_missing_key(mock_get_env):
    mock_get_env.return_value = None

    from src.llm.client import _get_langchain_gemini

    with pytest.raises(ValueError, match="GEMINI_API_KEY not found"):
        _get_langchain_gemini()


@patch("src.llm.client.get_env_var")
def test_get_langchain_gemini_custom_model(mock_get_env):
    mock_get_env.return_value = "valid-key"
    mock_llm = MagicMock()

    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm
    ) as mock_chat_gemini:
        from src.llm.client import _get_langchain_gemini

        result = _get_langchain_gemini(model="gemini-1.5-flash", temperature=0.3)

        assert mock_chat_gemini.called
        call_kwargs = mock_chat_gemini.call_args[1] if mock_chat_gemini.call_args else {}
        if "model" in call_kwargs:
            assert call_kwargs["model"] == "gemini-1.5-flash"
            assert call_kwargs["temperature"] == 0.3


@patch("src.llm.client.get_env_var")
def test_get_langchain_gemini_import_error(mock_get_env):
    mock_get_env.return_value = "valid-key"

    with patch(
        "langchain_google_genai.ChatGoogleGenerativeAI", side_effect=ImportError("No module")
    ):
        from src.llm.client import _get_langchain_gemini

        with pytest.raises(ImportError):
            _get_langchain_gemini()


@patch("src.llm.client._get_langchain_openai")
def test_get_langchain_llm_openai(mock_get_openai):
    mock_llm = MagicMock()
    mock_get_openai.return_value = mock_llm

    result = get_langchain_llm("openai")

    mock_get_openai.assert_called_once()
    assert result == mock_llm


@patch("src.llm.client._get_langchain_gemini")
def test_get_langchain_llm_gemini(mock_get_gemini):
    mock_llm = MagicMock()
    mock_get_gemini.return_value = mock_llm

    result = get_langchain_llm("gemini")

    mock_get_gemini.assert_called_once()
    assert result == mock_llm


def test_get_langchain_llm_invalid_provider():
    with pytest.raises(ValueError, match="Unknown LangChain provider"):
        get_langchain_llm("invalid_provider")


@patch("src.llm.client._get_langchain_openai")
def test_get_langchain_llm_kwargs_passthrough(mock_get_openai):
    mock_llm = MagicMock()
    mock_get_openai.return_value = mock_llm

    result = get_langchain_llm("openai", model="gpt-4", temperature=0.5, max_tokens=100)

    mock_get_openai.assert_called_once()
    call_kwargs = mock_get_openai.call_args[1] if mock_get_openai.call_args else {}
    if "model" in call_kwargs:
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
    else:
        assert mock_get_openai.called


@patch("src.llm.client.get_env_var")
def test_get_available_providers_both_installed(mock_get_env):
    mock_get_env.side_effect = lambda key: (
        "key" if key in ["OPENAI_API_KEY", "GEMINI_API_KEY"] else None
    )

    with (
        patch("langchain_openai.ChatOpenAI", create=True),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", create=True),
    ):
        providers = get_available_providers()

        assert "openai" in providers
        assert "gemini" in providers


@patch("src.llm.client.get_env_var")
def test_get_available_providers_only_openai(mock_get_env):
    mock_get_env.side_effect = lambda key: "key" if key == "OPENAI_API_KEY" else None

    with patch("src.llm.client.ChatOpenAI", create=True) as mock_openai:
        import sys

        mock_openai_module = MagicMock()
        mock_openai_module.ChatOpenAI = MagicMock()
        sys.modules["langchain_openai"] = mock_openai_module

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "langchain_google_genai":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            providers = get_available_providers()

            assert "openai" in providers
            assert "gemini" not in providers

        if "langchain_openai" in sys.modules:
            del sys.modules["langchain_openai"]


@patch("src.llm.client.get_env_var")
def test_get_available_providers_only_gemini(mock_get_env):
    mock_get_env.side_effect = lambda key: "key" if key == "GEMINI_API_KEY" else None

    import sys

    mock_gemini_module = MagicMock()
    mock_gemini_module.ChatGoogleGenerativeAI = MagicMock()
    sys.modules["langchain_google_genai"] = mock_gemini_module

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "langchain_openai":
            raise ImportError("No module")
        return original_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=mock_import):
            providers = get_available_providers()

            assert "openai" not in providers
            assert "gemini" in providers
    finally:
        if "langchain_google_genai" in sys.modules:
            del sys.modules["langchain_google_genai"]


@patch("src.llm.client.get_env_var")
def test_get_available_providers_neither_installed(mock_get_env):
    mock_get_env.return_value = None

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name in ("langchain_openai", "langchain_google_genai"):
            raise ImportError("No module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        providers = get_available_providers()

        assert len(providers) == 0


@patch("src.llm.client.get_env_var")
def test_get_available_providers_missing_api_keys(mock_get_env):
    mock_get_env.return_value = None

    with (
        patch("langchain_openai.ChatOpenAI", create=True),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", create=True),
    ):
        providers = get_available_providers()

        assert len(providers) == 0


@patch("src.llm.client.get_langchain_llm")
def test_validate_provider_valid(mock_get_llm):
    mock_get_llm.return_value = MagicMock()

    result = validate_provider("openai")

    assert result is True
    mock_get_llm.assert_called_once_with("openai")


@patch("src.llm.client.get_langchain_llm")
def test_validate_provider_invalid(mock_get_llm):
    mock_get_llm.side_effect = ValueError("Invalid provider")

    result = validate_provider("invalid")

    assert result is False


@patch("src.llm.client.get_langchain_llm")
def test_validate_provider_import_error(mock_get_llm):
    mock_get_llm.side_effect = ImportError("No module")

    result = validate_provider("openai")

    assert result is False
