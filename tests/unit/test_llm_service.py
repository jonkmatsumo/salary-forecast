import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.services.llm_service import LLMService

@pytest.fixture
def mock_client():
    return MagicMock()

@patch("src.services.llm_service.get_llm_client")
def test_generate_config_success(mock_get_client):
    mock_client_instance = MagicMock()
    mock_client_instance.generate.return_value = '{"model": {"targets": ["Salary"]}}'
    mock_get_client.return_value = mock_client_instance
    
    service = LLMService(provider="debug")
    df = pd.DataFrame({"Salary": [100]})
    
    config = service.generate_config(df)
    
    assert config["model"]["targets"] == ["Salary"]
    mock_client_instance.generate.assert_called_once()

@patch("src.services.llm_service.get_llm_client")
def test_generate_config_json_error(mock_get_client):
    mock_client_instance = MagicMock()
    mock_client_instance.generate.return_value = 'Not JSON'
    mock_get_client.return_value = mock_client_instance
    
    service = LLMService(provider="debug")
    df = pd.DataFrame({"Salary": [100]})
    
    with pytest.raises(ValueError, match="LLM did not return valid JSON"):
        service.generate_config(df)
