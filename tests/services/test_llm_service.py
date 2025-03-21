import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService
from app.models.base import DataMetadata, DataFormat

@pytest.mark.asyncio
async def test_generate_code_success(sample_data_metadata, mock_openai_response):
    """Test successful code generation"""
    with patch('openai.ChatCompletion.acreate') as mock_create:
        # Configure the mock
        mock_create.return_value = mock_openai_response
        
        # Create service instance
        service = LLMService()
        
        # Test code generation
        query = "Show me a bar plot of average values by category"
        generated_code = await service.generate_code(query, sample_data_metadata)
        
        # Verify the generated code
        assert "import pandas as pd" in generated_code
        assert "import matplotlib.pyplot as plt" in generated_code
        assert "df.groupby('category')['value'].mean()" in generated_code
        
        # Verify OpenAI was called with correct parameters
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args['model'] == "gpt-4"
        assert call_args['temperature'] == 0.7
        assert call_args['max_tokens'] == 1000
        assert len(call_args['messages']) == 2  # system and user messages

@pytest.mark.asyncio
async def test_generate_code_without_markdown(sample_data_metadata):
    """Test code generation when response doesn't contain markdown"""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "import pandas as pd\nprint('Hello')"
                }
            }
        ]
    }
    
    with patch('openai.ChatCompletion.acreate') as mock_create:
        mock_create.return_value = mock_response
        
        service = LLMService()
        generated_code = await service.generate_code("test query", sample_data_metadata)
        
        assert generated_code == "import pandas as pd\nprint('Hello')"

@pytest.mark.asyncio
async def test_generate_code_error(sample_data_metadata):
    """Test error handling in code generation"""
    with patch('openai.ChatCompletion.acreate') as mock_create:
        mock_create.side_effect = Exception("API Error")
        
        service = LLMService()
        
        with pytest.raises(Exception) as exc_info:
            await service.generate_code("test query", sample_data_metadata)
        
        assert "Failed to generate code" in str(exc_info.value)

def test_create_system_message(sample_data_metadata):
    """Test system message creation"""
    service = LLMService()
    message = service._create_system_message(sample_data_metadata)
    
    # Verify message contains all necessary information
    assert sample_data_metadata.filename in message
    assert str(sample_data_metadata.format) in message
    assert str(sample_data_metadata.rows) in message
    assert all(col in message for col in sample_data_metadata.columns)
    assert "pandas" in message
    assert "numpy" in message
    assert "matplotlib" in message
    assert "plotly" in message 