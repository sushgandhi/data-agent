import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.models.base import QueryRequest, ExecutionResult
import os

client = TestClient(app)

@pytest.fixture
def mock_services():
    """Mock the services used in the query endpoint"""
    with patch('app.api.endpoints.query.data_service') as mock_data_service, \
         patch('app.api.endpoints.query.LLMService') as mock_llm_service, \
         patch('app.api.endpoints.query.code_executor') as mock_code_executor:
        
        # Configure mock data service
        mock_data_service.get_metadata.return_value = MagicMock(
            filename="test.csv",
            format="csv",
            rows=100,
            columns=["id", "name", "value", "category"],
            preview=[{"id": 1, "name": "Item 1", "value": 10.5, "category": "A"}]
        )
        
        # Configure mock LLM service
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate_code.return_value = "import pandas as pd\nprint('test')"
        mock_llm_service.return_value = mock_llm_instance
        
        # Configure mock code executor
        mock_code_executor.execute_code.return_value = ExecutionResult(
            success=True,
            result={"test": "data"},
            execution_time=0.1,
            memory_usage=10.0,
            code_generated="import pandas as pd\nprint('test')"
        )
        
        yield {
            'data_service': mock_data_service,
            'llm_service': mock_llm_instance,
            'code_executor': mock_code_executor
        }

def test_process_query_success(mock_services):
    """Test successful query processing"""
    query_request = {
        "query": "Show me a bar plot of values by category",
        "filename": "test.csv"
    }
    
    response = client.post("/api/v1/query/process", json=query_request)
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert "result" in result
    assert "execution_time" in result
    assert "memory_usage" in result
    assert "code_generated" in result
    
    # Verify service calls
    mock_services['data_service'].get_metadata.assert_called_once()
    mock_services['llm_service'].generate_code.assert_called_once()
    mock_services['code_executor'].execute_code.assert_called_once()

def test_process_query_file_not_found(mock_services):
    """Test query processing with non-existent file"""
    mock_services['data_service'].get_metadata.side_effect = FileNotFoundError()
    
    query_request = {
        "query": "Show me a bar plot of values by category",
        "filename": "nonexistent.csv"
    }
    
    response = client.post("/api/v1/query/process", json=query_request)
    
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]

def test_process_query_llm_error(mock_services):
    """Test query processing with LLM service error"""
    mock_services['llm_service'].generate_code.side_effect = Exception("LLM Error")
    
    query_request = {
        "query": "Show me a bar plot of values by category",
        "filename": "test.csv"
    }
    
    response = client.post("/api/v1/query/process", json=query_request)
    
    assert response.status_code == 500
    assert "LLM Error" in response.json()["detail"]

def test_process_query_execution_error(mock_services):
    """Test query processing with code execution error"""
    mock_services['code_executor'].execute_code.return_value = ExecutionResult(
        success=False,
        error="Execution Error",
        execution_time=0.1,
        code_generated="import pandas as pd\nprint('test')"
    )
    
    query_request = {
        "query": "Show me a bar plot of values by category",
        "filename": "test.csv"
    }
    
    response = client.post("/api/v1/query/process", json=query_request)
    
    assert response.status_code == 500
    assert "Execution Error" in response.json()["detail"] 