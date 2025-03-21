import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.services.code_executor import CodeExecutor
from app.models.base import ExecutionResult
import os
import tempfile

@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        return f.name

@pytest.mark.asyncio
async def test_execute_code_success(sample_csv_file):
    """Test successful code execution"""
    code = """
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('{}')

# Create a simple bar plot
plt.figure(figsize=(10, 6))
df.groupby('category')['value'].mean().plot(kind='bar')
plt.title('Average Value by Category')

# Store the result
result = plt.gcf()
""".format(sample_csv_file)

    executor = CodeExecutor()
    result = await executor.execute_code(code, sample_csv_file)
    
    assert result.success
    assert result.execution_time > 0
    assert result.memory_usage is not None
    assert result.code_generated == code
    assert result.result is not None

@pytest.mark.asyncio
async def test_execute_code_with_dataframe_result(sample_csv_file):
    """Test code execution returning a DataFrame"""
    code = """
import pandas as pd

# Read the data
df = pd.read_csv('{}')

# Calculate summary statistics
result = df.groupby('category')['value'].agg(['mean', 'count'])
""".format(sample_csv_file)

    executor = CodeExecutor()
    result = await executor.execute_code(code, sample_csv_file)
    
    assert result.success
    assert isinstance(result.result, pd.DataFrame)
    assert 'mean' in result.result.columns
    assert 'count' in result.result.columns

@pytest.mark.asyncio
async def test_execute_code_error(sample_csv_file):
    """Test error handling in code execution"""
    code = """
import pandas as pd

# This will raise an error
df = pd.read_csv('nonexistent_file.csv')
"""

    executor = CodeExecutor()
    result = await executor.execute_code(code, sample_csv_file)
    
    assert not result.success
    assert result.error is not None
    assert result.execution_time > 0
    assert result.code_generated == code

@pytest.mark.asyncio
async def test_execute_code_with_plotly(sample_csv_file):
    """Test code execution with Plotly visualization"""
    code = """
import pandas as pd
import plotly.express as px

# Read the data
df = pd.read_csv('{}')

# Create a Plotly figure
result = px.bar(df, x='category', y='value', title='Values by Category')
""".format(sample_csv_file)

    executor = CodeExecutor()
    result = await executor.execute_code(code, sample_csv_file)
    
    assert result.success
    assert result.result is not None
    # Plotly figures are dictionaries with specific keys
    assert isinstance(result.result, dict)
    assert 'data' in result.result
    assert 'layout' in result.result

def test_cleanup_after_execution(sample_csv_file):
    """Test that matplotlib figures are cleaned up after execution"""
    code = """
import matplotlib.pyplot as plt

# Create a figure
plt.figure()
"""

    executor = CodeExecutor()
    executor.execute_code(code, sample_csv_file)
    
    # Verify no figures are left open
    assert len(plt.get_fignums()) == 0 