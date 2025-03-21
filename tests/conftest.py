import pytest
from datetime import datetime
from app.models.base import DataMetadata, DataFormat

@pytest.fixture
def sample_data_metadata():
    """Fixture providing sample data metadata for testing"""
    return DataMetadata(
        filename="test_data.csv",
        format=DataFormat.CSV,
        upload_time=datetime.now(),
        rows=100,
        columns=["id", "name", "value", "category"],
        preview=[
            {"id": 1, "name": "Item 1", "value": 10.5, "category": "A"},
            {"id": 2, "name": "Item 2", "value": 20.3, "category": "B"}
        ]
    )

@pytest.fixture
def sample_csv_content():
    """Fixture providing sample CSV content for testing"""
    return """id,name,value,category
1,Item 1,10.5,A
2,Item 2,20.3,B
3,Item 3,15.7,A"""

@pytest.fixture
def mock_openai_response():
    """Fixture providing a mock OpenAI response"""
    return {
        "choices": [
            {
                "message": {
                    "content": """```python
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('test_data.csv')

# Create a bar plot
plt.figure(figsize=(10, 6))
df.groupby('category')['value'].mean().plot(kind='bar')
plt.title('Average Value by Category')
plt.xlabel('Category')
plt.ylabel('Average Value')

# Store the result
result = plt.gcf()
```"""
                }
            }
        ]
    } 