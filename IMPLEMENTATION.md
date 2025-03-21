# Data Agent - Implementation Details

## System Architecture

### 1. Core Components

#### 1.1 FastAPI Application Structure
```
app/
├── api/
│   ├── endpoints/
│   │   ├── data.py      # Data upload and management
│   │   ├── query.py     # Query processing
│   │   └── results.py   # Results handling
│   └── router.py
├── core/
│   ├── config.py        # Configuration management
│   ├── security.py      # Security utilities
│   └── logging.py       # Logging configuration
├── services/
│   ├── data_service.py  # Data processing
│   ├── llm_service.py   # LLM integration
│   └── exec_service.py  # Code execution
├── models/
│   ├── data.py         # Data models
│   ├── query.py        # Query models
│   └── response.py     # Response models
└── main.py
```

### 2. Key Classes and Interfaces

#### 2.1 Data Management
```python
class DataManager:
    """Handles data file processing and storage"""
    async def process_upload(self, file: UploadFile) -> DataMetadata
    async def get_metadata(self, file_id: str) -> DataMetadata
    async def get_preview(self, file_id: str) -> DataFrame
```

#### 2.2 Query Processing
```python
class QueryProcessor:
    """Processes natural language queries"""
    async def process_query(self, query: str, context: QueryContext) -> QueryResult
    async def validate_query(self, query: str) -> bool
```

#### 2.3 Code Execution
```python
class CodeExecutor:
    """Handles safe code execution"""
    async def execute(self, code: str, context: ExecutionContext) -> ExecutionResult
    async def validate_code(self, code: str) -> ValidationResult
```

### 3. Data Models

#### 3.1 Request/Response Models
```python
class DataMetadata(BaseModel):
    file_id: str
    columns: List[ColumnInfo]
    row_count: int
    preview_data: List[Dict]

class QueryRequest(BaseModel):
    query: str
    file_id: str
    context: Optional[Dict]

class ExecutionResult(BaseModel):
    status: str
    output: Union[str, Dict, bytes]
    error: Optional[str]
```

### 4. API Endpoints

#### 4.1 Data Management
- POST `/api/v1/data/upload`
- GET `/api/v1/data/{file_id}/metadata`
- GET `/api/v1/data/{file_id}/preview`

#### 4.2 Query Processing
- POST `/api/v1/query`
- GET `/api/v1/query/{query_id}/status`
- GET `/api/v1/query/{query_id}/result`

### 5. Security Measures

#### 5.1 Code Execution Safety
- Sandboxed environment configuration
- Resource usage limits
- Allowed imports whitelist
- Execution timeout settings

#### 5.2 Input Validation
- File type validation
- Size limits
- Content validation
- Query validation

### 6. Error Handling

#### 6.1 Custom Exceptions
```python
class DataAgentException(Exception):
    """Base exception for all app-specific errors"""

class InvalidFileError(DataAgentException):
    """Invalid file upload error"""

class ExecutionError(DataAgentException):
    """Code execution error"""
```

### 7. Configuration Management

#### 7.1 Environment Variables
```python
class Settings(BaseSettings):
    APP_NAME: str = "data-agent"
    DEBUG: bool = False
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx"]
    LLM_API_KEY: str
    EXECUTION_TIMEOUT: int = 30  # seconds
```

### 8. Deployment Configuration

#### 8.1 Docker Configuration
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
``` 