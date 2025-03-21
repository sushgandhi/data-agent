from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DataFormat(str, Enum):
    CSV = "csv"
    EXCEL = "xlsx"
    XLS = "xls"

class DataMetadata(BaseModel):
    filename: str
    format: DataFormat
    upload_time: datetime = Field(default_factory=datetime.now)
    rows: Optional[int] = None
    columns: Optional[List[str]] = None
    preview: Optional[List[Dict[str, Any]]] = None

class QueryRequest(BaseModel):
    query: str
    filename: str
    context: Optional[Dict[str, Any]] = None

class ExecutionResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    memory_usage: Optional[float] = None
    code_generated: Optional[str] = None 