from pydantic import BaseModel
from typing import Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for processing a query."""
    query: str
    filename: str

class QueryResponse(BaseModel):
    """Response model for query processing results."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None 