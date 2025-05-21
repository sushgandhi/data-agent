"""
API interface for the Data Flow Agent system.

This module provides a FastAPI application that exposes the Data Flow Agent
functionality through RESTful endpoints.

To run the API server:
    uvicorn api:app --reload

Then access the API documentation at:
    http://localhost:8000/docs

Available endpoints:
- POST /query: Process a natural language query on a data file
- GET /metadata: Get metadata about the currently loaded data
- GET /health: Health check endpoint
- POST /upload: Upload a data file
"""

import os
import json
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_flow_agent.application import ApplicationManager

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize application manager
app_manager = ApplicationManager()
app_manager.initialize()

# Create FastAPI app
app = FastAPI(
    title="Data Flow Agent API",
    description="API for analyzing data using natural language queries",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom JSON encoder to handle numpy and pandas types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif np.isnan(obj):
            return None
        return super().default(obj)

def serialize_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize response data to ensure it's JSON-compatible.
    
    Args:
        response_data: The response data to serialize
        
    Returns:
        JSON-compatible dictionary
    """
    # First, convert any DataFrame results to dict format
    if isinstance(response_data, dict) and 'result' in response_data:
        result = response_data['result']
        if isinstance(result, dict) and 'result_df' in result:
            # Store DataFrame as records
            result['result_df'] = result['result_df'].to_dict(orient='records')
    
    # Then use the custom encoder for any remaining numpy types
    json_str = json.dumps(response_data, cls=CustomJSONEncoder)
    return json.loads(json_str)

# Request models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query to analyze the data")
    file_path: str = Field(..., description="Path to the data file (CSV or Excel)")
    sheet_name: Optional[str] = Field(None, description="Sheet name for Excel files")

# Response models
class QueryResponse(BaseModel):
    type: str
    result: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None

@app.post("/query", response_model=Dict[str, Any])
async def query_data(request: QueryRequest) -> Dict[str, Any]:
    """
    Process a natural language query on a data file.
    
    Args:
        request: Query request containing the query and file path
        
    Returns:
        Result of processing the query
    """
    try:
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Load the file
        app_manager.load_data(request.file_path, request.sheet_name)
        
        # Process the query
        result = app_manager.process_query(request.query)
        
        # Serialize the response to ensure it's JSON-compatible
        serialized_result = serialize_response(result)
        
        return serialized_result
        
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error_type": error_type,
                "error_message": error_detail
            }
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file for analysis.
    
    Args:
        file: The file to upload
        
    Returns:
        Dictionary with file path and success status
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.csv', '.xlsx', '.xls']:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Invalid file type. Only CSV and Excel files are supported."
                }
            )
        
        # Create a unique filename to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return the full path
        return {
            "success": True,
            "file_path": str(file_path.absolute()),
            "filename": file.filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/metadata")
async def get_metadata() -> Dict[str, Any]:
    """
    Get metadata about the currently loaded data.
    
    Returns:
        Metadata about the current data
    """
    try:
        # Get metadata from data context
        metadata = app_manager.data_context.get_metadata()
        
        # Serialize the response
        serialized_metadata = serialize_response(metadata)
        
        return serialized_metadata
        
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)
        
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error_type": error_type,
                "error_message": error_detail
            }
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080) 
