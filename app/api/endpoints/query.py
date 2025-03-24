from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.data_service import data_service
from ...services.llm_service import LLMService
from ...services.code_executor import code_executor
from ...models.query import QueryRequest, QueryResponse
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
llm_service = LLMService()

@router.post("/process", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> Dict[str, Any]:
    """Process a query and return the analysis results."""
    try:
        # Get file path and validate
        file_path = os.path.join(data_service.upload_dir, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Get stored metadata
        try:
            data_context = data_service.get_metadata(request.filename)
            logger.info(f"Retrieved metadata for file: {data_context}")
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving file metadata: {str(e)}")
        
        # Generate code using LLM
        generated_code = await llm_service.generate_code(
            query=request.query,
            data_info={"filename": request.filename}
        )
        print("Generated code: ", generated_code)
        # Execute the generated code
        result = code_executor.execute_code(
            code=generated_code,
            file_path=f"data/{request.filename}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 