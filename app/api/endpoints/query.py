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
        file_path = os.path.join("uploads", request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found at {file_path}")
            
        # Get stored metadata
        try:
            data_context = data_service.get_metadata(request.filename)
            logger.info(f"Retrieved metadata for file: {data_context}")
        except Exception as e:
            logger.warning(f"Could not retrieve metadata: {str(e)}")
            data_context = {}
        
        # Generate code using LLM
        try:
            generated_code = await llm_service.generate_code(
                query=request.query,
                data_info=data_context
            )
            logger.info(f"Generated code: {generated_code}")
            if not generated_code:
                raise HTTPException(status_code=500, detail="Failed to generate code: Empty response from LLM")
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(e)}")
        
        # Execute the generated code
        try:
            result = code_executor.execute_code(generated_code, file_path)
            logger.info(f"Execution result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to execute code: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 