from fastapi import APIRouter, HTTPException
from ...services.data_service import data_service
from ...services.llm_service import LLMService
from ...services.code_executor import code_executor
from ...models.base import QueryRequest, ExecutionResult
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
llm_service = LLMService()

@router.post("/process", response_model=ExecutionResult)
async def process_query(query_request: QueryRequest):
    """
    Process a natural language query and return the execution results
    """
    try:
        # Get file path and validate
        file_path = os.path.join(data_service.upload_dir, query_request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Get stored metadata
        try:
            data_context = data_service.get_metadata(query_request.filename)
            logger.info(f"Retrieved metadata for file: {data_context}")
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving file metadata: {str(e)}")
        
        # Generate code using LLM
        try:
            generated_code = await llm_service.generate_code(
                query_request.query,
                data_context
            )
            logger.info(f"Generated code: {generated_code}")
            if not generated_code:
                raise HTTPException(status_code=500, detail="Failed to generate code: Empty response from LLM")
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(e)}")
        
        # Execute the generated code
        try:
            result = await code_executor.execute_code(generated_code, file_path)
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