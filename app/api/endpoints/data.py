from fastapi import APIRouter, UploadFile, File, HTTPException
from ...services.data_service import data_service
from ...models.base import DataMetadata
from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a data file (CSV or Excel)
    """
    try:
        logger.debug(f"Uploading file: {file.filename}")
        file_path = await data_service.save_file(file)
        logger.debug(f"File saved at: {file_path}")
        return {"filename": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files", response_model=List[str])
async def list_files():
    """
    List all uploaded files
    """
    try:
        logger.debug("Listing files in upload directory")
        files = [f for f in os.listdir(data_service.upload_dir) 
                if os.path.isfile(os.path.join(data_service.upload_dir, f))]
        logger.debug(f"Found files: {files}")
        return files
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metadata/{filename}")
async def get_file_metadata(filename: str):
    """
    Get basic info about a specific file
    """
    logger.debug(f"Getting info for file: {filename}")
    file_path = os.path.join(data_service.upload_dir, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Just return basic file info
        return {
            "filename": filename,
            "path": file_path,
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        logger.error(f"Error in get_file_metadata: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 