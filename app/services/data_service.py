import os
import json
import pandas as pd
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
from ..core.config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.metadata_dir = os.path.join(self.upload_dir, ".metadata")
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    async def save_file(self, file: UploadFile) -> str:
        """Save uploaded file and its metadata, then return the file path"""
        if not self._is_valid_extension(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format")
            
        file_path = os.path.join(self.upload_dir, file.filename)
        try:
            # Save the file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
                
            # Generate and save metadata
            metadata = self._generate_metadata(file_path)
            self._save_metadata(file.filename, metadata)
            
            return file_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def get_metadata(self, filename: str) -> Dict[str, Any]:
        """Retrieve metadata for a file"""
        metadata_path = os.path.join(self.metadata_dir, f"{filename}.json")
        logger.info(f"Looking for metadata at: {metadata_path}")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Metadata not found")
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Retrieved metadata structure: {json.dumps(metadata, indent=2)}")
                return metadata
        except Exception as e:
            logger.error(f"Failed to read metadata: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to read metadata: {str(e)}")
    
    def _generate_metadata(self, file_path: str) -> Dict[str, Any]:
        """Generate metadata for a file"""
        try:
            # Read just the first few rows to get column info
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df_preview = pd.read_csv(file_path, nrows=5)
            elif file_ext in ['.xlsx', '.xls']:
                df_preview = pd.read_excel(file_path, nrows=5)
            else:
                raise ValueError("Unsupported file format")
            
            # Get column info with data types
            column_info = {}
            for col in df_preview.columns:
                dtype = str(df_preview[col].dtype)
                # Make dtype names more readable/intuitive
                if dtype.startswith('int'):
                    dtype = 'integer'
                elif dtype.startswith('float'):
                    dtype = 'float'
                elif dtype.startswith('datetime'):
                    dtype = 'datetime'
                elif dtype.startswith('bool'):
                    dtype = 'boolean'
                else:
                    dtype = 'string'
                
                # Include sample values for better context
                non_null_values = df_preview[col].dropna()
                # Convert numpy values to native Python types
                sample_values = []
                for val in non_null_values.head(3):
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    elif isinstance(val, np.bool_):
                        val = bool(val)
                    elif isinstance(val, np.datetime64):
                        val = pd.Timestamp(val).isoformat()
                    sample_values.append(val)
                
                column_info[col] = {
                    'type': dtype,
                    'sample_values': sample_values,
                    'has_nulls': bool(df_preview[col].isnull().any())  # Convert numpy.bool_ to Python bool
                }
            
            # Read total rows (with native type conversion)
            if file_ext == '.csv':
                total_rows = len(pd.read_csv(file_path))
            else:
                total_rows = len(pd.read_excel(file_path))
                
            return {
                "filename": os.path.basename(file_path),
                "columns": list(df_preview.columns),  # Convert Index to list
                "column_info": column_info,
                "total_rows": total_rows,
                "preview_rows": len(df_preview),
                "file_path": str(file_path),  # Ensure path is string
                "size_bytes": int(os.path.getsize(file_path)),  # Convert to int
                "created_at": float(os.path.getctime(file_path))  # Convert to float
            }
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate metadata: {str(e)}")
    
    def _save_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to a JSON file"""
        metadata_path = os.path.join(self.metadata_dir, f"{filename}.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")
    
    def _is_valid_extension(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        return ext in settings.ALLOWED_EXTENSIONS

# Create global instance
data_service = DataService() 