import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import time
import psutil
import os
from ..core.config import settings
from ..models.base import ExecutionResult
import logging

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self):
        self.process = psutil.Process()
        
    def _clean_code(self, code: str) -> str:
        """Clean up the code by removing markdown formatting and imports."""
        # Remove markdown code block markers
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].strip()
            
        # Remove import statements
        lines = code.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('import ')]
        return '\n'.join(cleaned_lines)
        
    async def execute_code(self, code: str, file_path: str) -> ExecutionResult:
        """
        Execute the generated code in a sandboxed environment
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        try:
            # Clean up the code
            code = self._clean_code(code)
            logger.info(f"Cleaned code:\n{code}")
            
            # Create a local namespace for code execution
            local_ns = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'px': px,
                'go': go,
                'file_path': file_path
            }
            
            # Create a global namespace with the same imports
            global_ns = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'px': px,
                'go': go,
                'file_path': file_path
            }
            
            # Execute the code to define the analyze_data function
            exec(code, global_ns, local_ns)
            
            # Check if analyze_data function exists
            if 'analyze_data' not in local_ns:
                raise ValueError("Generated code does not contain 'analyze_data' function")
            
            # Call the analyze_data function with the file path
            result = local_ns['analyze_data'](file_path)
            
            # Handle plotly figures by converting to JSON
            if hasattr(result, 'to_json'):
                result = {
                    'plot': result.to_json(),
                    'data': None
                }
            elif isinstance(result, (pd.DataFrame, pd.Series)):
                result = {
                    'plot': None,
                    'data': result.to_dict()
                }
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            end_memory = self.process.memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return ExecutionResult(
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                memory_usage=memory_usage,
                code_generated=code
            )
            
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time,
                memory_usage=None,
                code_generated=code
            )
        
        finally:
            # Clean up matplotlib figures
            plt.close('all')

# Create global instance
code_executor = CodeExecutor() 