import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any, Optional, Tuple
import time
import psutil
import os
from ..core.config import settings
from ..models.base import ExecutionResult
import logging
import json
import traceback
from io import StringIO
import sys

logger = logging.getLogger(__name__)

class CodeExecutor:
    def __init__(self):
        self.process = psutil.Process()
        self._captured_output = StringIO()
        self._original_stdout = sys.stdout
        self.required_libraries = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go
        }
        
    def _clean_code(self, code: str) -> str:
        """Clean the code by removing markdown formatting and ensuring proper imports."""
        # Remove markdown code block markers
        code = code.replace('```python', '').replace('```', '').strip()
        
        # Remove any existing imports (we'll add them back)
        lines = code.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
        
        # Add required imports at the top
        imports = [
            'import pandas as pd',
            'import numpy as np',
            'import plotly.express as px',
            'import plotly.graph_objects as go',
            'from typing import Dict, Any, Tuple'
        ]
        
        return '\n'.join(imports + [''] + cleaned_lines)

    def _handle_non_json_values(self, obj):
        """Handle non-JSON compliant values by converting them to strings."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._handle_non_json_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._handle_non_json_values(item) for item in obj]
        elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return str(obj)
        return obj
        
    def execute_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Execute the generated code and return the results."""
        try:
            # Reset captured output
            self._captured_output = StringIO()
            sys.stdout = self._captured_output

            # Clean the code
            cleaned_code = self._clean_code(code)
            logger.info(f"Cleaned code:\n{cleaned_code}")

            # Create both global and local namespaces with required libraries
            global_vars = self.required_libraries.copy()
            local_vars = self.required_libraries.copy()
            local_vars['file_path'] = file_path
            
            # Execute the code with both global and local namespaces
            exec(cleaned_code, global_vars, local_vars)
            
            # Get the analyze_data function
            analyze_data_func = local_vars.get('analyze_data')
            if not analyze_data_func:
                raise ValueError("No analyze_data function found in the generated code")
            
            # Execute the analyze_data function
            logger.info("Executing analyze_data function...")
            result = analyze_data_func(file_path)
            logger.info(f"Function result: {result}")
            
            # Ensure result is a tuple
            if not isinstance(result, tuple):
                result = (None, result)
            
            # Convert Plotly figure to JSON if it exists
            fig, data = result
            if fig is not None:
                try:
                    plot_data = fig.to_json()
                    logger.info(f"Converted plot data: {plot_data[:200]}...")
                except Exception as e:
                    logger.error(f"Error converting plot to JSON: {str(e)}")
                    plot_data = None
            else:
                plot_data = None
            
            # Convert numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            converted_data = convert_numpy_types(data)
            logger.info(f"Converted data: {json.dumps(converted_data, indent=2)}")
            
            return {
                "success": True,
                "result": {
                    "plot": plot_data,
                    "data": converted_data
                }
            }
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

        finally:
            # Restore original stdout
            sys.stdout = self._original_stdout
            # Clean up matplotlib figures
            plt.close('all')

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        return obj

# Create global instance
code_executor = CodeExecutor() 