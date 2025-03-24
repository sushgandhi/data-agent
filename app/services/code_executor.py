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

            # Clean up the code
            cleaned_code = self._clean_code(code)

            # Create a local namespace for code execution with all required imports
            local_vars = {
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'file_path': file_path
            }

            # Add imports to the code
            code_with_imports = f"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

{cleaned_code}
"""
            
            # Execute the code
            exec(code_with_imports, {}, local_vars)

            # Get the result from the analyze_data function
            result = local_vars.get('analyze_data')(file_path)

            # Ensure result is a tuple
            if not isinstance(result, tuple):
                result = (None, result)

            fig, data = result

            # Convert Plotly figure to JSON if it exists
            plot_data = None
            if fig is not None and hasattr(fig, 'to_json'):
                try:
                    plot_data = json.loads(fig.to_json())
                except Exception as e:
                    print(f"Error converting plot to JSON: {str(e)}")
                    plot_data = None

            # Convert numpy types to Python native types in data
            if data is not None:
                data = self._convert_numpy_types(data)

            return {
                "success": True,
                "result": {
                    "plot": plot_data,
                    "data": data
                }
            }

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
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