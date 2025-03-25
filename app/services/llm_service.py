import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

logger = logging.getLogger(__name__)
load_dotenv()

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"
        self.available_libraries = {
            "pandas": "Data manipulation and analysis",
            "numpy": "Numerical computing",
            "plotly": "Interactive visualizations",
            "scipy": "Scientific computing",
            "sklearn": "Machine learning and statistics"
        }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return f"""You are a data analysis expert. Your task is to generate Python code for data analysis and visualization.
        
Available libraries and their purposes:
{json.dumps(self.available_libraries, indent=2)}

Requirements:
1. Write a function named `analyze_data` that takes a file path as input
2. The function should return a tuple of (figure, data) where:
   - figure: A Plotly figure object for visualization
   - data: A dictionary containing analysis results
3. Use pandas for data manipulation and analysis
4. Use plotly for creating interactive visualizations
5. Handle errors gracefully and provide meaningful error messages
6. Include proper data type conversions and error handling
7. Return both visualization and data analysis results

Example function structure:
```python
def analyze_data(file_path: str) -> Tuple[Any, Dict[str, Any]]:
    # Your analysis code here
    return fig, data
```"""

    def _get_user_prompt(self, query: str, data_info: Dict[str, Any]) -> str:
        """Get the user prompt for the LLM."""
        # Extract metadata from data_info
        metadata = data_info.get('metadata', {})
        column_info = metadata.get('column_info', {})
        
        # Build column descriptions
        column_descriptions = []
        for col_name, col_info in column_info.items():
            dtype = col_info.get('dtype', 'unknown')
            sample_values = col_info.get('sample_values', [])
            sample_str = ', '.join(str(v) for v in sample_values[:3])
            column_descriptions.append(f"- {col_name} ({dtype}): Sample values: {sample_str}")
        
        # Build the prompt
        prompt = f"""Dataset Information:
- Total rows: {metadata.get('total_rows', 'unknown')}
- Columns: {', '.join(column_info.keys())}

Column Details:
{chr(10).join(column_descriptions)}

Please analyze the data based on this query:
{query}

   Generate Python code that:
    1. Reads the data using pandas
    2. Performs the requested analysis
    3. Creates visualizations if appropriate
    4. Returns both the figure and data in a tuple, if there is no figure and data, return None
    5. only return the code, no other text

    For data exploration queries, include:
    - Basic information about the dataset (shape, columns, data types)
    - Missing value analysis
    - Basic statistics for numeric columns
    - Value counts for categorical columns
    - Appropriate visualizations

    The code should be complete and ready to run. Do NOT include any explanations or markdown formatting."""

        return prompt

    async def generate_code(self, query: str, data_info: Dict[str, Any]) -> str:
        """Generate Python code for data analysis based on the query and data info."""
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._get_user_prompt(query, data_info)}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            generated_code = response.choices[0].message.content
            logger.info(f"Generated code: {generated_code}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            raise

    async def analyze_data(self, query: str, data_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Analyze data based on the query and return visualization and analysis results."""
        try:
            # Generate code
            generated_code = await self.generate_code(query, data_info)
            
            # Execute code and get results
            result = code_executor.execute_code(generated_code, data_info.get('file_path'))
            
            # Extract figure and data from result
            fig = result.get('plot')
            data = result.get('data', {})
            
            return fig, data
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
            raise

    async def analyze_data_old(self, file_path: str, query: str) -> Tuple[Any, Dict[str, Any]]:
        """Analyze the data based on the query and return results."""
        try:
            # Read the data to get information
            df = pd.read_csv(file_path)
            
            # For data exploration queries, return comprehensive analysis
            if query.lower().startswith(('explain', 'describe', 'show', 'tell')):
                # Create comprehensive data exploration result
                data = {
                    "basic_info": {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "missing_values": df.isnull().sum().to_dict()
                    },
                    "numeric_stats": df.describe().to_dict() if df.select_dtypes(include=[np.number]).columns.any() else None,
                    "categorical_stats": {
                        col: df[col].value_counts().to_dict()
                        for col in df.select_dtypes(include=['object', 'category']).columns
                    } if df.select_dtypes(include=['object', 'category']).columns.any() else None
                }
                
                # Create a summary visualization
                fig = None
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.box(df, y=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                
                return fig, data
            
            # For other queries, generate and execute code
            data_info = {
                "file_path": file_path,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head().to_dict()
            }
            
            # Generate code based on the query
            code = await self.generate_code(query, data_info)
            print(f"Generated code:\n{code}")
            
            # Create a local namespace for code execution
            local_vars = {
                'pd': pd,
                'np': np,
                'px': px,
                'go': go
            }
            
            # Execute the generated code
            exec(code, {}, local_vars)
            
            # Get the result from the analyze_data function
            result = local_vars.get('analyze_data')(file_path)
            
            # Ensure result is a tuple
            if not isinstance(result, tuple):
                result = (None, result)
            
            return result
            
        except Exception as e:
            print(f"Error in analyze_data: {str(e)}")
            raise 