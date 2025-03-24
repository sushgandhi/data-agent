import os
from openai import AsyncOpenAI
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo-preview"

    def _get_system_prompt(self) -> str:
        return """You are a data analysis expert. Your task is to help users analyze their data by generating Python code.
        The code should be focused on data analysis and visualization using pandas, numpy, and plotly.
        
        Important rules:
        1. NEVER use plt.show() or fig.show() - instead return the figure and data as a tuple: (figure, data)
        2. Always use plotly for visualizations
        3. For descriptive statistics, use pandas describe() and other relevant methods
        4. For data exploration, use appropriate pandas methods like info(), columns, etc.
        5. Return both the figure (if any) and the data in a tuple format
        6. Handle missing values appropriately
        7. Use appropriate data types for analysis
        
        The code should be in a function named analyze_data that takes a file_path parameter.
        The function should return a tuple of (figure, data) where:
        - figure: A plotly figure object (or None if no visualization)
        - data: A dictionary containing the analysis results (or None if no data to return)
        """

    def _get_user_prompt(self, query: str, data_info: Dict[str, Any]) -> str:
        return f"""Please analyze the data based on this query: "{query}"

        Dataset Information:
        - File path: {data_info.get('file_path')}
        - Columns: {data_info.get('columns', [])}
        - Data types: {data_info.get('dtypes', {})}
        - Sample data: {data_info.get('sample_data', {})}

        Generate Python code that:
        1. Reads the data using pandas
        2. Performs the requested analysis
        3. Creates visualizations if appropriate
        4. Returns both the figure and data in a tuple

        The code should be complete and ready to run. Do not include any explanations or markdown formatting.
        """

    async def generate_code(self, query: str, data_info: Dict[str, Any]) -> str:
        """Generate Python code based on the user's query and data information."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._get_user_prompt(query, data_info)}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            raise

    async def analyze_data(self, file_path: str, query: str) -> Tuple[Any, Dict[str, Any]]:
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