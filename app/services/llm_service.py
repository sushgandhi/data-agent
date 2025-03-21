from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from ..core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            project=settings.OPENAI_PROJECT_ID
        )
        
    async def generate_code(self, query: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Generate Python code to answer the query using the dataset metadata."""
        try:
            # Log input parameters
            logger.info(f"Generating code for query: {query}")
            logger.info(f"Using metadata: {json.dumps(metadata, indent=2)}")

            # Construct a detailed system message with data context
            system_message = f"""You are a data analysis expert. You have access to a dataset with the following characteristics:

Filename: {metadata['filename']}
Total rows: {metadata['total_rows']}

Columns:
{self._format_columns(metadata['column_info'])}

Instructions:
1. Create a function named 'analyze_data' that takes 'file_path' as a parameter
2. Use pandas for data analysis (pd is already imported)
3. Create visualizations using plotly (px and go are already imported)
4. Handle null values appropriately
5. Include error handling
6. Return the code as a single string without markdown formatting
7. Use the exact column names from the metadata
8. Include data type information in your analysis
9. The function should return the analysis results
10. Do not include any import statements - they are already available

Example function structure:
def analyze_data(file_path):
    # Your code here
    return result

The code should be ready to execute and should return the results in a format that can be displayed in a web interface. 
No debug or print statements needed."""

            logger.info(f"System message: {system_message}")

            # Create the chat completion
            logger.info("Making API call to OpenAI...")
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Write Python code to answer this question: {query}"}
                ],
                max_completion_tokens=1000
            )
            logger.info(f"OpenAI API response: {response}")

            # Extract the generated code
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid response structure from OpenAI")
                
            generated_code = response.choices[0].message.content
            if not generated_code:
                raise ValueError("OpenAI returned empty response")
            
            # Clean up the code
            generated_code = generated_code.strip()
            
            # Log the generated code for debugging
            logger.info(f"Generated code:\n{generated_code}")
            
            return generated_code

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate code: {str(e)}")

    def _format_columns(self, column_info: Dict[str, Dict[str, Any]]) -> str:
        """Format column information for the system message."""
        formatted_columns = []
        for col_name, info in column_info.items():
            col_info = [
                f"- {col_name} ({info['type']})",
                f"  Sample values: {info['sample_values']}",
                f"  Has null values: {info['has_nulls']}"
            ]
            formatted_columns.extend(col_info)
        return "\n".join(formatted_columns)

    def _create_system_message(self, data_context: Dict[str, Any]) -> str:
        """Create a system message with data context for the LLM"""
        return f"""You are a Python code generator that helps analyze data files.
        
The data is in a file named '{data_context['filename']}'.
The file has the following columns: {', '.join(data_context['columns'])}

You have access to these Python libraries:
- pandas (as pd)
- numpy (as np)
- matplotlib.pyplot (as plt)
- plotly.express (as px)
- plotly.graph_objects (as go)

The data will be loaded into a pandas DataFrame named 'df'.
Write code that answers the user's question about this data.
Only include the code in your response, no explanations or markdown.
Make sure your code is complete and handles errors appropriately.""" 