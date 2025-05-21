import os
import uuid
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
from utils.logging import log_error, log_info


def format_data_table(df: pd.DataFrame, format_type: str = 'markdown', max_rows: int = 20, max_cols: int = 10) -> str:
    """
    Format a DataFrame as a table in various formats.
    
    Args:
        df: DataFrame to format
        format_type: Format type ('markdown', 'html', or 'json')
        max_rows: Maximum number of rows to include
        max_cols: Maximum number of columns to include
        
    Returns:
        Formatted table as a string
    """
    # Log the format request
    log_info(f"Formatting data table", {
        "format_type": format_type,
        "df_shape": str(df.shape) if df is not None else "None",
        "max_rows": max_rows,
        "max_cols": max_cols
    })
    
    # Handle empty DataFrame
    if df is None or len(df) == 0:
        if format_type == 'markdown':
            return "No data available."
        elif format_type == 'html':
            return "<p>No data available.</p>"
        elif format_type == 'json':
            return "[]"
        else:
            return "No data available."
    
    # Limit rows and columns if necessary
    if len(df) > max_rows:
        df = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
        df_truncated = True
    else:
        df_truncated = False
        
    if len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols]
        cols_truncated = True
    else:
        cols_truncated = False
    
    # Format based on type
    try:
        if format_type == 'markdown':
            try:
                # First attempt with tabulate
                result = df.to_markdown(index=True)
                log_info("Successfully formatted table with to_markdown")
            except ImportError as e:
                # Fallback if tabulate is not available
                log_error("Tabulate dependency missing", e, {
                    "format_type": format_type,
                    "fallback": "simple string representation"
                })
                # Create a simple string representation
                result = "Data Preview:\n\n"
                result += str(df)
                result += "\n\n(Note: Install 'tabulate' for better formatting)"
                
            if df_truncated:
                result += "\n\n*Note: Table truncated. Showing first and last rows.*"
            if cols_truncated:
                result += "\n\n*Note: Table truncated. Showing first few columns.*"
            return result
            
        elif format_type == 'html':
            result = df.to_html(index=True, classes='table table-striped')
            if df_truncated or cols_truncated:
                note = "<p><em>Note: "
                if df_truncated:
                    note += "Table truncated. Showing first and last rows."
                if cols_truncated:
                    if df_truncated:
                        note += " "
                    note += "Table truncated. Showing first few columns."
                note += "</em></p>"
                result += note
            return result
            
        elif format_type == 'json':
            result = df.to_json(orient='records')
            if df_truncated or cols_truncated:
                # Add a note in the JSON about truncation
                json_data = json.loads(result)
                if isinstance(json_data, list) and len(json_data) > 0:
                    json_data.append({
                        "_note": {
                            "truncated_rows": df_truncated,
                            "truncated_cols": cols_truncated,
                            "total_rows": len(df),
                            "total_cols": len(df.columns)
                        }
                    })
                result = json.dumps(json_data)
            return result
            
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    except Exception as e:
        # General error handling for any other exceptions
        log_error("Error formatting data table", e, {
            "format_type": format_type,
            "error_type": type(e).__name__
        })
        
        # Return a simple string representation as fallback
        return f"Data preview (Error in formatting: {str(e)}):\n\n{str(df.head(5))}\n\n[...additional rows omitted...]"


def format_result_for_display(result: Dict[str, Any], format_type: str = 'markdown') -> Dict[str, Any]:
    """
    Format a tool result for display to the user.
    
    Args:
        result: Tool execution result
        format_type: Format type ('markdown', 'html', or 'json')
        
    Returns:
        Dictionary with formatted display data
    """
    display_data = {
        'message': result.get('message', ''),
        'data_table': None,
        'plot_url': None
    }
    
    # Format DataFrame if present
    if 'result_df' in result and isinstance(result['result_df'], pd.DataFrame):
        display_data['data_table'] = format_data_table(result['result_df'], format_type)
    
    # Include plot URL if present
    if 'plot_url' in result:
        display_data['plot_url'] = result['plot_url']
    
    # Include metadata in a readable format
    if 'metadata' in result:
        metadata = result['metadata']
        if format_type == 'markdown':
            metadata_str = "\n\n**Metadata:**\n\n"
            for key, value in metadata.items():
                metadata_str += f"- **{key}**: {value}\n"
            display_data['message'] += metadata_str
        elif format_type == 'html':
            metadata_str = "<h4>Metadata:</h4><ul>"
            for key, value in metadata.items():
                metadata_str += f"<li><strong>{key}</strong>: {value}</li>"
            metadata_str += "</ul>"
            display_data['message'] += metadata_str
        elif format_type == 'json':
            if not display_data['message']:
                display_data['message'] = {}
            display_data['message']['metadata'] = metadata
    
    return display_data


def save_plot(fig, plot_dir: str = 'storage/plots') -> str:
    """
    Save a matplotlib figure to a file and return the URL.
    
    Args:
        fig: matplotlib figure to save
        plot_dir: Directory to save the plot in
        
    Returns:
        URL to the saved plot
    """
    # Create directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate a unique filename
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(plot_dir, filename)
    
    # Save the figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Return the URL
    return f"/plot/{filename}"


def parse_filter_criteria(criteria_str: str) -> list:
    """
    Parse filter criteria from a string format to a list of dictionaries.
    
    Args:
        criteria_str: String representation of filter criteria
        
    Returns:
        List of filter criteria dictionaries
    """
    # This is a simplified parser for demonstration
    # In a real application, you would use more robust parsing
    criteria = []
    
    # Split by 'AND' to get individual conditions
    conditions = criteria_str.split(' AND ')
    
    for condition in conditions:
        # Look for known operators
        for op in ['>=', '<=', '!=', '=', '>', '<']:
            if op in condition:
                parts = condition.split(op, 1)
                column = parts[0].strip()
                value = parts[1].strip()
                
                # Convert value to appropriate type if possible
                try:
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # Try to convert to numeric
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            # Keep as string
                            pass
                except:
                    # If any errors, keep as is
                    pass
                
                # Map operator to pandas-compatible operator
                if op == '=':
                    op = '=='
                
                criteria.append({
                    'column': column,
                    'operator': op,
                    'value': value
                })
                break
    
    return criteria


def get_file_summary(df, file_name, max_cols=10):
    """
    Get a human-readable summary of a DataFrame.
    
    Args:
        df: Pandas DataFrame to summarize
        file_name: Name of the file
        max_cols: Maximum number of column names to include
        
    Returns:
        String with file summary
    """
    if df is None:
        return f"File '{file_name}' does not have any data loaded."
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # List of column names (up to max_cols)
    column_names = df.columns.tolist()
    shown_cols = column_names[:max_cols]
    hidden_cols = total_cols - len(shown_cols)
    
    # Basic data info
    summary = f"File '{file_name}' contains {total_rows} rows and {total_cols} columns."
    
    # Add column names
    if shown_cols:
        if hidden_cols > 0:
            summary += f" Columns include: {', '.join(shown_cols)}, and {hidden_cols} more."
        else:
            summary += f" Columns: {', '.join(shown_cols)}."
    
    # Include basic statistics for numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        summary += f" Contains {len(num_cols)} numerical columns."
    
    # Include basic info for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        summary += f" Contains {len(cat_cols)} categorical columns."
    
    return summary


def make_serializable(obj: Any) -> Any:
    """
    Make an object JSON-serializable by converting pandas and numpy types.
    
    Args:
        obj: Object to make serializable
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return None
    else:
        return obj


# Custom JSON encoder for serializing response data
class DataFrameEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return super().default(obj)


def to_serializable_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a dictionary with potential pandas/numpy objects to JSON-serializable format.
    
    Args:
        data: Dictionary containing data to serialize
        
    Returns:
        JSON-serializable dictionary
    """
    try:
        # First try to convert any nested objects
        serializable_data = make_serializable(data)
        
        # Then use custom encoder to catch any remaining numpy types
        json_str = json.dumps(serializable_data, cls=DataFrameEncoder)
        return json.loads(json_str)
    except:
        # Fall back to best-effort conversion
        return make_serializable(data) 
