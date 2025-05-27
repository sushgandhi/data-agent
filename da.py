import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union


#################################################
# HELPER FUNCTIONS
#################################################

def load_data(file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads data from a CSV or Excel file into a pandas DataFrame.
    
    Args:
        file_path: Path to the file to load
        sheet_name: For Excel files, name of the sheet to load (default: first sheet)
        
    Returns:
        Dictionary containing the loaded DataFrame and metadata about the file
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                'success': False, 
                'error': f"File not found: {file_path}"
            }
        
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Load based on file type
        if file_extension == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            file_type = 'csv'
            sheets = None
            sheet_loaded = None
        elif file_extension in ['.xlsx', '.xls']:
            # For Excel files
            excel_file = pd.ExcelFile(file_path)
            all_sheets = excel_file.sheet_names
            
            # If sheet_name is None, read the first sheet
            if sheet_name is None:
                sheet_name = all_sheets[0]
            elif sheet_name not in all_sheets:
                return {
                    'success': False,
                    'error': f"Sheet '{sheet_name}' not found in Excel file. Available sheets: {', '.join(all_sheets)}"
                }
            
            # Load the specified sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            file_type = 'excel'
            sheets = all_sheets
            sheet_loaded = sheet_name
        else:
            return {
                'success': False,
                'error': f"Unsupported file type: {file_extension}. Supported formats: .csv, .xlsx, .xls"
            }
        
        # Basic data cleaning and preparation
        # 1. Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # 2. Attempt to convert numeric columns stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric, but only if it doesn't introduce NaNs
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # If no NaNs were introduced, or the NaNs were already there, convert
                if numeric_series.isna().sum() == df[col].isna().sum():
                    df[col] = numeric_series
            except:
                # If any error occurs, keep the column as is
                pass
        
        # Prepare result with metadata
        result = {
            'success': True,
            'result_df': df,
            'message': f"File '{os.path.basename(file_path)}' loaded successfully with {len(df)} rows and {len(df.columns)} columns.",
            'metadata': {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_type': file_type,
                'file_size_bytes': os.path.getsize(file_path),
                'sheet_name': sheet_loaded,
                'available_sheets': sheets,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'column_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'has_missing_values': df.isna().any().any(),
                'missing_value_counts': df.isna().sum().to_dict()
            }
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error loading file '{os.path.basename(file_path) if file_path else 'unknown'}': {str(e)}"
        }


def save_plot(fig, format='png', dpi=150, plots_dir='plots'):
    """
    Save a matplotlib figure to a file and return the file path.
    
    Args:
        fig: The matplotlib figure to save
        format: The file format (default: 'png')
        dpi: The resolution in dots per inch (default: 150)
        plots_dir: Directory to save plots (default: 'plots')
        
    Returns:
        Path to the saved file
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate a unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.{format}"
    filepath = os.path.join(plots_dir, filename)
    
    # Save the figure with explicit tight layout
    plt.tight_layout()
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    
    # Return the path to the saved file
    return filepath


def find_actual_column(df, column_name):
    """Helper function to find the actual column name with case-insensitive matching"""
    if column_name in df.columns:
        return column_name
    
    # Try case-insensitive matching
    column_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == column_lower:
            return col
            
    return None


#################################################
# TOOL FUNCTIONS
#################################################

def filter_data(data_location: str, filter_criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters a pandas DataFrame based on multiple criteria.

    Args:
        data_location: Path to the data file
        filter_criteria: A list of dictionaries, where each dictionary
                         specifies a column, operator (==, >, <, >=, <=, !=),
                         and value for filtering.
                         Example: [{'column': 'Sales', 'operator': '>', 'value': 1000}]

    Returns:
        Dictionary containing the filtered DataFrame and metadata about the operation.
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']

    if not isinstance(filter_criteria, list):
        return {
            'success': False,
            'error': f"filter_criteria must be a list, got {type(filter_criteria).__name__}"
        }

    # Copy DataFrame to avoid modifying the original
    filtered_df = df.copy()
    original_row_count = len(filtered_df)
    
    # Keep track of how many rows were filtered at each step
    filter_steps = []
    
    # Apply each filter criterion
    for i, criterion in enumerate(filter_criteria):
        if not isinstance(criterion, dict):
            return {
                'success': False,
                'error': f"Filter criterion at index {i} must be a dictionary"
            }
        
        # Check required keys
        required_keys = ['column', 'operator', 'value']
        for key in required_keys:
            if key not in criterion:
                return {
                    'success': False,
                    'error': f"Filter criterion at index {i} missing required key: '{key}'"
                }
        
        # Extract filter components
        column = criterion['column']
        operator = criterion['operator']
        filter_value = criterion['value']
        
        # Find the actual column name with case-insensitive matching
        actual_column = find_actual_column(filtered_df, column)
        if actual_column is None:
            return {
                'success': False,
                'error': f"Column '{column}' not found in DataFrame. Available columns: {', '.join(filtered_df.columns)}"
            }
        column = actual_column
        
        # Validate operator
        valid_operators = ['==', '!=', '>', '<', '>=', '<=', 'contains', 'startswith', 'endswith', 'in', 'not in']
        if operator not in valid_operators:
            return {
                'success': False,
                'error': f"Invalid operator: '{operator}'. Valid operators are: {', '.join(valid_operators)}"
            }
        
        # Record pre-filter row count
        pre_filter_count = len(filtered_df)
        
        try:
            # Apply filter based on operator
            if operator == '==':
                filtered_df = filtered_df[filtered_df[column] == filter_value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[column] != filter_value]
            elif operator == '>':
                filtered_df = filtered_df[filtered_df[column] > filter_value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < filter_value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= filter_value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= filter_value]
            elif operator == 'contains':
                if not pd.api.types.is_string_dtype(filtered_df[column]):
                    return {
                        'success': False,
                        'error': f"'contains' operator can only be used with string columns. Column '{column}' is {filtered_df[column].dtype}"
                    }
                filtered_df = filtered_df[filtered_df[column].str.contains(str(filter_value), na=False)]
            elif operator == 'startswith':
                if not pd.api.types.is_string_dtype(filtered_df[column]):
                    return {
                        'success': False,
                        'error': f"'startswith' operator can only be used with string columns. Column '{column}' is {filtered_df[column].dtype}"
                    }
                filtered_df = filtered_df[filtered_df[column].str.startswith(str(filter_value), na=False)]
            elif operator == 'endswith':
                if not pd.api.types.is_string_dtype(filtered_df[column]):
                    return {
                        'success': False,
                        'error': f"'endswith' operator can only be used with string columns. Column '{column}' is {filtered_df[column].dtype}"
                    }
                filtered_df = filtered_df[filtered_df[column].str.endswith(str(filter_value), na=False)]
            elif operator == 'in':
                if not isinstance(filter_value, list):
                    return {
                        'success': False,
                        'error': f"'in' operator requires a list value. Got {type(filter_value).__name__}"
                    }
                filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            elif operator == 'not in':
                if not isinstance(filter_value, list):
                    return {
                        'success': False,
                        'error': f"'not in' operator requires a list value. Got {type(filter_value).__name__}"
                    }
                filtered_df = filtered_df[~filtered_df[column].isin(filter_value)]
            
            # Record filter results
            post_filter_count = len(filtered_df)
            rows_removed = pre_filter_count - post_filter_count
            
            filter_steps.append({
                'filter_idx': i,
                'column': column,
                'operator': operator,
                'value': filter_value,
                'rows_before': pre_filter_count,
                'rows_after': post_filter_count,
                'rows_removed': rows_removed,
                'percent_removed': round((rows_removed / pre_filter_count * 100), 2) if pre_filter_count > 0 else 0
            })
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error applying filter at index {i} on column '{column}': {str(e)}"
            }
    
    # Prepare result
    result = {
        'success': True,
        'result_df': filtered_df,
        'message': f"Filtered data from {original_row_count} to {len(filtered_df)} rows ({original_row_count - len(filtered_df)} rows removed).",
        'metadata': {
            'original_row_count': original_row_count,
            'filtered_row_count': len(filtered_df),
            'rows_removed': original_row_count - len(filtered_df),
            'percent_removed': round(((original_row_count - len(filtered_df)) / original_row_count * 100), 2) if original_row_count > 0 else 0,
            'filter_criteria': filter_criteria,
            'filter_steps': filter_steps
        }
    }
    
    return result 

def group_and_aggregate(data_location: str, group_by_cols: List[str], agg_definitions: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Groups a DataFrame by specified columns and performs aggregation operations.
    If group_by_cols is empty, performs global aggregation across the entire dataset.
    
    Args:
        data_location: Path to the data file
        group_by_cols: List of column names to group by (can be empty for global aggregation)
        agg_definitions: List of aggregation definitions, each with column, function, and optional new_column_name
            Example: [{"column": "Sales", "function": "sum", "new_column_name": "Total Sales"}]
            
    Returns:
        Dictionary containing the aggregated DataFrame and metadata
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    if not isinstance(group_by_cols, list):
        return {
            'success': False,
            'error': "group_by_cols must be a list of column names (can be empty for global aggregation)"
        }
    
    if not isinstance(agg_definitions, list) or len(agg_definitions) == 0:
        return {
            'success': False,
            'error': "agg_definitions must be a non-empty list of aggregation definitions"
        }
    
    # Validate all group_by columns exist in the DataFrame
    for col in group_by_cols:
        actual_col = find_actual_column(df, col)
        if actual_col is None:
            return {
                'success': False,
                'error': f"Group by column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
    
    # Prepare for aggregation
    agg_dict = {}
    agg_columns = []
    
    for i, agg_def in enumerate(agg_definitions):
        # Check required keys
        if not isinstance(agg_def, dict):
            return {
                'success': False,
                'error': f"Aggregation definition at index {i} must be a dictionary"
            }
        
        if "column" not in agg_def or "function" not in agg_def:
            return {
                'success': False,
                'error': f"Aggregation definition at index {i} missing required keys 'column' and/or 'function'"
            }
        
        col = agg_def["column"]
        func = agg_def["function"]
        
        # Find the actual column name with case-insensitive matching
        actual_col = find_actual_column(df, col)
        if actual_col is None:
            return {
                'success': False,
                'error': f"Aggregation column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
        col = actual_col
        
        # Validate aggregation function
        supported_functions = ["sum", "mean", "count", "min", "max", "std", "median", "first", "last"]
        if func not in supported_functions:
            return {
                'success': False,
                'error': f"Unsupported aggregation function: '{func}'. Supported functions: {', '.join(supported_functions)}"
            }
        
        # Track column for later checking and build aggregation dictionary
        agg_columns.append(col)
        
        # If no previous aggregation for this column, initialize with a list
        if col not in agg_dict:
            agg_dict[col] = []
        
        # Add the function to the list for this column
        agg_dict[col].append(func)
    
    try:
        # Check if at least one aggregation column was specified
        if not agg_columns:
            return {
                'success': False,
                'error': "No valid aggregation columns specified"
            }
        
        # Perform the aggregation
        if group_by_cols:
            # Group by the specified columns
            grouped = df.groupby(group_by_cols)
            
            # Aggregate using the specified functions
            agg_result = grouped.agg(agg_dict)
            
            # Flatten multi-level column index if necessary
            if isinstance(agg_result.columns, pd.MultiIndex):
                # Rename columns to more descriptive names
                new_columns = []
                for col in agg_result.columns:
                    if col[1] == 'count':
                        new_name = f"Count of {col[0]}"
                    else:
                        new_name = f"{col[1].capitalize()} of {col[0]}"
                    new_columns.append(new_name)
                
                agg_result.columns = new_columns
            
            # Reset index to convert group columns to regular columns
            agg_result = agg_result.reset_index()
            
        else:
            # Global aggregation (no grouping)
            global_agg = {}
            
            for col, funcs in agg_dict.items():
                for func in funcs:
                    if func == 'sum':
                        global_agg[f"Sum of {col}"] = df[col].sum()
                    elif func == 'mean':
                        global_agg[f"Mean of {col}"] = df[col].mean()
                    elif func == 'count':
                        global_agg[f"Count of {col}"] = df[col].count()
                    elif func == 'min':
                        global_agg[f"Min of {col}"] = df[col].min()
                    elif func == 'max':
                        global_agg[f"Max of {col}"] = df[col].max()
                    elif func == 'std':
                        global_agg[f"Std Dev of {col}"] = df[col].std()
                    elif func == 'median':
                        global_agg[f"Median of {col}"] = df[col].median()
                    elif func == 'first':
                        global_agg[f"First of {col}"] = df[col].iloc[0] if len(df) > 0 else None
                    elif func == 'last':
                        global_agg[f"Last of {col}"] = df[col].iloc[-1] if len(df) > 0 else None
            
            # Convert to DataFrame for consistent return type
            agg_result = pd.DataFrame([global_agg])
        
        # Prepare result
        if group_by_cols:
            message = f"Grouped data by {', '.join(group_by_cols)} and calculated {len(agg_columns)} aggregations."
        else:
            message = f"Calculated global aggregations for {len(agg_columns)} columns."
        
        result = {
            'success': True,
            'result_df': agg_result,
            'message': message,
            'metadata': {
                'original_row_count': len(df),
                'aggregated_row_count': len(agg_result),
                'group_by_columns': group_by_cols,
                'aggregated_columns': agg_columns,
                'aggregation_functions': agg_dict,
                'aggregation_type': 'grouped' if group_by_cols else 'global'
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error performing aggregation: {str(e)}"
        } 

def create_pivot_table(data_location: str, index: List[str], values: List[str], 
                      columns: Optional[List[str]] = None, agg_func: str = "mean") -> Dict[str, Any]:
    """
    Creates a pivot table from a DataFrame.
    
    Args:
        data_location: Path to the data file
        index: Column names to use as the pivot table index (rows)
        values: Column names to use for the values in the pivot table
        columns: Column names to use for the pivot table columns (optional)
        agg_func: Aggregation function to apply (default: "mean")
        
    Returns:
        Dictionary containing the pivot table DataFrame and metadata
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    if not isinstance(index, list) or len(index) == 0:
        return {
            'success': False,
            'error': "index must be a non-empty list of column names"
        }
    
    # Validate all index columns exist in the DataFrame
    for col in index:
        actual_col = find_actual_column(df, col)
        if actual_col is None:
            return {
                'success': False,
                'error': f"Index column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
    
    if not isinstance(values, list) or len(values) == 0:
        return {
            'success': False,
            'error': "values must be a non-empty list of column names"
        }
    
    # Validate all values columns exist in the DataFrame
    for col in values:
        actual_col = find_actual_column(df, col)
        if actual_col is None:
            return {
                'success': False,
                'error': f"Values column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
    
    # Validate columns parameter if provided
    if columns is not None:
        if not isinstance(columns, list):
            return {
                'success': False,
                'error': "columns must be a list of column names"
            }
        
        for col in columns:
            actual_col = find_actual_column(df, col)
            if actual_col is None:
                return {
                    'success': False,
                    'error': f"Column column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
                }
    
    # Validate aggregation function
    supported_agg_funcs = ["mean", "sum", "count", "min", "max", "std", "median", "first", "last"]
    if agg_func not in supported_agg_funcs:
        return {
            'success': False,
            'error': f"Unsupported aggregation function: '{agg_func}'. Supported functions: {', '.join(supported_agg_funcs)}"
        }
    
    try:
        # Create the pivot table
        pivot_table = pd.pivot_table(
            df,
            index=index,
            values=values,
            columns=columns,
            aggfunc=agg_func
        )
        
        # Reset the index to convert index columns to regular columns
        pivot_table = pivot_table.reset_index()
        
        # Flatten column names if they are in a multi-index
        if isinstance(pivot_table.columns, pd.MultiIndex):
            # Create new column names
            new_columns = []
            for col in pivot_table.columns:
                if col[0] in index:
                    # For index columns, just use the original name
                    new_columns.append(col[0])
                elif col[1] is None or pd.isna(col[1]):
                    # For values without a column category
                    new_columns.append(f"{agg_func.capitalize()} of {col[0]}")
                else:
                    # For values with a column category
                    new_columns.append(f"{agg_func.capitalize()} of {col[0]} for {col[1]}")
            
            pivot_table.columns = new_columns
        
        # Prepare result
        column_desc = f" by {', '.join(columns)}" if columns else ""
        message = f"Created pivot table with {', '.join(index)} as index and {', '.join(values)} as values{column_desc} using {agg_func} aggregation."
        
        result = {
            'success': True,
            'result_df': pivot_table,
            'message': message,
            'metadata': {
                'original_row_count': len(df),
                'pivot_row_count': len(pivot_table),
                'index_columns': index,
                'values_columns': values,
                'columns_used': columns,
                'aggregation_function': agg_func
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error creating pivot table: {str(e)}"
        } 

def visualize_data(
    data_location: str, 
    plot_type: str, 
    x: str, 
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[List[float]] = None,
    bins: Optional[int] = None,
    orientation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a visualization of the data.
    
    Args:
        data_location: Path to the data file
        plot_type: Type of plot to generate (bar, line, scatter, histogram, boxplot, pie, etc.)
        x: Column to use for the x-axis (or main data for histogram/pie)
        y: Column to use for the y-axis (not required for histogram, pie)
        color: Column to use for color differentiation (optional)
        title: Title for the plot (optional)
        figsize: Figure size as [width, height] in inches (optional)
        bins: Number of bins for histogram (optional)
        orientation: Orientation of the plot (horizontal, vertical) (optional)
        
    Returns:
        Dictionary containing the plot URL and metadata
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    if not plot_type:
        return {
            'success': False,
            'error': "plot_type is required"
        }
    
    # Validate plot type
    supported_plot_types = [
        "bar", "line", "scatter", "histogram", "hist", "boxplot", "box", 
        "pie", "heatmap", "area", "violin", "kde", "density"
    ]
    
    if plot_type.lower() not in supported_plot_types:
        return {
            'success': False,
            'error': f"Unsupported plot type: '{plot_type}'. Supported types: {', '.join(supported_plot_types)}"
        }
    
    # Standardize plot type names
    if plot_type.lower() == "hist":
        plot_type = "histogram"
    elif plot_type.lower() == "box":
        plot_type = "boxplot"
    elif plot_type.lower() == "density":
        plot_type = "kde"
    
    # Validate x column
    if not x:
        return {
            'success': False,
            'error': "x column is required"
        }
    
    actual_x = find_actual_column(df, x)
    if actual_x is None:
        return {
            'success': False,
            'error': f"Column '{x}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    x = actual_x
    
    # Validate y column if required for the plot type
    y_required_plots = ["bar", "line", "scatter", "boxplot", "area"]
    if plot_type.lower() in y_required_plots and not y:
        return {
            'success': False,
            'error': f"y column is required for {plot_type} plots"
        }
    
    if y:
        actual_y = find_actual_column(df, y)
        if actual_y is None:
            return {
                'success': False,
                'error': f"Column '{y}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
        y = actual_y
    
    # Validate color column if provided
    if color:
        actual_color = find_actual_column(df, color)
        if actual_color is None:
            return {
                'success': False,
                'error': f"Color column '{color}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
        color = actual_color
    
    # Validate numeric columns for certain plot types
    numeric_required_plots = ["scatter", "line", "histogram", "kde", "area"]
    
    if plot_type.lower() in numeric_required_plots:
        # Check if x is numeric
        if not pd.api.types.is_numeric_dtype(df[x]):
            return {
                'success': False,
                'error': f"Column '{x}' must be numeric for {plot_type} plots"
            }
        
        # Check if y is numeric (if provided and required)
        if y and plot_type.lower() != "histogram" and not pd.api.types.is_numeric_dtype(df[y]):
            return {
                'success': False,
                'error': f"Column '{y}' must be numeric for {plot_type} plots"
            }
    
    # Validate orientation if provided
    if orientation and orientation.lower() not in ["horizontal", "vertical", "h", "v"]:
        return {
            'success': False,
            'error': "orientation must be 'horizontal' or 'vertical'"
        }
    
    # Normalize orientation
    if orientation:
        orientation = "h" if orientation.lower() in ["horizontal", "h"] else "v"
    
    try:
        # Set aesthetic style
        sns.set_style("whitegrid")
        
        # Create figure with specified size or default
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate the appropriate plot based on type
        if plot_type.lower() == "bar":
            if color:
                # Create grouped bar chart
                pivot_df = df.pivot_table(index=x, columns=color, values=y, aggfunc='mean')
                pivot_df.plot(kind='bar', ax=ax)
            else:
                # Create regular bar chart
                if orientation == "h":
                    sns.barplot(x=y, y=x, data=df, ax=ax)
                else:
                    sns.barplot(x=x, y=y, data=df, ax=ax)
        
        elif plot_type.lower() == "line":
            if color:
                # Create multi-line chart
                for val in df[color].unique():
                    subset = df[df[color] == val]
                    ax.plot(subset[x], subset[y], label=val)
                ax.legend()
            else:
                # Create simple line chart
                ax.plot(df[x], df[y])
        
        elif plot_type.lower() == "scatter":
            # Create scatter plot
            sns.scatterplot(x=x, y=y, hue=color, data=df, ax=ax)
        
        elif plot_type.lower() == "histogram":
            # Create histogram
            sns.histplot(data=df, x=x, bins=bins or 10, ax=ax)
        
        elif plot_type.lower() == "boxplot":
            # Create boxplot
            if orientation == "h":
                sns.boxplot(x=y, y=x, hue=color, data=df, ax=ax)
            else:
                sns.boxplot(x=x, y=y, hue=color, data=df, ax=ax)
        
        elif plot_type.lower() == "pie":
            # Create pie chart - need to aggregate data first
            pie_data = df[x].value_counts()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        elif plot_type.lower() == "heatmap":
            # For heatmap, we need a pivot table or correlation matrix
            if y and color:
                # Create pivot table heatmap
                pivot_df = df.pivot_table(index=x, columns=y, values=color, aggfunc='mean')
                sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", ax=ax)
            else:
                # Create correlation heatmap of numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                correlation = numeric_df.corr()
                sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
        
        elif plot_type.lower() == "area":
            # Create area plot
            df.plot.area(x=x, y=y, ax=ax)
        
        elif plot_type.lower() == "violin":
            # Create violin plot
            if orientation == "h":
                sns.violinplot(x=y, y=x, hue=color, data=df, ax=ax)
            else:
                sns.violinplot(x=x, y=y, hue=color, data=df, ax=ax)
        
        elif plot_type.lower() == "kde":
            # Create KDE plot
            sns.kdeplot(data=df, x=x, y=y, ax=ax)
        
        # Set plot title if provided
        if title:
            ax.set_title(title)
        else:
            # Generate a default title
            if y:
                ax.set_title(f"{plot_type.capitalize()} of {y} by {x}")
            else:
                ax.set_title(f"{plot_type.capitalize()} of {x}")
        
        # Set axis labels
        ax.set_xlabel(x)
        if y:
            ax.set_ylabel(y)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot and get URL
        plot_url = save_plot(fig)
        
        # Prepare result
        result = {
            'success': True,
            'message': f"{plot_type.capitalize()} plot created successfully.",
            'plot_url': plot_url,
            'metadata': {
                'plot_type': plot_type,
                'x_column': x,
                'y_column': y,
                'color_column': color,
                'title': title or f"{plot_type.capitalize()} of {y or x}",
                'rows_used': len(df)
            }
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error creating visualization: {str(e)}"
        } 

def summarize_column(data_location: str, column: str) -> Dict[str, Any]:
    """
    Provides descriptive statistics and information about a single column in a DataFrame.
    
    Args:
        data_location: Path to the data file
        column: Name of the column to summarize
        
    Returns:
        Dictionary containing summary information about the column
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    # Case-insensitive column name matching
    actual_column = find_actual_column(df, column)
    if actual_column is None:
        return {
            'success': False,
            'error': f"Column '{column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    column = actual_column
    
    try:
        # Extract the column
        series = df[column]
        
        # Basic information
        summary = {
            'column_name': column,
            'data_type': str(series.dtype),
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isna().sum(),
            'null_percent': round((series.isna().sum() / len(series) * 100), 2),
            'unique_values': series.nunique(),
            'memory_usage': series.memory_usage(deep=True)
        }
        
        # Different stats based on dtype
        if pd.api.types.is_numeric_dtype(series):
            # For numeric columns
            summary['numeric_stats'] = {
                'mean': series.mean() if not all(series.isna()) else None,
                'median': series.median() if not all(series.isna()) else None,
                'std_dev': series.std() if not all(series.isna()) else None,
                'min': series.min() if not all(series.isna()) else None,
                'max': series.max() if not all(series.isna()) else None,
                'quantiles': {
                    '25%': series.quantile(0.25) if not all(series.isna()) else None,
                    '50%': series.quantile(0.50) if not all(series.isna()) else None,
                    '75%': series.quantile(0.75) if not all(series.isna()) else None,
                },
                'skew': series.skew() if not all(series.isna()) else None,
                'kurtosis': series.kurtosis() if not all(series.isna()) else None
            }
            
            # Check if column appears to be a boolean encoded as numeric
            if series.dropna().isin([0, 1]).all():
                summary['appears_boolean'] = True
                value_counts = series.value_counts().to_dict()
                summary['value_distribution'] = {
                    'count_0': value_counts.get(0, 0),
                    'count_1': value_counts.get(1, 0),
                    'percentage_0': round((value_counts.get(0, 0) / series.count() * 100), 2) if series.count() > 0 else 0,
                    'percentage_1': round((value_counts.get(1, 0) / series.count() * 100), 2) if series.count() > 0 else 0
                }
            
        elif pd.api.types.is_datetime64_dtype(series):
            # For datetime columns
            if not all(series.isna()):
                summary['datetime_stats'] = {
                    'min_date': series.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'max_date': series.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'range_days': (series.max() - series.min()).days
                }
            else:
                summary['datetime_stats'] = {
                    'min_date': None,
                    'max_date': None,
                    'range_days': None
                }
        
        else:
            # For categorical/text columns
            if series.nunique() <= 20:  # Only for columns with a reasonable number of categories
                value_counts = series.value_counts().head(10).to_dict()  # Top 10 values
                top_values = {}
                
                for val, count in value_counts.items():
                    if pd.isna(val):
                        key = "NA"
                    elif val == "":
                        key = "(empty string)"
                    else:
                        key = str(val)
                    
                    top_values[key] = {
                        'count': count,
                        'percentage': round((count / series.count() * 100), 2) if series.count() > 0 else 0
                    }
                
                summary['top_values'] = top_values
            
            if pd.api.types.is_string_dtype(series):
                # For string columns, add string-specific stats
                non_empty_strings = series.dropna().astype(str).str.strip()
                if len(non_empty_strings) > 0:
                    summary['string_stats'] = {
                        'mean_length': round(non_empty_strings.str.len().mean(), 2),
                        'max_length': non_empty_strings.str.len().max(),
                        'min_length': non_empty_strings.str.len().min(),
                        'empty_count': (non_empty_strings == '').sum(),
                        'contains_numbers': non_empty_strings.str.contains(r'\d').any(),
                        'contains_letters': non_empty_strings.str.contains(r'[a-zA-Z]').any()
                    }
        
        # Prepare summary DataFrame for display
        summary_data = []
        summary_data.append({'Metric': 'Data Type', 'Value': summary['data_type']})
        summary_data.append({'Metric': 'Count', 'Value': summary['count']})
        summary_data.append({'Metric': 'Non-Null Count', 'Value': summary['non_null_count']})
        summary_data.append({'Metric': 'Null Count', 'Value': summary['null_count']})
        summary_data.append({'Metric': 'Null Percent', 'Value': f"{summary['null_percent']}%"})
        summary_data.append({'Metric': 'Unique Values', 'Value': summary['unique_values']})
        
        if 'numeric_stats' in summary:
            for metric, value in summary['numeric_stats'].items():
                if metric == 'quantiles':
                    for q, q_val in value.items():
                        summary_data.append({'Metric': f"Quantile {q}", 'Value': q_val})
                else:
                    summary_data.append({'Metric': metric.replace('_', ' ').title(), 'Value': value})
        
        if 'datetime_stats' in summary:
            for metric, value in summary['datetime_stats'].items():
                summary_data.append({'Metric': metric.replace('_', ' ').title(), 'Value': value})
        
        if 'string_stats' in summary:
            for metric, value in summary['string_stats'].items():
                summary_data.append({'Metric': metric.replace('_', ' ').title(), 'Value': value})
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create a second DataFrame for value distribution if available
        value_distribution_df = None
        if 'top_values' in summary:
            value_data = []
            for value, stats in summary['top_values'].items():
                value_data.append({
                    'Value': value,
                    'Count': stats['count'],
                    'Percentage': f"{stats['percentage']}%"
                })
            value_distribution_df = pd.DataFrame(value_data)
        
        # Prepare result
        result = {
            'success': True,
            'result_df': summary_df,
            'message': f"Summary for column '{column}'",
            'metadata': summary
        }
        
        # Add value distribution DataFrame if available
        if value_distribution_df is not None:
            result['value_distribution_df'] = value_distribution_df
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error summarizing column: {str(e)}"
        }


def summarize_sheet(data_location: str) -> Dict[str, Any]:
    """
    Provides overall statistics and information about a DataFrame.
    
    Args:
        data_location: Path to the data file
        
    Returns:
        Dictionary containing summary information about the DataFrame
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    try:
        # Basic DataFrame info
        summary = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'memory_usage_bytes': df.memory_usage(deep=True).sum(),
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isna().sum().to_dict(),
            'null_percentage': {col: round((null_count / len(df) * 100), 2) if len(df) > 0 else 0
                              for col, null_count in df.isna().sum().to_dict().items()},
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': round((df.duplicated().sum() / len(df) * 100), 2) if len(df) > 0 else 0
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe().to_dict()
            # Round numeric values for readability
            for col, stats in numeric_stats.items():
                numeric_stats[col] = {k: round(v, 4) if isinstance(v, (float, np.float64, np.float32)) else v 
                                     for k, v in stats.items()}
            summary['numeric_columns'] = numeric_cols
            summary['numeric_stats'] = numeric_stats
        
        # Categorical column statistics
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_stats = {}
            for col in cat_cols:
                cat_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head(5).to_dict() if df[col].nunique() < 50 else "Too many values to display"
                }
            summary['categorical_columns'] = cat_cols
            summary['categorical_stats'] = cat_stats
        
        # Datetime column detection
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        if date_cols:
            date_stats = {}
            for col in date_cols:
                if not all(df[col].isna()):
                    date_stats[col] = {
                        'min_date': df[col].min().strftime('%Y-%m-%d %H:%M:%S'),
                        'max_date': df[col].max().strftime('%Y-%m-%d %H:%M:%S'),
                        'range_days': (df[col].max() - df[col].min()).days
                    }
                else:
                    date_stats[col] = {
                        'min_date': None,
                        'max_date': None,
                        'range_days': None
                    }
            summary['datetime_columns'] = date_cols
            summary['datetime_stats'] = date_stats
        
        # Prepare DataFrame for display
        info_data = []
        
        # Add basic info
        info_data.append({'Metric': 'Rows', 'Value': summary['shape']['rows']})
        info_data.append({'Metric': 'Columns', 'Value': summary['shape']['columns']})
        info_data.append({'Metric': 'Memory Usage', 'Value': f"{summary['memory_usage_bytes'] / (1024*1024):.2f} MB"})
        info_data.append({'Metric': 'Duplicate Rows', 'Value': summary['duplicate_rows']})
        info_data.append({'Metric': 'Duplicate Percentage', 'Value': f"{summary['duplicate_percentage']}%"})
        
        # Add column types
        info_data.append({'Metric': 'Numeric Columns', 'Value': len(numeric_cols) if 'numeric_columns' in summary else 0})
        info_data.append({'Metric': 'Categorical Columns', 'Value': len(cat_cols) if 'categorical_columns' in summary else 0})
        info_data.append({'Metric': 'Datetime Columns', 'Value': len(date_cols) if 'datetime_columns' in summary else 0})
        
        # Add null statistics
        total_nulls = sum(summary['null_counts'].values())
        info_data.append({'Metric': 'Total Null Values', 'Value': total_nulls})
        info_data.append({'Metric': 'Percentage Null', 'Value': f"{(total_nulls / (len(df) * len(df.columns)) * 100):.2f}%" if len(df) * len(df.columns) > 0 else "0.00%"})
        
        # Create DataFrame for display
        info_df = pd.DataFrame(info_data)
        
        # Prepare result
        result = {
            'success': True,
            'result_df': info_df,
            'message': f"DataFrame summary: {len(df)} rows × {len(df.columns)} columns",
            'metadata': summary
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error summarizing sheet: {str(e)}"
        } 

def compare_columns(data_location: str, col1: str, col2: str, comparison_type: str) -> Dict[str, Any]:
    """
    Compares two columns in a DataFrame.
    
    Args:
        data_location: Path to the data file
        col1: Name of the first column to compare
        col2: Name of the second column to compare
        comparison_type: Type of comparison to perform:
            - 'correlation': Calculate correlation coefficient
            - 'difference': Calculate differences between values
            - 'distribution': Compare distributions visually
            - 'value_comparison': Compare values side by side
        
    Returns:
        Dictionary containing comparison results and optional visualization
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    file_info = loaded_data['metadata']
    
    # Validate columns exist
    actual_col1 = find_actual_column(df, col1)
    if actual_col1 is None:
        return {
            'success': False,
            'error': f"Column '{col1}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    col1 = actual_col1
    
    actual_col2 = find_actual_column(df, col2)
    if actual_col2 is None:
        return {
            'success': False,
            'error': f"Column '{col2}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    col2 = actual_col2
    
    # Validate comparison type
    supported_comparison_types = ["correlation", "difference", "distribution", "value_comparison"]
    if comparison_type.lower() not in supported_comparison_types:
        return {
            'success': False,
            'error': f"Unsupported comparison type: '{comparison_type}'. Supported types: {', '.join(supported_comparison_types)}"
        }
    
    try:
        # Prepare result dictionary
        result = {
            'success': True,
            'message': f"Comparison of '{col1}' and '{col2}' columns",
            'metadata': {
                'comparison_type': comparison_type,
                'column1': col1,
                'column2': col2
            }
        }
        
        # Perform comparison based on type
        if comparison_type.lower() == "correlation":
            # Check if both columns are numeric
            if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                return {
                    'success': False,
                    'error': f"Correlation requires numeric columns. '{col1}' is {df[col1].dtype} and '{col2}' is {df[col2].dtype}"
                }
            
            # Calculate correlation
            correlation = df[col1].corr(df[col2])
            
            # Create scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=df[col1], y=df[col2], ax=ax)
            ax.set_title(f"Correlation between {col1} and {col2}: {correlation:.4f}")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            plt.tight_layout()
            
            # Save plot
            plot_url = save_plot(fig)
            
            # Prepare result
            result['plot_url'] = plot_url
            result['metadata']['correlation'] = correlation
            result['result_df'] = pd.DataFrame([{'Column 1': col1, 'Column 2': col2, 'Correlation': correlation}])
            
        elif comparison_type.lower() == "difference":
            # Check if both columns are numeric
            if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                return {
                    'success': False,
                    'error': f"Difference calculation requires numeric columns. '{col1}' is {df[col1].dtype} and '{col2}' is {df[col2].dtype}"
                }
            
            # Calculate differences for numeric columns
            difference = df[col1] - df[col2]
            
            # Calculate statistics
            mean_diff = difference.mean()
            median_diff = difference.median()
            std_diff = difference.std()
            min_diff = difference.min()
            max_diff = difference.max()
            
            # Create histogram of differences
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(difference, kde=True, ax=ax)
            ax.set_title(f"Histogram of differences ({col1} - {col2})")
            ax.axvline(x=0, color='red', linestyle='--')
            plt.tight_layout()
            
            # Save plot
            plot_url = save_plot(fig)
            
            # Prepare result
            result['plot_url'] = plot_url
            result['metadata'].update({
                'mean_difference': round(mean_diff, 4),
                'median_difference': round(median_diff, 4),
                'std_difference': round(std_diff, 4),
                'min_difference': round(min_diff, 4),
                'max_difference': round(max_diff, 4)
            })
            
            # Create result DataFrame with differences
            diff_df = pd.DataFrame({
                col1: df[col1],
                col2: df[col2],
                'Difference': difference
            })
            result['result_df'] = diff_df
            
        elif comparison_type.lower() == "distribution":
            # Create distribution plots
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Adjust plot based on data types
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # For numeric columns, use KDE plots
                sns.kdeplot(df[col1], ax=ax, label=col1)
                sns.kdeplot(df[col2], ax=ax, label=col2)
                ax.set_title(f"Distribution Comparison: {col1} vs {col2}")
                ax.legend()
            elif pd.api.types.is_numeric_dtype(df[col1]) or pd.api.types.is_numeric_dtype(df[col2]):
                # If only one is numeric, create a box plot
                df_melt = pd.melt(df[[col1, col2]], var_name='Column', value_name='Value')
                try:
                    sns.boxplot(x='Column', y='Value', data=df_melt, ax=ax)
                    ax.set_title(f"Distribution Comparison: {col1} vs {col2}")
                except:
                    ax.text(0.5, 0.5, "Cannot create distribution comparison due to incompatible data types", 
                          horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title("Distribution Comparison Error")
            else:
                # For categorical columns, use count plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Limit to top 10 categories for each
                col1_counts = df[col1].value_counts().head(10)
                col2_counts = df[col2].value_counts().head(10)
                
                # Create count plots
                sns.barplot(x=col1_counts.index, y=col1_counts.values, ax=ax1)
                sns.barplot(x=col2_counts.index, y=col2_counts.values, ax=ax2)
                
                # Set titles and rotate labels
                ax1.set_title(f"Top 10 values in {col1}")
                ax2.set_title(f"Top 10 values in {col2}")
                ax1.tick_params(axis='x', rotation=45)
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_url = save_plot(fig)
            
            # Prepare result
            result['plot_url'] = plot_url
            
            # Create summary statistics for both columns
            summary_data = []
            
            # Add count, mean, std, etc. for numeric columns
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                for stat, name in [
                    ('count', 'Count'), 
                    ('mean', 'Mean'), 
                    ('std', 'Std Dev'), 
                    ('min', 'Min'), 
                    ('max', 'Max')
                ]:
                    summary_data.append({
                        'Statistic': name,
                        col1: getattr(df[col1], stat)() if hasattr(df[col1], stat) else None,
                        col2: getattr(df[col2], stat)() if hasattr(df[col2], stat) else None
                    })
            else:
                # For categorical columns, add unique count and most common
                summary_data.append({
                    'Statistic': 'Count',
                    col1: df[col1].count(),
                    col2: df[col2].count()
                })
                
                summary_data.append({
                    'Statistic': 'Unique Values',
                    col1: df[col1].nunique(),
                    col2: df[col2].nunique()
                })
                
                # Most common value
                most_common1 = df[col1].value_counts().index[0] if len(df[col1].dropna()) > 0 else None
                most_common2 = df[col2].value_counts().index[0] if len(df[col2].dropna()) > 0 else None
                
                summary_data.append({
                    'Statistic': 'Most Common',
                    col1: most_common1,
                    col2: most_common2
                })
            
            # Add null counts
            summary_data.append({
                'Statistic': 'Null Count',
                col1: df[col1].isna().sum(),
                col2: df[col2].isna().sum()
            })
            
            result['result_df'] = pd.DataFrame(summary_data)
            
        elif comparison_type.lower() == "value_comparison":
            # Create a side-by-side comparison of values
            comparison_df = df[[col1, col2]].copy()
            
            # Add a column indicating if values are equal
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                comparison_df['Equal'] = df[col1] == df[col2]
                comparison_df['Difference'] = df[col1] - df[col2]
            else:
                comparison_df['Equal'] = df[col1].astype(str) == df[col2].astype(str)
            
            # Calculate percentage of matching values
            matching_pct = (comparison_df['Equal'].sum() / len(comparison_df)) * 100
            
            # Add metadata
            result['metadata'].update({
                'matching_values': comparison_df['Equal'].sum(),
                'matching_percentage': round(matching_pct, 2),
                'non_matching': len(comparison_df) - comparison_df['Equal'].sum(),
                'total_rows': len(comparison_df)
            })
            
            # Add summary message
            result['message'] = f"Value Comparison: {comparison_df['Equal'].sum()} matching values ({matching_pct:.2f}%)"
            
            # Set result DataFrame
            result['result_df'] = comparison_df
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error comparing columns: {str(e)}"
        } 
