import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings("ignore")

# For time series analysis
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

# For statistical tests
from scipy import stats
import scipy.stats as ss

# Import helper functions from standalone_tools
from standalone_tools import load_data, save_plot, find_actual_column


#################################################
# TIME SERIES ANALYSIS TOOLS
#################################################

def forecast_time_series(
    data_location: str,
    date_column: str,
    target_column: str,
    forecast_periods: int = 10,
    model_type: str = "auto",
    seasonal_periods: Optional[int] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Performs time series forecasting using models like ARIMA, SARIMA, or ETS.
    
    Args:
        data_location: Path to the data file
        date_column: Name of the column containing dates
        target_column: Name of the column to forecast
        forecast_periods: Number of periods to forecast
        model_type: Type of model to use ("auto", "arima", "sarima", "ets")
        seasonal_periods: Number of periods in a seasonal cycle (required for SARIMA)
        test_size: Proportion of data to use for testing model performance
        
    Returns:
        Dictionary containing forecast results and plot
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    
    # Validate columns
    actual_date_column = find_actual_column(df, date_column)
    if actual_date_column is None:
        return {
            'success': False,
            'error': f"Date column '{date_column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    date_column = actual_date_column
    
    actual_target_column = find_actual_column(df, target_column)
    if actual_target_column is None:
        return {
            'success': False,
            'error': f"Target column '{target_column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    target_column = actual_target_column
    
    # Validate model type
    supported_models = ["auto", "arima", "sarima", "ets"]
    if model_type.lower() not in supported_models:
        return {
            'success': False,
            'error': f"Unsupported model type: '{model_type}'. Supported types: {', '.join(supported_models)}"
        }
    
    # Ensure date column is datetime
    try:
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        return {
            'success': False,
            'error': f"Could not convert '{date_column}' to datetime: {str(e)}"
        }
    
    # Ensure target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        try:
            df[target_column] = pd.to_numeric(df[target_column])
        except Exception as e:
            return {
                'success': False,
                'error': f"Target column '{target_column}' must be numeric: {str(e)}"
            }
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Set date as index
    df = df.set_index(date_column)
    
    # Extract the time series
    time_series = df[target_column].astype(float)
    
    # Check for missing values
    if time_series.isnull().any():
        # Basic forward-fill imputation for missing values
        time_series = time_series.fillna(method='ffill').fillna(method='bfill')
    
    # Split into train and test
    train_size = int(len(time_series) * (1 - test_size))
    train, test = time_series[:train_size], time_series[train_size:]
    
    if len(train) < 10:
        return {
            'success': False,
            'error': f"Insufficient data for forecasting. Need at least 10 observations, but got {len(train)} in training set."
        }
    
    # Determine seasonal_periods if auto and not provided
    if seasonal_periods is None:
        # Try to infer from the index frequency
        if isinstance(time_series.index, pd.DatetimeIndex):
            if time_series.index.inferred_freq == 'MS' or time_series.index.inferred_freq == 'M':
                seasonal_periods = 12  # Monthly data
            elif time_series.index.inferred_freq == 'QS' or time_series.index.inferred_freq == 'Q':
                seasonal_periods = 4   # Quarterly data
            elif time_series.index.inferred_freq == 'D':
                seasonal_periods = 7   # Daily data
            elif time_series.index.inferred_freq == 'H':
                seasonal_periods = 24  # Hourly data
            elif time_series.index.inferred_freq == 'W':
                seasonal_periods = 52  # Weekly data
            else:
                seasonal_periods = 1   # Default/unknown
    
    # Forecast with selected model
    try:
        # Auto-selection of model
        if model_type.lower() == "auto":
            # Simple heuristic for model selection:
            # Check for stationarity with ADF test
            adf_result = adfuller(train)
            is_stationary = adf_result[1] < 0.05
            
            # Check for seasonality
            is_seasonal = False
            if seasonal_periods > 1:
                # Look at autocorrelation at seasonal lag
                acf_values = acf(train, nlags=seasonal_periods+1)
                is_seasonal = abs(acf_values[seasonal_periods]) > 0.3
            
            if is_seasonal:
                model_type = "sarima"
            elif is_stationary:
                model_type = "arima"
            else:
                model_type = "ets"
        
        # Now use the selected model
        if model_type.lower() == "arima":
            # Simple ARIMA(1,1,1) model
            model = ARIMA(train, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(test) + forecast_periods)
            model_description = "ARIMA(1,1,1)"
            
        elif model_type.lower() == "sarima":
            if seasonal_periods is None or seasonal_periods <= 1:
                return {
                    'success': False,
                    'error': "SARIMA requires seasonal_periods > 1"
                }
            
            # Simple SARIMA model with default parameters
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=len(test) + forecast_periods)
            model_description = f"SARIMA(1,1,1)x(1,1,1,{seasonal_periods})"
            
        elif model_type.lower() == "ets":
            # Simple ETS model (Holt-Winters)
            if seasonal_periods is None or seasonal_periods <= 1:
                # Additive trend, no seasonality
                model = ExponentialSmoothing(train, trend='add')
                seasonal_type = "None"
            else:
                # Additive trend and seasonality
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
                seasonal_type = "Additive"
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(len(test) + forecast_periods)
            model_description = f"ETS(A,{seasonal_type})"
        
        # Create evaluation metrics
        if len(test) > 0:
            # Calculate test error metrics
            test_forecast = forecast[:len(test)]
            mse = ((test_forecast - test) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = abs(test_forecast - test).mean()
            mape = abs((test_forecast - test) / test).mean() * 100
            
            evaluation_metrics = {
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'mape': round(mape, 4) if not np.isinf(mape) else "N/A"
            }
        else:
            evaluation_metrics = {"note": "No test data available for evaluation"}
        
        # Extract future forecast
        future_forecast = forecast[-forecast_periods:]
        
        # Create forecast DataFrame
        forecast_index = pd.date_range(
            start=time_series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=pd.infer_freq(time_series.index)
        )
        
        forecast_df = pd.DataFrame({
            'forecast': future_forecast
        }, index=forecast_index)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data
        ax.plot(time_series.index, time_series, label='Original Data', color='blue')
        
        # Plot test predictions if available
        if len(test) > 0:
            ax.plot(test.index, test_forecast, label='Test Predictions', color='green', linestyle='--')
        
        # Plot future forecast
        ax.plot(forecast_index, future_forecast, label='Forecast', color='red', linestyle='-.')
        
        # Add confidence intervals (simple approximation)
        if len(test) > 0:
            error_std = np.std(test - test_forecast)
            ax.fill_between(
                forecast_index,
                future_forecast - 1.96 * error_std,
                future_forecast + 1.96 * error_std,
                color='red', alpha=0.2, label='95% Confidence Interval'
            )
        
        ax.set_title(f'Time Series Forecast using {model_description}')
        ax.set_xlabel('Date')
        ax.set_ylabel(target_column)
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        plot_url = save_plot(fig)
        
        # Prepare result
        result = {
            'success': True,
            'message': f"Successfully forecasted {forecast_periods} periods using {model_description}",
            'result_df': forecast_df.reset_index().rename(columns={'index': date_column}),
            'plot_url': plot_url,
            'metadata': {
                'model_type': model_type,
                'model_description': model_description,
                'seasonal_periods': seasonal_periods,
                'forecast_periods': forecast_periods,
                'evaluation_metrics': evaluation_metrics,
                'train_size': len(train),
                'test_size': len(test)
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error in time series forecasting: {str(e)}"
        }


def decompose_time_series(
    data_location: str,
    date_column: str,
    target_column: str,
    model_type: str = "additive",
    seasonal_periods: Optional[int] = None
) -> Dict[str, Any]:
    """
    Decomposes a time series into trend, seasonal, and residual components.
    
    Args:
        data_location: Path to the data file
        date_column: Name of the column containing dates
        target_column: Name of the column to decompose
        model_type: Type of decomposition ("additive" or "multiplicative")
        seasonal_periods: Number of periods in a seasonal cycle
        
    Returns:
        Dictionary containing decomposition results and plot
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    
    # Validate columns
    actual_date_column = find_actual_column(df, date_column)
    if actual_date_column is None:
        return {
            'success': False,
            'error': f"Date column '{date_column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    date_column = actual_date_column
    
    actual_target_column = find_actual_column(df, target_column)
    if actual_target_column is None:
        return {
            'success': False,
            'error': f"Target column '{target_column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    target_column = actual_target_column
    
    # Validate model type
    if model_type.lower() not in ["additive", "multiplicative"]:
        return {
            'success': False,
            'error': f"Unsupported model type: '{model_type}'. Supported types: 'additive', 'multiplicative'"
        }
    
    # Ensure date column is datetime
    try:
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        return {
            'success': False,
            'error': f"Could not convert '{date_column}' to datetime: {str(e)}"
        }
    
    # Ensure target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        try:
            df[target_column] = pd.to_numeric(df[target_column])
        except Exception as e:
            return {
                'success': False,
                'error': f"Target column '{target_column}' must be numeric: {str(e)}"
            }
    
    # Sort by date and set as index
    df = df.sort_values(by=date_column)
    df = df.set_index(date_column)
    
    # Extract the time series
    time_series = df[target_column].astype(float)
    
    # Check for missing values
    if time_series.isnull().any():
        time_series = time_series.fillna(method='ffill').fillna(method='bfill')
    
    # Try to infer seasonal_periods if not provided
    if seasonal_periods is None:
        if isinstance(time_series.index, pd.DatetimeIndex):
            freq = pd.infer_freq(time_series.index)
            if freq in ['MS', 'M']:
                seasonal_periods = 12  # Monthly data
            elif freq in ['QS', 'Q']:
                seasonal_periods = 4   # Quarterly data
            elif freq == 'D':
                seasonal_periods = 7   # Daily data
            elif freq == 'H':
                seasonal_periods = 24  # Hourly data
            elif freq == 'W':
                seasonal_periods = 52  # Weekly data
            else:
                # Default to 12 if we can't determine
                seasonal_periods = 12
    
    if seasonal_periods is None or seasonal_periods < 2:
        return {
            'success': False,
            'error': "seasonal_periods must be at least 2 for decomposition"
        }
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(
            time_series, 
            model=model_type.lower(), 
            period=seasonal_periods
        )
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        # Create DataFrame with components
        components_df = pd.DataFrame({
            'original': time_series,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        })
        
        # Handle any NaN values from decomposition
        components_df = components_df.dropna()
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Plot original data
        axes[0].plot(components_df.index, components_df['original'], label='Original')
        axes[0].set_ylabel('Original')
        axes[0].legend()
        
        # Plot trend
        axes[1].plot(components_df.index, components_df['trend'], label='Trend', color='green')
        axes[1].set_ylabel('Trend')
        axes[1].legend()
        
        # Plot seasonal component
        axes[2].plot(components_df.index, components_df['seasonal'], label='Seasonal', color='red')
        axes[2].set_ylabel('Seasonal')
        axes[2].legend()
        
        # Plot residual
        axes[3].plot(components_df.index, components_df['residual'], label='Residual', color='purple')
        axes[3].set_ylabel('Residual')
        axes[3].legend()
        
        plt.suptitle(f'{model_type.capitalize()} Time Series Decomposition ({target_column})', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save plot
        plot_url = save_plot(fig)
        
        # Prepare result
        result = {
            'success': True,
            'message': f"Successfully decomposed time series using {model_type} model",
            'result_df': components_df.reset_index().rename(columns={'index': date_column}),
            'plot_url': plot_url,
            'metadata': {
                'model_type': model_type,
                'seasonal_periods': seasonal_periods,
                'series_length': len(time_series),
                'components_stats': {
                    'trend': {
                        'mean': components_df['trend'].mean(),
                        'std': components_df['trend'].std(),
                        'min': components_df['trend'].min(),
                        'max': components_df['trend'].max()
                    },
                    'seasonal': {
                        'mean': components_df['seasonal'].mean(),
                        'std': components_df['seasonal'].std(),
                        'min': components_df['seasonal'].min(),
                        'max': components_df['seasonal'].max()
                    },
                    'residual': {
                        'mean': components_df['residual'].mean(),
                        'std': components_df['residual'].std(),
                        'min': components_df['residual'].min(),
                        'max': components_df['residual'].max()
                    }
                }
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error in time series decomposition: {str(e)}"
        }


#################################################
# STATISTICAL ANALYSIS TOOLS
#################################################

def hypothesis_test(
    data_location: str,
    column1: str,
    column2: Optional[str] = None,
    test_type: str = "ttest",
    alpha: float = 0.05,
    paired: bool = False,
    equal_var: bool = True
) -> Dict[str, Any]:
    """
    Performs statistical hypothesis testing.
    
    Args:
        data_location: Path to the data file
        column1: First column to test
        column2: Second column to test (for two-sample tests)
        test_type: Type of test ("ttest", "anova", "chi2", "correlation", "normality")
        alpha: Significance level
        paired: Whether the samples are paired (for t-test)
        equal_var: Whether to assume equal variance (for t-test)
        
    Returns:
        Dictionary containing test results and interpretation
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    
    # Validate columns
    actual_column1 = find_actual_column(df, column1)
    if actual_column1 is None:
        return {
            'success': False,
            'error': f"Column '{column1}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    column1 = actual_column1
    
    if column2 is not None:
        actual_column2 = find_actual_column(df, column2)
        if actual_column2 is None:
            return {
                'success': False,
                'error': f"Column '{column2}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
        column2 = actual_column2
    
    # Validate test type
    supported_tests = ["ttest", "anova", "chi2", "correlation", "normality"]
    if test_type.lower() not in supported_tests:
        return {
            'success': False,
            'error': f"Unsupported test type: '{test_type}'. Supported types: {', '.join(supported_tests)}"
        }
    
    try:
        # Extract data
        data1 = df[column1].dropna()
        
        # Initialize result components
        test_name = ""
        p_value = None
        statistic = None
        result_df = None
        plot_url = None
        interpretation = ""
        additional_info = {}
        
        # Perform the appropriate test
        if test_type.lower() == "ttest":
            # Validate column types
            if not pd.api.types.is_numeric_dtype(df[column1]):
                return {
                    'success': False,
                    'error': f"Column '{column1}' must be numeric for t-test"
                }
            
            if column2 is None:
                # One-sample t-test against 0
                statistic, p_value = stats.ttest_1samp(data1, 0)
                test_name = "One-sample t-test"
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data1, kde=True, ax=ax)
                ax.axvline(x=0, color='red', linestyle='--', label='Test Value (0)')
                ax.axvline(x=data1.mean(), color='green', linestyle='-', label='Sample Mean')
                ax.set_title(f'Distribution of {column1} (One-sample t-test)')
                ax.legend()
                plt.tight_layout()
                
                plot_url = save_plot(fig)
                
                # Create result DataFrame
                result_df = pd.DataFrame({
                    'Statistic': ['Sample Mean', 'Sample Std Dev', 'Test Value', 't-statistic', 'p-value', 'Significant'],
                    'Value': [
                        data1.mean(), 
                        data1.std(), 
                        0, 
                        statistic, 
                        p_value, 
                        p_value < alpha
                    ]
                })
                
                # Interpretation
                if p_value < alpha:
                    interpretation = f"The mean of '{column1}' is significantly different from 0 (p={p_value:.4f} < {alpha})."
                else:
                    interpretation = f"There is not enough evidence to suggest that the mean of '{column1}' is different from 0 (p={p_value:.4f} > {alpha})."
                
            else:
                # Two-sample t-test
                if not pd.api.types.is_numeric_dtype(df[column2]):
                    return {
                        'success': False,
                        'error': f"Column '{column2}' must be numeric for t-test"
                    }
                
                data2 = df[column2].dropna()
                
                if paired:
                    # Ensure same length for paired test
                    if len(data1) != len(data2):
                        # Use only common indices
                        common_index = data1.index.intersection(data2.index)
                        data1 = data1.loc[common_index]
                        data2 = data2.loc[common_index]
                        
                        if len(data1) == 0:
                            return {
                                'success': False,
                                'error': "No common non-null values found for paired t-test"
                            }
                    
                    statistic, p_value = stats.ttest_rel(data1, data2)
                    test_name = "Paired-samples t-test"
                else:
                    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                    test_name = "Independent samples t-test"
                    if not equal_var:
                        test_name += " (Welch's t-test)"
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data1, kde=True, ax=ax, label=column1, alpha=0.5)
                sns.histplot(data2, kde=True, ax=ax, label=column2, alpha=0.5)
                ax.axvline(x=data1.mean(), color='blue', linestyle='--', label=f'{column1} Mean')
                ax.axvline(x=data2.mean(), color='orange', linestyle='--', label=f'{column2} Mean')
                ax.set_title(f'Distribution Comparison ({test_name})')
                ax.legend()
                plt.tight_layout()
                
                plot_url = save_plot(fig)
                
                # Create result DataFrame
                result_df = pd.DataFrame({
                    'Statistic': [
                        f'{column1} Mean', 
                        f'{column2} Mean', 
                        'Mean Difference', 
                        f'{column1} Std Dev', 
                        f'{column2} Std Dev', 
                        't-statistic', 
                        'p-value', 
                        'Significant'
                    ],
                    'Value': [
                        data1.mean(), 
                        data2.mean(), 
                        data1.mean() - data2.mean(), 
                        data1.std(), 
                        data2.std(), 
                        statistic, 
                        p_value, 
                        p_value < alpha
                    ]
                })
                
                # Interpretation
                if p_value < alpha:
                    interpretation = f"There is a significant difference between '{column1}' and '{column2}' (p={p_value:.4f} < {alpha})."
                else:
                    interpretation = f"There is not enough evidence to suggest a difference between '{column1}' and '{column2}' (p={p_value:.4f} > {alpha})."
                
                # Add confidence interval
                if paired:
                    diff = data1 - data2
                    conf_int = stats.t.interval(
                        1 - alpha, 
                        len(diff) - 1, 
                        loc=diff.mean(), 
                        scale=stats.sem(diff)
                    )
                else:
                    # Approximation for independent samples
                    conf_int = stats.t.interval(
                        1 - alpha, 
                        len(data1) + len(data2) - 2, 
                        loc=data1.mean() - data2.mean(), 
                        scale=np.sqrt(data1.var() / len(data1) + data2.var() / len(data2))
                    )
                
                additional_info['confidence_interval'] = {
                    'lower': conf_int[0],
                    'upper': conf_int[1],
                    'confidence_level': 1 - alpha
                }
        
        elif test_type.lower() == "anova":
            # One-way ANOVA - requires a categorical column and a numeric column
            if column2 is None:
                return {
                    'success': False,
                    'error': "ANOVA requires both a categorical column (column2) and a numeric column (column1)"
                }
            
            # Ensure column1 is numeric
            if not pd.api.types.is_numeric_dtype(df[column1]):
                return {
                    'success': False,
                    'error': f"Column '{column1}' must be numeric for ANOVA"
                }
            
            # Get unique groups from column2
            groups = []
            group_data = {}
            
            for group in df[column2].dropna().unique():
                group_values = df[df[column2] == group][column1].dropna()
                if len(group_values) > 0:
                    groups.append(str(group))
                    group_data[str(group)] = group_values
            
            if len(groups) < 2:
                return {
                    'success': False,
                    'error': f"ANOVA requires at least 2 groups in '{column2}', but found {len(groups)}"
                }
            
            # Perform ANOVA
            test_name = "One-way ANOVA"
            anova_groups = [group_data[group] for group in groups]
            statistic, p_value = stats.f_oneway(*anova_groups)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Boxplot
            sns.boxplot(x=column2, y=column1, data=df, ax=ax)
            ax.set_title(f'Group Comparison: {column1} by {column2}')
            plt.tight_layout()
            
            plot_url = save_plot(fig)
            
            # Create result DataFrame with group statistics
            group_stats = []
            for group in groups:
                group_stats.append({
                    'Group': group,
                    'Count': len(group_data[group]),
                    'Mean': group_data[group].mean(),
                    'Std Dev': group_data[group].std(),
                    'Min': group_data[group].min(),
                    'Max': group_data[group].max()
                })
            
            result_df = pd.DataFrame(group_stats)
            
            # Add ANOVA results
            additional_info['anova_results'] = {
                'f_statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha,
                'num_groups': len(groups),
                'total_observations': sum(len(data) for data in anova_groups)
            }
            
            # Interpretation
            if p_value < alpha:
                interpretation = f"There is a significant difference in '{column1}' between groups in '{column2}' (F={statistic:.4f}, p={p_value:.4f} < {alpha})."
            else:
                interpretation = f"There is not enough evidence to suggest a difference in '{column1}' between groups in '{column2}' (F={statistic:.4f}, p={p_value:.4f} > {alpha})."
                
            # Add post-hoc test (Tukey HSD) if significant
            if p_value < alpha and len(groups) > 2:
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    
                    # Prepare data for Tukey's test
                    all_data = []
                    all_groups = []
                    
                    for group in groups:
                        all_data.extend(group_data[group].tolist())
                        all_groups.extend([group] * len(group_data[group]))
                    
                    # Perform Tukey's HSD test
                    tukey_result = pairwise_tukeyhsd(all_data, all_groups, alpha=alpha)
                    
                    # Convert to DataFrame
                    tukey_df = pd.DataFrame(
                        data=tukey_result._results_table.data[1:],
                        columns=tukey_result._results_table.data[0]
                    )
                    
                    additional_info['tukey_hsd'] = tukey_df.to_dict('records')
                    
                    # Add to interpretation
                    significant_pairs = tukey_df[tukey_df['reject']]
                    if len(significant_pairs) > 0:
                        interpretation += f" Post-hoc analysis (Tukey HSD) shows significant differences between {len(significant_pairs)} group pairs."
                except:
                    pass
        
        elif test_type.lower() == "chi2":
            # Chi-square test of independence
            if column2 is None:
                return {
                    'success': False,
                    'error': "Chi-square test requires two categorical columns"
                }
            
            # Create contingency table
            contingency = pd.crosstab(df[column1], df[column2])
            
            # Check if we have enough data
            if contingency.size < 4:
                return {
                    'success': False,
                    'error': "Chi-square test requires at least 2x2 contingency table"
                }
            
            # Check for low expected frequencies
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            test_name = "Chi-square test of independence"
            statistic = chi2
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Contingency table heatmap
            sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title(f'Contingency Table: {column1} vs {column2}')
            
            # Expected frequencies heatmap
            sns.heatmap(expected, annot=True, fmt='.1f', cmap='Greens', ax=ax2)
            ax2.set_title('Expected Frequencies (if independent)')
            
            plt.tight_layout()
            
            plot_url = save_plot(fig)
            
            # Add observed and expected as separate DataFrames
            observed_df = contingency.reset_index()
            expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).reset_index()
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'Statistic': ['Chi-square', 'p-value', 'Degrees of Freedom', 'Significant'],
                'Value': [chi2, p_value, dof, p_value < alpha]
            })
            
            # Add contingency tables to additional info
            additional_info['contingency_table'] = observed_df.to_dict('records')
            additional_info['expected_frequencies'] = expected_df.to_dict('records')
            
            # Check for low expected frequencies
            low_expected = (expected < 5).any().any()
            if low_expected:
                additional_info['warning'] = "Some expected frequencies are less than 5, which may affect the validity of the chi-square test."
            
            # Interpretation
            if p_value < alpha:
                interpretation = f"There is a significant association between '{column1}' and '{column2}' (χ²={chi2:.4f}, p={p_value:.4f} < {alpha})."
            else:
                interpretation = f"There is not enough evidence to suggest an association between '{column1}' and '{column2}' (χ²={chi2:.4f}, p={p_value:.4f} > {alpha})."
            
            # Add Cramer's V as effect size
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            
            additional_info['effect_size'] = {
                'cramers_v': cramers_v,
                'interpretation': 'Weak' if cramers_v < 0.3 else ('Moderate' if cramers_v < 0.5 else 'Strong')
            }
        
        elif test_type.lower() == "correlation":
            # Correlation test
            if column2 is None:
                return {
                    'success': False,
                    'error': "Correlation test requires two numeric columns"
                }
            
            # Ensure both columns are numeric
            if not pd.api.types.is_numeric_dtype(df[column1]) or not pd.api.types.is_numeric_dtype(df[column2]):
                return {
                    'success': False,
                    'error': f"Both columns must be numeric for correlation test"
                }
            
            data2 = df[column2].dropna()
            
            # Get only rows where both columns have values
            common_index = data1.index.intersection(data2.index)
            data1 = data1.loc[common_index]
            data2 = data2.loc[common_index]
            
            if len(data1) < 2:
                return {
                    'success': False,
                    'error': "Not enough common non-null values for correlation"
                }
            
            # Calculate Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(data1, data2)
            
            # Calculate Spearman rank correlation
            spearman_r, spearman_p = stats.spearmanr(data1, data2)
            
            test_name = "Correlation test"
            statistic = pearson_r  # Use Pearson's r as the main statistic
            p_value = pearson_p
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot with regression line
            sns.regplot(x=data1, y=data2, ax=ax, scatter_kws={'alpha': 0.5})
            ax.set_title(f'Correlation between {column1} and {column2}')
            ax.set_xlabel(column1)
            ax.set_ylabel(column2)
            
            # Add correlation coefficient text
            ax.annotate(f'Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})\nSpearman r = {spearman_r:.4f} (p = {spearman_p:.4f})', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            
            plot_url = save_plot(fig)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'Correlation Type': ['Pearson', 'Spearman'],
                'Correlation Coefficient': [pearson_r, spearman_r],
                'p-value': [pearson_p, spearman_p],
                'Significant': [pearson_p < alpha, spearman_p < alpha]
            })
            
            # Interpretation
            if pearson_p < alpha:
                strength = 'weak' if abs(pearson_r) < 0.3 else ('moderate' if abs(pearson_r) < 0.7 else 'strong')
                direction = 'positive' if pearson_r > 0 else 'negative'
                interpretation = f"There is a significant {strength} {direction} correlation between '{column1}' and '{column2}' (r={pearson_r:.4f}, p={pearson_p:.4f} < {alpha})."
            else:
                interpretation = f"There is not enough evidence to suggest a correlation between '{column1}' and '{column2}' (r={pearson_r:.4f}, p={pearson_p:.4f} > {alpha})."
            
            # Add confidence interval
            z = np.arctanh(pearson_r)
            se = 1/np.sqrt(len(data1)-3)
            ci_z = stats.norm.interval(1-alpha, loc=z, scale=se)
            ci_r = np.tanh(ci_z)
            
            additional_info['confidence_interval'] = {
                'lower': ci_r[0],
                'upper': ci_r[1],
                'confidence_level': 1 - alpha
            }
        
        elif test_type.lower() == "normality":
            # Normality test (Shapiro-Wilk)
            if not pd.api.types.is_numeric_dtype(df[column1]):
                return {
                    'success': False,
                    'error': f"Column '{column1}' must be numeric for normality test"
                }
            
            # Shapiro-Wilk test
            if len(data1) <= 5000:  # Shapiro-Wilk is only valid for n <= 5000
                statistic, p_value = stats.shapiro(data1)
                test_name = "Shapiro-Wilk normality test"
            else:
                # Use D'Agostino's K^2 test for larger samples
                statistic, p_value = stats.normaltest(data1)
                test_name = "D'Agostino's K^2 normality test"
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogram with normal curve
            sns.histplot(data1, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {column1}')
            
            # Q-Q plot
            sm.qqplot(data1, line='s', ax=ax2)
            ax2.set_title('Q-Q Plot')
            
            plt.tight_layout()
            
            plot_url = save_plot(fig)
            
            # Calculate descriptive statistics
            mean = data1.mean()
            median = data1.median()
            std_dev = data1.std()
            skewness = stats.skew(data1)
            kurtosis = stats.kurtosis(data1)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Test Statistic', 'p-value', 'Normal Distribution'],
                'Value': [mean, median, std_dev, skewness, kurtosis, statistic, p_value, p_value >= alpha]
            })
            
            # Interpretation
            if p_value < alpha:
                interpretation = f"The distribution of '{column1}' is significantly different from normal (p={p_value:.4f} < {alpha})."
            else:
                interpretation = f"There is not enough evidence to suggest that the distribution of '{column1}' differs from normal (p={p_value:.4f} > {alpha})."
            
            # Add descriptive statistics to additional info
            additional_info['descriptive_stats'] = {
                'mean': mean,
                'median': median,
                'std_dev': std_dev,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min': data1.min(),
                'max': data1.max(),
                'range': data1.max() - data1.min()
            }
        
        # Prepare final result
        result = {
            'success': True,
            'message': f"Successfully performed {test_name} on '{column1}'{' and ' + column2 if column2 else ''}.",
            'result_df': result_df,
            'plot_url': plot_url,
            'metadata': {
                'test_type': test_type,
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'alpha': alpha,
                'significant': p_value < alpha,
                'column1': column1,
                'column2': column2,
                'interpretation': interpretation,
                **additional_info
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error in hypothesis testing: {str(e)}"
        }

def sensitivity_analysis(
    data_location: str,
    target_column: str,
    variable_columns: List[str],
    scenario_definitions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Performs sensitivity analysis by simulating different scenarios.
    
    Args:
        data_location: Path to the data file
        target_column: Column to measure impact on
        variable_columns: Columns to vary in scenarios
        scenario_definitions: List of scenario definitions with variable values
            Example: [
                {"name": "Baseline", "variables": {"price": None, "cost": None}}, 
                {"name": "High Price", "variables": {"price": 1.1, "cost": 1.0}}
            ]
            where None means "use current values" and numbers are multipliers
        
    Returns:
        Dictionary containing scenario analysis results and comparison
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    
    # Validate columns
    actual_target_column = find_actual_column(df, target_column)
    if actual_target_column is None:
        return {
            'success': False,
            'error': f"Target column '{target_column}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
        }
    target_column = actual_target_column
    
    # Validate variable columns
    valid_variable_columns = []
    for col in variable_columns:
        actual_col = find_actual_column(df, col)
        if actual_col is None:
            return {
                'success': False,
                'error': f"Variable column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
            }
        valid_variable_columns.append(actual_col)
    
    variable_columns = valid_variable_columns
    
    # Validate scenario definitions
    if not scenario_definitions or not isinstance(scenario_definitions, list):
        return {
            'success': False,
            'error': "scenario_definitions must be a non-empty list of scenario definitions"
        }
    
    for i, scenario in enumerate(scenario_definitions):
        if not isinstance(scenario, dict):
            return {
                'success': False,
                'error': f"Scenario at index {i} must be a dictionary"
            }
        
        if "name" not in scenario:
            return {
                'success': False,
                'error': f"Scenario at index {i} missing required key 'name'"
            }
        
        if "variables" not in scenario or not isinstance(scenario["variables"], dict):
            return {
                'success': False,
                'error': f"Scenario at index {i} missing required key 'variables' or 'variables' is not a dictionary"
            }
        
        # Check that variables reference columns in variable_columns
        for var_name in scenario["variables"].keys():
            if var_name not in variable_columns:
                return {
                    'success': False,
                    'error': f"Variable '{var_name}' in scenario '{scenario['name']}' not in variable_columns list"
                }
    
    # Ensure all columns are numeric
    for col in [target_column] + variable_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                return {
                    'success': False,
                    'error': f"Column '{col}' must be numeric for sensitivity analysis"
                }
    
    try:
        # Create a copy of the DataFrame for scenarios
        scenario_results = []
        baseline_data = None
        
        # Process each scenario
        for scenario in scenario_definitions:
            scenario_name = scenario["name"]
            variables = scenario["variables"]
            
            # Create a copy of the DataFrame for this scenario
            scenario_df = df.copy()
            
            # Apply variable changes
            for var_name, multiplier in variables.items():
                if multiplier is not None:
                    scenario_df[var_name] = scenario_df[var_name] * multiplier
            
            # Calculate target column statistics
            target_stats = {
                'scenario': scenario_name,
                'mean': scenario_df[target_column].mean(),
                'median': scenario_df[target_column].median(),
                'std': scenario_df[target_column].std(),
                'min': scenario_df[target_column].min(),
                'max': scenario_df[target_column].max(),
                'sum': scenario_df[target_column].sum()
            }
            
            # Add variable values
            for var_name in variable_columns:
                multiplier = variables.get(var_name, None)
                if multiplier is None:
                    target_stats[f'{var_name}_value'] = "Unchanged"
                    target_stats[f'{var_name}_multiplier'] = 1.0
                else:
                    target_stats[f'{var_name}_value'] = f"{multiplier:.2f}x"
                    target_stats[f'{var_name}_multiplier'] = multiplier
            
            # Save scenario results
            scenario_results.append(target_stats)
            
            # Save baseline data if this is the first scenario
            if scenario_name.lower() == "baseline" or baseline_data is None:
                baseline_data = target_stats
        
        # Convert to DataFrame
        result_df = pd.DataFrame(scenario_results)
        
        # Calculate % changes from baseline
        if baseline_data is not None:
            baseline_mean = baseline_data['mean']
            baseline_sum = baseline_data['sum']
            
            for i, row in result_df.iterrows():
                if row['scenario'] != baseline_data['scenario']:
                    result_df.at[i, 'pct_change_mean'] = (row['mean'] / baseline_mean - 1) * 100
                    result_df.at[i, 'pct_change_sum'] = (row['sum'] / baseline_sum - 1) * 100
        
        # Create tornado chart visualization
        if len(scenario_results) > 1:
            # Prepare data for tornado chart
            tornado_data = []
            
            for scenario in scenario_results:
                if scenario['scenario'] != baseline_data['scenario']:
                    # Calculate percentage change
                    pct_change = (scenario['mean'] / baseline_data['mean'] - 1) * 100
                    
                    # Get the most significant variable change
                    max_var_change = None
                    max_var_name = None
                    
                    for var_name in variable_columns:
                        multiplier = scenario.get(f'{var_name}_multiplier', 1.0)
                        if multiplier != 1.0:
                            var_change = abs(multiplier - 1.0)
                            if max_var_change is None or var_change > max_var_change:
                                max_var_change = var_change
                                max_var_name = var_name
                    
                    if max_var_name:
                        tornado_data.append({
                            'variable': max_var_name,
                            'scenario': scenario['scenario'],
                            'pct_change': pct_change
                        })
            
            if tornado_data:
                # Create tornado chart
                tornado_df = pd.DataFrame(tornado_data)
                
                # Sort by absolute percentage change
                tornado_df['abs_change'] = tornado_df['pct_change'].abs()
                tornado_df = tornado_df.sort_values('abs_change', ascending=True)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot bars
                colors = ['red' if x < 0 else 'green' for x in tornado_df['pct_change']]
                bars = ax.barh(tornado_df['variable'] + ' (' + tornado_df['scenario'] + ')', 
                              tornado_df['pct_change'], color=colors)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width if width >= 0 else width - 1
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                           f"{width:.1f}%", va='center')
                
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel('Percentage Change in ' + target_column)
                ax.set_title('Sensitivity Analysis: Impact on ' + target_column)
                plt.tight_layout()
                
                # Save plot
                tornado_plot_url = save_plot(fig)
            else:
                tornado_plot_url = None
        else:
            tornado_plot_url = None
        
        # Create comparison visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean values for each scenario
        bars = ax.bar(result_df['scenario'], result_df['mean'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.2f}", ha='center', va='bottom')
        
        ax.set_ylabel(target_column + ' (Mean)')
        ax.set_title('Scenario Comparison: Impact on ' + target_column)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_url = save_plot(fig)
        
        # Prepare result
        result = {
            'success': True,
            'message': f"Successfully performed sensitivity analysis on {len(scenario_definitions)} scenarios.",
            'result_df': result_df,
            'plot_url': plot_url,
            'metadata': {
                'target_column': target_column,
                'variable_columns': variable_columns,
                'scenarios': len(scenario_definitions),
                'baseline': baseline_data['scenario'] if baseline_data else None,
                'max_impact_scenario': result_df.loc[result_df['pct_change_mean'].abs().idxmax()]['scenario'] if 'pct_change_mean' in result_df.columns else None,
                'tornado_plot_url': tornado_plot_url
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error in sensitivity analysis: {str(e)}"
        }

#################################################
# DATA TRANSFORMATION TOOLS
#################################################

def reshape_data(
    data_location: str,
    format_type: str,
    id_vars: Optional[List[str]] = None,
    value_vars: Optional[List[str]] = None,
    var_name: str = "variable",
    value_name: str = "value"
) -> Dict[str, Any]:
    """
    Reshapes data between wide and long formats.
    
    Args:
        data_location: Path to the data file
        format_type: Target format ("wide_to_long" or "long_to_wide")
        id_vars: Columns to use as identifiers (required for wide_to_long)
        value_vars: Columns to unpivot (for wide_to_long) or to use as values (for long_to_wide)
        var_name: Name for the variable column (for wide_to_long)
        value_name: Name for the value column (for wide_to_long)
        
    Returns:
        Dictionary containing reshaped DataFrame
    """
    # Load the data
    loaded_data = load_data(data_location)
    if not loaded_data.get('success', False):
        return {
            'success': False,
            'error': loaded_data.get('error', 'Unknown error loading data')
        }

    df = loaded_data['result_df']
    
    # Validate format type
    if format_type.lower() not in ["wide_to_long", "long_to_wide"]:
        return {
            'success': False,
            'error': f"Invalid format_type: '{format_type}'. Must be 'wide_to_long' or 'long_to_wide'"
        }
    
    try:
        # Wide to Long format (unpivot/melt)
        if format_type.lower() == "wide_to_long":
            if not id_vars:
                return {
                    'success': False,
                    'error': "id_vars is required for wide_to_long transformation"
                }
            
            # Validate id_vars columns
            valid_id_vars = []
            for col in id_vars:
                actual_col = find_actual_column(df, col)
                if actual_col is None:
                    return {
                        'success': False,
                        'error': f"ID column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
                    }
                valid_id_vars.append(actual_col)
            
            # If value_vars is not specified, use all columns not in id_vars
            if not value_vars:
                value_vars = [col for col in df.columns if col not in valid_id_vars]
            else:
                # Validate value_vars columns
                valid_value_vars = []
                for col in value_vars:
                    actual_col = find_actual_column(df, col)
                    if actual_col is None:
                        return {
                            'success': False,
                            'error': f"Value column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
                        }
                    valid_value_vars.append(actual_col)
                value_vars = valid_value_vars
            
            # Perform the melt operation
            melted_df = pd.melt(
                df,
                id_vars=valid_id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name
            )
            
            # Create visualization to show the transformation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Sample of original data
            ax1.set_title("Original Data (Wide Format)")
            ax1.axis('tight')
            ax1.axis('off')
            wide_sample = df.head(min(5, len(df)))
            table1 = ax1.table(
                cellText=wide_sample.values,
                colLabels=wide_sample.columns,
                cellLoc='center',
                loc='center'
            )
            table1.auto_set_font_size(False)
            table1.set_fontsize(9)
            table1.scale(1.2, 1.2)
            
            # Sample of melted data
            ax2.set_title("Transformed Data (Long Format)")
            ax2.axis('tight')
            ax2.axis('off')
            long_sample = melted_df.head(min(10, len(melted_df)))
            table2 = ax2.table(
                cellText=long_sample.values,
                colLabels=long_sample.columns,
                cellLoc='center',
                loc='center'
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(9)
            table2.scale(1.2, 1.2)
            
            plt.tight_layout()
            
            # Save plot
            plot_url = save_plot(fig)
            
            # Prepare result
            result = {
                'success': True,
                'message': f"Successfully reshaped data from wide to long format.",
                'result_df': melted_df,
                'plot_url': plot_url,
                'metadata': {
                    'format_type': 'wide_to_long',
                    'original_shape': df.shape,
                    'result_shape': melted_df.shape,
                    'id_vars': valid_id_vars,
                    'value_vars': value_vars,
                    'var_name': var_name,
                    'value_name': value_name
                }
            }
            
            return result
            
        # Long to Wide format (pivot)
        else:  # format_type.lower() == "long_to_wide"
            if not id_vars or not value_vars or len(value_vars) != 2:
                return {
                    'success': False,
                    'error': "For long_to_wide, id_vars must be specified and value_vars must contain exactly 2 columns: the column to use as new columns, and the column containing values"
                }
            
            # Validate columns
            valid_id_vars = []
            for col in id_vars:
                actual_col = find_actual_column(df, col)
                if actual_col is None:
                    return {
                        'success': False,
                        'error': f"ID column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
                    }
                valid_id_vars.append(actual_col)
            
            valid_value_vars = []
            for col in value_vars:
                actual_col = find_actual_column(df, col)
                if actual_col is None:
                    return {
                        'success': False,
                        'error': f"Value column '{col}' not found in DataFrame. Available columns: {', '.join(df.columns)}"
                    }
                valid_value_vars.append(actual_col)
            
            # Extract column names for pivot
            pivot_columns = valid_value_vars[0]  # Column to use for new column names
            pivot_values = valid_value_vars[1]   # Column containing values
            
            # Perform the pivot operation
            pivoted_df = df.pivot_table(
                index=valid_id_vars,
                columns=pivot_columns,
                values=pivot_values,
                aggfunc='first'  # Use first value if there are duplicates
            ).reset_index()
            
            # Create visualization to show the transformation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Sample of original data
            ax1.set_title("Original Data (Long Format)")
            ax1.axis('tight')
            ax1.axis('off')
            long_sample = df.head(min(10, len(df)))
            table1 = ax1.table(
                cellText=long_sample.values,
                colLabels=long_sample.columns,
                cellLoc='center',
                loc='center'
            )
            table1.auto_set_font_size(False)
            table1.set_fontsize(9)
            table1.scale(1.2, 1.2)
            
            # Sample of pivoted data
            ax2.set_title("Transformed Data (Wide Format)")
            ax2.axis('tight')
            ax2.axis('off')
            wide_sample = pivoted_df.head(min(5, len(pivoted_df)))
            table2 = ax2.table(
                cellText=wide_sample.values,
                colLabels=wide_sample.columns,
                cellLoc='center',
                loc='center'
            )
            table2.auto_set_font_size(False)
            table2.set_fontsize(9)
            table2.scale(1.2, 1.2)
            
            plt.tight_layout()
            
            # Save plot
            plot_url = save_plot(fig)
            
            # Prepare result
            result = {
                'success': True,
                'message': f"Successfully reshaped data from long to wide format.",
                'result_df': pivoted_df,
                'plot_url': plot_url,
                'metadata': {
                    'format_type': 'long_to_wide',
                    'original_shape': df.shape,
                    'result_shape': pivoted_df.shape,
                    'id_vars': valid_id_vars,
                    'pivot_columns': pivot_columns,
                    'pivot_values': pivot_values,
                    'unique_values_count': df[pivot_columns].nunique()
                }
            }
            
            return result
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error in reshaping data: {str(e)}"
        } 
