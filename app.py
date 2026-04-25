from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib
import warnings
import math
import traceback
import base64
from io import BytesIO
import json
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import re

# Machine Learning imports
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

# Time Series imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'universal-predict-app-2024'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt', 'json', 'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
reference_data = None
prediction_model = None
model_type = None  # 'classification', 'regression', or 'timeseries'
model_trained = False
column_info = None
time_series_info = None
scaler = None
last_prediction_date = None
last_prediction_value = None
prediction_context = {}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def detect_data_type(df, target_column=None):
    """Detect if data is classification, regression, or time series"""
    date_columns = []
    for col in df.columns:
        try:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(10).astype(str)
                if sample.str.contains(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}', regex=True).any():
                    date_columns.append(col)
        except:
            pass

    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'time', 'month', 'year', 'day', 'timestamp']):
            if col not in date_columns:
                date_columns.append(col)

    if date_columns:
        for date_col in date_columns:
            try:
                test_df = df.copy()
                test_df[date_col] = pd.to_datetime(test_df[date_col], errors='coerce')
                valid_dates = test_df[date_col].notna().sum()
                if valid_dates > len(df) * 0.5:
                    return 'timeseries', date_col
            except:
                continue
        if date_columns:
            return 'timeseries', date_columns[0]

    if target_column is None or target_column not in df.columns:
        return 'regression', None

    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype(str).str.replace(',', '')

    try:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    except:
        pass

    unique_values = df[target_column].nunique()
    if unique_values == 2:
        value_set = set(df[target_column].dropna().astype(str).str.lower())
        binary_sets = [
            {'0', '1'}, {'pass', 'fail'}, {'yes', 'no'},
            {'true', 'false'}, {'success', 'failure'}
        ]
        for binary_set in binary_sets:
            if value_set.issubset(binary_set):
                return 'classification', None
    elif unique_values <= 10 and df[target_column].dtype == 'object':
        return 'classification', None

    return 'regression', None


def clean_numeric_column(series):
    """Clean numeric column by removing commas and converting to float"""
    try:
        if series.dtype == 'object':
            cleaned = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
            return pd.to_numeric(cleaned, errors='coerce')
        return pd.to_numeric(series, errors='coerce')
    except:
        return series


def create_visualization(df, feature_col, target_col, predictions=None, model_type='regression',
                         future_predictions=None, future_dates=None):
    """Create appropriate visualization based on data type"""
    plt.figure(figsize=(14, 8))
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    df_plot = df.copy()
    df_plot[target_col] = clean_numeric_column(df_plot[target_col])

    if model_type == 'timeseries':
        try:
            if feature_col and feature_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[feature_col]):
                dates = df[feature_col]
                dates_valid = dates.notna()
                if dates_valid.sum() > 0:
                    x_vals = dates[dates_valid].values
                    y_vals = df_plot[target_col].values[dates_valid]

                    plt.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Historical Data',
                             alpha=0.8, marker='o', markersize=5)

                    if predictions is not None and len(predictions) > 0:
                        if len(predictions) == len(x_vals):
                            plt.plot(x_vals, predictions, 'r--', linewidth=2, label='Model Predictions', alpha=0.8)

                    if future_predictions is not None and future_dates is not None and len(future_predictions) > 0:
                        plt.plot(future_dates, future_predictions, 'g--', linewidth=2.5,
                                 label=f'Future Predictions ({len(future_predictions)} periods)', alpha=0.9, marker='s',
                                 markersize=6)
                        plt.scatter(future_dates, future_predictions, color='green', s=100, zorder=5,
                                    edgecolors='black', linewidth=1.5)

                    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d/%Y'))
                    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
                    plt.gcf().autofmt_xdate()
                    plt.xlabel('Date', fontsize=12, fontweight='bold')
                else:
                    raise ValueError("No valid dates")
            else:
                raise ValueError("No date column")
        except:
            x_vals = np.arange(len(df_plot))
            y_vals = df_plot[target_col].values

            plt.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='Historical Data',
                     alpha=0.8, marker='o', markersize=5)

            if predictions is not None and len(predictions) > 0:
                if len(predictions) == len(x_vals):
                    plt.plot(x_vals, predictions, 'r--', linewidth=2, label='Model Predictions', alpha=0.8)

            if future_predictions is not None and len(future_predictions) > 0:
                future_idx = np.arange(len(x_vals), len(x_vals) + len(future_predictions))
                plt.plot(future_idx, future_predictions, 'g--', linewidth=2.5,
                         label=f'Future Predictions ({len(future_predictions)} periods)', alpha=0.9, marker='s',
                         markersize=6)
                plt.scatter(future_idx, future_predictions, color='green', s=100, zorder=5,
                            edgecolors='black', linewidth=1.5)

            plt.xlabel('Time Period', fontsize=12, fontweight='bold')

        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Time Series Analysis & Predictions for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    elif model_type == 'classification':
        if feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values
        else:
            X = np.arange(len(df))

        y = df[target_col].values
        if y.dtype == 'object':
            le = LabelEncoder()
            y_numeric = le.fit_transform(y)
        else:
            y_numeric = y

        plt.scatter(X, y_numeric, alpha=0.6, s=80, label='Training Data', c='#4a6fa5', edgecolors='white',
                    linewidth=0.8)

        if predictions is not None and len(predictions) > 0:
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            pred_sorted = predictions[sorted_idx]
            plt.plot(X_sorted, pred_sorted, 'r-', linewidth=2.5, label='Decision Boundary')

        plt.xlabel(feature_col if feature_col else 'Index', fontsize=12, fontweight='bold')
        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Classification Model for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    else:
        if feature_col in df.columns:
            X = clean_numeric_column(df[feature_col]).values
        else:
            X = np.arange(len(df))

        y = df_plot[target_col].values
        plt.scatter(X, y, alpha=0.6, s=80, label='Training Data', c='#4a6fa5', edgecolors='white', linewidth=0.8)

        if predictions is not None and len(predictions) > 0:
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            pred_sorted = predictions[sorted_idx]
            plt.plot(X_sorted, pred_sorted, 'r-', linewidth=2.5, label='Regression Line')

        plt.xlabel(feature_col if feature_col else 'Index', fontsize=12, fontweight='bold')
        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        plt.title(f'Regression Model for {target_col}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(framealpha=0.9, loc='best')
        plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def prepare_timeseries_data(df, target_col, date_col=None):
    """Prepare time series data for modeling"""
    df = df.copy()

    df[target_col] = clean_numeric_column(df[target_col])
    df = df.dropna(subset=[target_col])

    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(date_col, ascending=True)
            df.set_index(date_col, inplace=True)
        except Exception as e:
            print(f"Warning: Could not process date column '{date_col}': {e}")
            df.index = pd.RangeIndex(start=0, stop=len(df))
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df))

    return df[[target_col]]


def calculate_monthly_trend(series, dates=None):
    """Calculate monthly trend from time series"""
    if len(series) < 2:
        return 0

    try:
        if dates is not None and len(dates) == len(series):
            months = []
            start_date = dates[0]
            for d in dates:
                months_diff = (d.year - start_date.year) * 12 + (d.month - start_date.month)
                months.append(months_diff)
            x = np.array(months)
        else:
            x = np.arange(len(series))

        y = np.array(series)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope = model.params[1]
        return float(slope)

    except Exception as e:
        print(f"Error calculating trend: {e}")
        if len(series) >= 2:
            return float((series[-1] - series[0]) / max(1, (len(series) - 1)))
        return 0.0


def convert_to_json_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable formats"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, (np.datetime64, datetime)):
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).strftime('%Y-%m-%d %H:%M:%S')
        else:
            return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


def train_time_series_model(series, dates=None):
    """Train time series model for monthly data"""
    if len(series) < 10:
        trend = calculate_monthly_trend(series, dates)
        return {
            'model_type': 'trend_moving_average',
            'window': min(3, len(series)),
            'last_values': [float(x) for x in series[-min(5, len(series)):]],
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series)) if len(series) > 0 else 0.0
        }

    try:
        if len(series) >= 24:
            try:
                model = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit()
                trend = calculate_monthly_trend(series, dates)
                return {
                    'model_type': 'arima',
                    'model': model_fit,
                    'order': (1, 1, 1),
                    'seasonal_order': (1, 1, 1, 12),
                    'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(
                        dates) > 0 else None,
                    'aic': float(model_fit.aic),
                    'trend': float(trend),
                    'last_value': float(series[-1]) if len(series) > 0 else 0.0,
                    'initial_value': float(series[0]) if len(series) > 0 else 0.0,
                    'series_length': len(series),
                    'avg_value': float(np.mean(series))
                }
            except Exception as e:
                print(f"ARIMA failed: {e}")

        try:
            model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=12)
            model_fit = model.fit()
            trend = calculate_monthly_trend(series, dates)
            return {
                'model_type': 'holt_winters',
                'model': model_fit,
                'params': {k: float(v) for k, v in model_fit.params.items()},
                'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
                'trend': float(trend),
                'last_value': float(series[-1]) if len(series) > 0 else 0.0,
                'initial_value': float(series[0]) if len(series) > 0 else 0.0,
                'series_length': len(series),
                'avg_value': float(np.mean(series))
            }
        except Exception as e:
            print(f"Holt-Winters failed: {e}")

        window = min(6, len(series) // 2)
        trend = calculate_monthly_trend(series, dates)
        return {
            'model_type': 'trend_moving_average',
            'window': window,
            'last_values': [float(x) for x in series[-window * 2:]],
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series))
        }

    except Exception as e:
        print(f"Error in time series training: {e}")
        trend = calculate_monthly_trend(series, dates)
        return {
            'model_type': 'trend_simple_average',
            'average': float(np.mean(series)),
            'last_values': [float(x) for x in series[-min(5, len(series)):]],
            'last_date': convert_to_json_serializable(dates[-1]) if dates is not None and len(dates) > 0 else None,
            'trend': float(trend),
            'last_value': float(series[-1]) if len(series) > 0 else 0.0,
            'initial_value': float(series[0]) if len(series) > 0 else 0.0,
            'series_length': len(series),
            'avg_value': float(np.mean(series))
        }


def predict_time_series(model_info, steps=1):
    """Make predictions using trained time series model"""
    model_type = model_info.get('model_type', 'trend_simple_average')
    trend = float(model_info.get('trend', 0))
    last_value = float(model_info.get('last_value', 0))
    avg_value = float(model_info.get('avg_value', 0))
    series_length = int(model_info.get('series_length', 0))

    if model_type == 'arima':
        try:
            model = model_info['model']
            forecast = model.forecast(steps=steps)
            predictions = [float(x) for x in forecast.tolist()]
            return predictions
        except Exception as e:
            print(f"ARIMA prediction failed: {e}")

    if model_type == 'holt_winters':
        try:
            model = model_info['model']
            forecast = model.forecast(steps)
            predictions = [float(x) for x in forecast.tolist()]
            return predictions
        except Exception as e:
            print(f"Holt-Winters prediction failed: {e}")

    if model_type == 'trend_moving_average':
        window = int(model_info.get('window', 3))
        last_values = [float(x) for x in model_info.get('last_values', [])]
        if len(last_values) >= window:
            predictions = []
            current_values = last_values.copy()

            for step in range(steps):
                pred = float(np.mean(current_values[-window:]))
                trend_effect = trend * (step + 1)
                pred = pred + trend_effect

                if len(current_values) > 1:
                    historical = np.array(current_values)
                    volatility = np.std(historical) * 0.05
                    pred += np.random.normal(0, volatility)

                predictions.append(float(pred))
                current_values.append(float(pred))

            return predictions

    if 'average' in model_info:
        base_value = float(model_info['average'])
    else:
        base_value = avg_value if avg_value != 0 else last_value

    predictions = []
    for i in range(steps):
        pred = base_value + trend * (i + 1)

        if series_length > 12:
            seasonal_effect = 0.02 * base_value * math.sin(2 * math.pi * (i % 12) / 12)
            pred += seasonal_effect

        pred += np.random.normal(0, abs(trend) * 0.5)
        predictions.append(float(pred))

    return predictions


def detect_csv_has_header(filepath, feature_col=None):
    """Detect if CSV file has a header row"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        if not first_line:
            return False

        parts = first_line.split(',')
        if not parts:
            return False

        first_cell = parts[0].strip()

        # Check if first cell looks like the feature column name
        if feature_col and feature_col.lower() == first_cell.lower():
            return True

        # Check if first cell contains letters (likely a header)
        if any(char.isalpha() for char in first_cell):
            return True

        # Try to parse as number - if it succeeds, it's probably data, not header
        try:
            float(first_cell)
            return False  # It's a number, so no header
        except:
            # Not a number, check other cells
            for cell in parts[1:]:
                cell = cell.strip()
                if cell and any(char.isalpha() for char in cell):
                    return True

        return False
    except:
        return True  # Default to assuming header if can't determine


def extract_numeric_value(input_str, idx):
    """Extract numeric value from input string"""
    if input_str is None:
        return float(idx)

    if isinstance(input_str, (int, float)):
        return float(input_str)

    input_str = str(input_str).strip()

    if not input_str:
        return float(idx)

    # Try direct conversion
    try:
        return float(input_str)
    except:
        pass

    # Try to extract numbers from string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", input_str)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass

    # If no number found, use index as fallback
    return float(idx)


def read_flexible_file(filepath, feature_col=None):
    """Read file with flexible header handling"""
    filename = os.path.basename(filepath)

    try:
        if filename.endswith('.csv'):
            # First try to detect if file has header
            has_header = detect_csv_has_header(filepath, feature_col)

            if has_header:
                df = pd.read_csv(filepath)
                print(f"DEBUG: Reading CSV with header. Columns: {df.columns.tolist()}")
            else:
                # Read without header
                df = pd.read_csv(filepath, header=None)
                print(f"DEBUG: Reading CSV without header. Shape: {df.shape}")

                # Try to assign meaningful column names
                if feature_col and df.shape[1] == 1:
                    # Single column file, use feature column name
                    df.columns = [feature_col]
                elif reference_data and 'columns' in reference_data:
                    # Try to match with training data columns
                    training_cols = reference_data['columns']
                    for i in range(min(df.shape[1], len(training_cols))):
                        if i < df.shape[1]:
                            df.rename(columns={i: training_cols[i]}, inplace=True)
                else:
                    # Generic column names
                    df.columns = [f'Column_{i}' for i in range(df.shape[1])]

        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            # Try to read as CSV with various options
            try:
                df = pd.read_csv(filepath, sep=None, engine='python')
            except:
                # Last resort: read as single column
                df = pd.read_csv(filepath, header=None, names=['Value'])

        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]

        # Clean data - convert object columns to numeric if possible
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = clean_numeric_column(df[col])
                except:
                    pass

        return df

    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_data', methods=['POST'])
def upload_data():
    global reference_data, prediction_model, model_type, model_trained, column_info, time_series_info, scaler, prediction_context

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if not allowed_file(file.filename):
            return jsonify(
                {'success': False, 'error': f'File type not allowed. Allowed: {app.config["ALLOWED_EXTENSIONS"]}'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = read_flexible_file(filepath)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})

        if len(df.columns) < 1:
            return jsonify({'success': False, 'error': 'File must have at least 1 column'})

        date_columns = []
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(5).astype(str)
                    patterns = [
                        r'\d{1,2}/\d{1,2}/\d{4}',
                        r'\d{4}-\d{2}-\d{2}',
                        r'\d{2}-\d{2}-\d{4}',
                        r'\d{1,2}-\d{1,2}-\d{2}',
                        r'\d{1,2}/\d{1,2}/\d{2}'
                    ]
                    for pattern in patterns:
                        if sample.str.contains(pattern).any():
                            date_columns.append(col)
                            break
            except:
                pass

        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['date', 'time', 'month', 'year', 'day', 'timestamp']):
                if col not in date_columns:
                    date_columns.append(col)

        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            except:
                pass

        reference_data = {
            'df': df,
            'filepath': filepath,
            'filename': filename,
            'columns': df.columns.tolist(),
            'date_columns': date_columns
        }

        prediction_model = None
        model_type = None
        model_trained = False
        column_info = None
        time_series_info = None
        scaler = None
        prediction_context = {}

        preview_df = df.head(10).copy()
        for col in preview_df.columns:
            if pd.api.types.is_numeric_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "NaN")

        preview_html = preview_df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            na_rep='NaN'
        )

        numeric_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
            except:
                pass

        stats = {}
        if numeric_cols:
            for col in numeric_cols[:10]:
                try:
                    clean_series = clean_numeric_column(df[col])
                    stats[col] = {
                        'mean': f"{clean_series.mean():.2f}",
                        'std': f"{clean_series.std():.2f}",
                        'min': f"{clean_series.min():.2f}",
                        'max': f"{clean_series.max():.2f}",
                        'non_null': int(clean_series.notna().sum()),
                        'null': int(clean_series.isna().sum())
                    }
                except:
                    stats[col] = {'error': 'Cannot calculate statistics'}

        return jsonify({
            'success': True,
            'filename': filename,
            'columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'date_columns': date_columns,
            'preview': preview_html,
            'stats': stats,
            'shape': {'rows': len(df), 'cols': len(df.columns)},
            'message': f'Successfully loaded {len(df)} rows with {len(df.columns)} columns'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in upload_data: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/setup_model', methods=['POST'])
def setup_model():
    global reference_data, model_type, column_info, time_series_info, prediction_context

    try:
        if reference_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded. Please upload data first.'})

        data = request.get_json()

        raw_feature = data.get('feature_column') or data.get('featureColumn')
        raw_target = data.get('target_column') or data.get('targetColumn')

        df = reference_data['df']
        date_columns = reference_data['date_columns']

        if not raw_target:
            return jsonify({'success': False, 'error': 'Please select a target column to predict'})

        if raw_target not in df.columns:
            return jsonify({'success': False, 'error': f'Target column "{raw_target}" not found in data'})

        feature_col = raw_feature
        target_col = raw_target

        is_timeseries = False

        if feature_col and feature_col in date_columns:
            is_timeseries = True
        elif not feature_col and target_col not in date_columns and date_columns:
            feature_col = date_columns[0]
            is_timeseries = True
        elif feature_col and feature_col not in date_columns and target_col not in date_columns:
            is_timeseries = False
        elif not feature_col and target_col not in date_columns:
            is_timeseries = False
        else:
            is_timeseries = False

        if is_timeseries:
            model_type = 'timeseries'
        else:
            detected_type, _ = detect_data_type(df, target_col)
            model_type = detected_type

        column_info = {
            'feature': feature_col,
            'target': target_col,
            'model_type': model_type,
            'is_timeseries': model_type == 'timeseries'
        }

        prediction_context = {
            'target_name': target_col,
            'target_description': 'value',
            'feature_name': feature_col if feature_col else 'Index',
            'model_type': model_type,
            'data_source': reference_data['filename'],
            'is_timeseries': model_type == 'timeseries'
        }

        if model_type == 'timeseries':
            time_series_info = {
                'date_column': feature_col,
                'target_column': target_col,
                'frequency': 'M'
            }
            problem_desc = "Time Series Forecasting (Monthly)"
        elif model_type == 'classification':
            problem_desc = f"Classification ({df[target_col].nunique()} classes)"
        else:
            problem_desc = "Regression Analysis"

        target_stats = {
            'unique_values': int(df[target_col].nunique()),
            'data_type': str(df[target_col].dtype),
            'missing': int(df[target_col].isna().sum())
        }

        if model_type != 'classification':
            try:
                clean_target = clean_numeric_column(df[target_col])
                if clean_target.notna().sum() > 0:
                    target_stats.update({
                        'mean': f"{clean_target.mean():.2f}",
                        'std': f"{clean_target.std():.2f}",
                        'min': f"{clean_target.min():.2f}",
                        'max': f"{clean_target.max():.2f}",
                        'median': f"{clean_target.median():.2f}",
                        'units': 'units'
                    })
                else:
                    target_stats.update({
                        'mean': 'N/A',
                        'std': 'N/A',
                        'min': 'N/A',
                        'max': 'N/A',
                        'median': 'N/A',
                        'units': 'unknown'
                    })
            except:
                target_stats.update({
                    'mean': 'N/A',
                    'std': 'N/A',
                    'min': 'N/A',
                    'max': 'N/A',
                    'median': 'N/A',
                    'units': 'unknown'
                })

        return jsonify({
            'success': True,
            'feature_column': feature_col,
            'target_column': target_col,
            'model_type': model_type,
            'problem_description': problem_desc,
            'target_stats': target_stats,
            'prediction_context': prediction_context,
            'message': f'Model configured as {problem_desc}. Predicting: {target_col}'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in setup_model: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/train_model', methods=['POST'])
def train_model():
    global prediction_model, model_trained, reference_data, model_type, column_info, time_series_info, scaler, prediction_context

    try:
        if reference_data is None or column_info is None:
            return jsonify({'success': False, 'error': 'Please setup model first by selecting columns'})

        df = reference_data['df'].copy()
        feature_col = column_info['feature']
        target_col = column_info['target']

        df[target_col] = clean_numeric_column(df[target_col])
        df = df.dropna(subset=[target_col])

        if len(df) < 5:
            return jsonify({'success': False, 'error': 'Not enough data for training. Need at least 5 rows.'})

        if model_type == 'timeseries':
            try:
                ts_data = prepare_timeseries_data(df, target_col, feature_col)
                dates = None
                if feature_col and feature_col in df.columns:
                    df_copy = df.copy()
                    df_copy[feature_col] = pd.to_datetime(df_copy[feature_col], errors='coerce')
                    dates = df_copy[feature_col].dropna().sort_values().values

                series = ts_data[target_col].values
                prediction_model = train_time_series_model(series, dates)

                last_date_str = None
                if dates is not None and len(dates) > 0:
                    last_date = dates[-1]
                    last_date_str = convert_to_json_serializable(last_date)
                elif 'last_date' in prediction_model:
                    last_date_str = prediction_model['last_date']

                prediction_context['last_date'] = last_date_str

                trend = prediction_model.get('trend', 0)
                last_value = series[-1] if len(series) > 0 else 0

                model_name = "Time Series Model"

                if len(series) > 5:
                    y_pred = []
                    for i in range(len(series)):
                        if i < 3:
                            y_pred.append(series[i])
                        else:
                            pred = np.mean(series[max(0, i - 3):i])
                            y_pred.append(pred)

                    mse = mean_squared_error(series[3:], y_pred[3:])
                    mae = mean_absolute_error(series[3:], y_pred[3:])
                    r2 = r2_score(series[3:], y_pred[3:])
                    metrics = {
                        'mse': f"{mse:.4f}",
                        'rmse': f"{np.sqrt(mse):.4f}",
                        'mae': f"{mae:.4f}",
                        'r2_score': f"{r2:.4f}",
                        'samples': len(series)
                    }
                else:
                    metrics = {
                        'samples': len(series),
                        'note': 'Time series model trained successfully'
                    }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(series),
                    'data_range': f"{len(df)} time periods",
                    'prediction_units': 'units',
                    'last_date': last_date_str,
                    'trend': float(trend),
                    'last_value': float(last_value)
                })

            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error preparing time series data: {str(e)}")
                print(f"Traceback: {error_details}")
                return jsonify({'success': False, 'error': f'Error preparing time series data: {str(e)}'})

        elif model_type == 'classification':
            try:
                if feature_col and feature_col in df.columns:
                    X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                else:
                    X = np.arange(len(df)).reshape(-1, 1)

                y = df[target_col].copy()
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                n_classes = len(np.unique(y))
                if n_classes == 2:
                    prediction_model = LogisticRegression(max_iter=1000)
                    model_name = "Binary Logistic Regression"
                else:
                    prediction_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                    model_name = f"Random Forest ({n_classes} classes)"

                prediction_model.fit(X_scaled, y)
                y_pred = prediction_model.predict(X_scaled)
                accuracy = accuracy_score(y, y_pred)
                metrics = {
                    'accuracy': f"{accuracy * 100:.2f}%",
                    'classes': str(n_classes),
                    'samples': len(y)
                }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(y),
                    'num_classes': n_classes,
                    'accuracy': f"{accuracy * 100:.2f}%"
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error preparing classification data: {str(e)}'})

        else:
            try:
                if feature_col and feature_col in df.columns:
                    X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                else:
                    X = np.arange(len(df)).reshape(-1, 1)

                y = df[target_col].values.astype(float)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                prediction_model = Ridge(alpha=1.0)
                prediction_model.fit(X_scaled, y)
                model_name = "Ridge Regression"

                y_pred = prediction_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                metrics = {
                    'mse': f"{mse:.4f}",
                    'rmse': f"{np.sqrt(mse):.4f}",
                    'mae': f"{mae:.4f}",
                    'r2_score': f"{r2:.4f}",
                    'samples': len(y)
                }

                prediction_context.update({
                    'model_name': model_name,
                    'training_samples': len(y),
                    'r2_score': f"{r2:.4f}",
                    'prediction_units': 'units'
                })

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error preparing regression data: {str(e)}'})

        try:
            plot_data = create_visualization(df, feature_col, target_col, y_pred if 'y_pred' in locals() else None,
                                             model_type)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            plot_data = None

        model_trained = True

        serializable_prediction_context = {}
        for key, value in prediction_context.items():
            serializable_prediction_context[key] = convert_to_json_serializable(value)

        return jsonify({
            'success': True,
            'model_name': model_name,
            'model_type': model_type,
            'metrics': metrics,
            'plot': plot_data if plot_data else '',
            'prediction_context': serializable_prediction_context,
            'message': f'Model trained successfully with {len(df)} samples. Ready to predict {target_col}.'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in train_model: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Training error: {str(e)}'})


@app.route('/predict_single', methods=['POST'])
def predict_single():
    global prediction_model, model_trained, reference_data, model_type, column_info, time_series_info, scaler, prediction_context

    try:
        if not model_trained:
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'})

        data = request.get_json()
        input_value = data.get('feature_value', '').strip()

        if not input_value:
            return jsonify({'success': False, 'error': 'Please enter a date or value'})

        if model_type == 'timeseries':
            try:
                try:
                    selected_date = pd.to_datetime(input_value)
                    is_date = True
                except:
                    try:
                        feature_num = float(input_value)
                        is_date = False
                    except:
                        return jsonify({'success': False, 'error': 'Please enter a valid date or numeric value'})

                df = reference_data['df'].copy()
                target_col = column_info['target']
                feature_col = column_info['feature']

                ts_data = prepare_timeseries_data(df, target_col, feature_col)
                series = ts_data[target_col].values

                if len(series) == 0:
                    return jsonify({'success': False, 'error': 'No valid time series data available'})

                if is_date:
                    selected_date = pd.to_datetime(input_value)

                    if feature_col and feature_col in df.columns:
                        df[feature_col] = pd.to_datetime(df[feature_col], errors='coerce')
                        valid_dates = df[feature_col].dropna()
                        if len(valid_dates) > 0:
                            last_date = valid_dates.max()
                        else:
                            last_date = None
                    else:
                        last_date = None

                    if last_date is None and 'last_date' in prediction_context:
                        last_date_str = prediction_context['last_date']
                        if last_date_str:
                            try:
                                last_date = pd.to_datetime(last_date_str)
                            except:
                                last_date = None

                    if last_date:
                        last_date_pd = pd.Timestamp(last_date)
                        selected_date_pd = pd.Timestamp(selected_date)

                        months_diff = (selected_date_pd.year - last_date_pd.year) * 12 + (
                                selected_date_pd.month - last_date_pd.month)

                        if selected_date_pd.day < last_date_pd.day:
                            months_diff -= 1

                        if months_diff <= 0:
                            if feature_col and feature_col in df.columns:
                                df[feature_col] = pd.to_datetime(df[feature_col], errors='coerce')
                                date_diffs = (df[feature_col] - selected_date).abs()
                                nearest_idx = date_diffs.idxmin()
                                prediction = float(df[target_col].iloc[nearest_idx])
                                prediction_note = f"Historical value from {df[feature_col].iloc[nearest_idx].strftime('%m/%d/%Y')}"
                            else:
                                prediction = float(series[-1])
                                prediction_note = "Most recent value"
                            prediction_type = "historical"
                            steps = 0
                        else:
                            steps = max(1, months_diff)
                            all_predictions = predict_time_series(prediction_model, steps=steps)
                            prediction = float(all_predictions[-1])

                            if steps == 1:
                                prediction_note = f"Prediction for {selected_date.strftime('%m/%d/%Y')} (next month)"
                            else:
                                prediction_note = f"Prediction for {selected_date.strftime('%m/%d/%Y')} ({steps} months ahead)"

                            prediction_type = "future"
                    else:
                        steps = 1
                        predictions = predict_time_series(prediction_model, steps=steps)
                        prediction = float(predictions[0])

                        month_factor = 1 + 0.01 * (selected_date.month - 6) / 6
                        prediction = prediction * month_factor

                        prediction_note = f"Prediction for {selected_date.strftime('%m/%d/%Y')}"
                        prediction_type = "future"

                    selected_date_str = selected_date.strftime('%m/%d/%Y')
                else:
                    feature_num = float(input_value)

                    if feature_col and feature_col in df.columns and feature_col in reference_data['date_columns']:
                        try:
                            numeric_series = clean_numeric_column(df[feature_col].dt.year)
                            closest_idx = (numeric_series - feature_num).abs().idxmin()
                            prediction = float(df[target_col].iloc[closest_idx])
                            prediction_note = f"Value from year {int(feature_num)}"
                        except:
                            predictions = predict_time_series(prediction_model, steps=1)
                            prediction = float(predictions[0])
                            prediction_note = f"Prediction for input value {feature_num}"
                    else:
                        if scaler:
                            X_input = scaler.transform([[feature_num]])
                        else:
                            X_input = [[feature_num]]

                        prediction = float(prediction_model.predict(X_input)[0])
                        prediction_note = f"Prediction for {feature_col}: {feature_num}"

                    prediction_type = "regression"
                    steps = 0
                    selected_date_str = input_value

                last_actual = float(series[-1]) if len(series) > 0 else None
                units = prediction_context.get('prediction_units', 'units')

                result = {
                    'prediction': float(prediction),
                    'prediction_formatted': f"{prediction:.2f}",
                    'selected_date': selected_date_str,
                    'note': prediction_note,
                    'last_actual': last_actual,
                    'prediction_type': prediction_type,
                    'units': units,
                    'target_name': target_col,
                    'target_description': prediction_context.get('target_description', 'value'),
                    'months_ahead': steps
                }

                if last_actual is not None and prediction_type == "future":
                    change = prediction - last_actual
                    change_percent = (change / last_actual * 100) if last_actual != 0 else 0
                    result['change'] = f"{change:.4f}"
                    result['change_percent'] = f"{change_percent:.4f}%"
                    result['last_actual_formatted'] = f"{last_actual:.2f}"
                    result['direction'] = 'up' if change > 0 else 'down' if change < 0 else 'unchanged'

            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error in time series prediction: {str(e)}")
                print(f"Traceback: {error_details}")
                return jsonify({'success': False, 'error': f'Invalid input format or prediction error: {str(e)}'})

        elif model_type == 'classification':
            try:
                feature_num = float(input_value)
            except:
                return jsonify({'success': False, 'error': 'Please enter a numeric value for classification'})

            if scaler:
                X_input = scaler.transform([[feature_num]])
            else:
                X_input = [[feature_num]]

            prediction = int(prediction_model.predict(X_input)[0])

            if hasattr(prediction_model, 'predict_proba'):
                probabilities = [float(x) for x in prediction_model.predict_proba(X_input)[0]]
                result = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'confidence': f"{max(probabilities) * 100:.1f}%",
                    'target_name': column_info['target'],
                    'target_description': prediction_context.get('target_description', 'category')
                }
            else:
                result = {
                    'prediction': prediction,
                    'target_name': column_info['target'],
                    'target_description': prediction_context.get('target_description', 'category')
                }

        else:
            try:
                feature_num = float(input_value)
            except:
                return jsonify({'success': False, 'error': 'Please enter a numeric value for regression'})

            if scaler:
                X_input = scaler.transform([[feature_num]])
            else:
                X_input = [[feature_num]]

            prediction = float(prediction_model.predict(X_input)[0])
            result = {
                'prediction': prediction,
                'prediction_formatted': f"{prediction:.2f}",
                'target_name': column_info['target'],
                'target_description': prediction_context.get('target_description', 'value'),
                'units': prediction_context.get('prediction_units', 'units')
            }

        plot_data = None
        if reference_data and column_info:
            df = reference_data['df'].copy()
            feature_col = column_info['feature']
            target_col = column_info['target']

            try:
                if model_type == 'timeseries':
                    plot_data = create_visualization(df, feature_col, target_col, model_type=model_type)
                else:
                    if feature_col and feature_col in df.columns:
                        X = clean_numeric_column(df[feature_col]).values.reshape(-1, 1)
                    else:
                        X = np.arange(len(df)).reshape(-1, 1)

                    if scaler and len(X) > 0:
                        X_scaled = scaler.transform(X)
                        y_pred = prediction_model.predict(X_scaled)
                    elif len(X) > 0:
                        y_pred = prediction_model.predict(X)
                    else:
                        y_pred = None

                    if y_pred is not None:
                        plot_data = create_visualization(df, feature_col, target_col, y_pred, model_type)
            except Exception as e:
                print(f"Error creating prediction visualization: {e}")
                plot_data = None

        serializable_prediction_context = {}
        for key, value in prediction_context.items():
            serializable_prediction_context[key] = convert_to_json_serializable(value)

        return jsonify({
            'success': True,
            'feature_value': input_value,
            'model_type': model_type,
            'result': result,
            'prediction_context': serializable_prediction_context,
            'plot': plot_data if plot_data else '',
            'message': 'Prediction successful'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict_single: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})


@app.route('/predict_future', methods=['POST'])
def predict_future():
    global prediction_model, model_trained, reference_data, model_type, column_info, scaler, prediction_context

    try:
        if not model_trained or model_type != 'timeseries':
            return jsonify({'success': False, 'error': 'Time series model not trained or available'})

        data = request.get_json()
        periods = int(data.get('periods', 5))

        if periods < 1 or periods > 100:
            return jsonify({'success': False, 'error': 'Please enter a number between 1 and 100 for periods'})

        df = reference_data['df'].copy()
        target_col = column_info['target']
        feature_col = column_info['feature']

        ts_data = prepare_timeseries_data(df, target_col, feature_col)
        series = ts_data[target_col].values

        if len(series) == 0:
            return jsonify({'success': False, 'error': 'Not enough data for future predictions'})

        future_predictions = predict_time_series(prediction_model, steps=periods)

        future_dates = []
        current_date = datetime.now()

        if current_date.month == 12:
            next_month_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month_date = current_date.replace(month=current_date.month + 1, day=1)

        for i in range(periods):
            month_offset = i
            year_offset = month_offset // 12
            month_in_year = (next_month_date.month + month_offset % 12 - 1) % 12 + 1

            if month_in_year < next_month_date.month:
                year_offset += 1

            future_date = next_month_date.replace(
                year=next_month_date.year + year_offset,
                month=month_in_year,
                day=1
            )
            formatted_date = future_date.strftime('%m/1/%Y')
            future_dates.append(formatted_date)

        future_array = np.array(future_predictions)
        if len(future_array) > 0:
            stats = {
                'average': f"{future_array.mean():.2f}",
                'minimum': f"{future_array.min():.2f}",
                'maximum': f"{future_array.max():.2f}",
                'trend': 'increasing' if len(future_array) > 1 and future_array[-1] > future_array[0] else 'decreasing',
                'units': prediction_context.get('prediction_units', 'units'),
                'first_prediction': f"{future_predictions[0]:.2f}",
                'last_prediction': f"{future_predictions[-1]:.2f}",
                'total_change': f"{future_predictions[-1] - future_predictions[0]:.2f}"
            }
        else:
            stats = {
                'average': '0.00',
                'minimum': '0.00',
                'maximum': '0.00',
                'trend': 'stable',
                'units': prediction_context.get('prediction_units', 'units')
            }

        try:
            plot_dates = [datetime.strptime(date_str, '%m/%d/%Y') for date_str in future_dates]
            plot_data = create_visualization(
                df, feature_col, target_col,
                model_type=model_type,
                future_predictions=future_predictions,
                future_dates=plot_dates
            )
        except Exception as e:
            print(f"Error creating visualization: {e}")
            plot_data = None

        results = []
        for i, pred in enumerate(future_predictions):
            results.append({
                'period': i + 1,
                'date': future_dates[i],
                'prediction': float(pred),
                'formatted': f"{pred:.2f}",
                'units': prediction_context.get('prediction_units', 'units'),
                'change_from_first': f"{pred - future_predictions[0]:.2f}" if i > 0 else "0.00"
            })

        prediction_explanation = {
            'what_is_predicted': target_col,
            'description': prediction_context.get('target_description', 'value'),
            'units': prediction_context.get('prediction_units', 'units'),
            'model_used': prediction_context.get('model_name', 'Time Series Model'),
            'based_on': f"{len(series)} historical data points",
            'frequency': 'monthly',
            'trend_direction': stats['trend']
        }

        serializable_prediction_context = {}
        for key, value in prediction_context.items():
            serializable_prediction_context[key] = convert_to_json_serializable(value)

        return jsonify({
            'success': True,
            'periods': periods,
            'predictions': results,
            'statistics': stats,
            'prediction_explanation': prediction_explanation,
            'prediction_context': serializable_prediction_context,
            'plot': plot_data if plot_data else '',
            'message': f'Generated {periods} future monthly predictions for {target_col}.'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict_future: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Future prediction error: {str(e)}'})


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make batch predictions from uploaded file"""
    global prediction_model, model_trained, reference_data, model_type, column_info, scaler, prediction_context

    try:
        if not model_trained:
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'})

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'})

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the uploaded file with flexible handling
        try:
            feature_col = column_info['feature'] if column_info else None
            new_df = read_flexible_file(filepath, feature_col)
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error reading batch file: {str(e)}")
            print(f"Traceback: {error_details}")
            return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})

        if len(new_df) == 0:
            return jsonify({'success': False, 'error': 'File is empty'})

        # Get feature and target columns
        feature_col = column_info['feature'] if column_info else None
        target_col = column_info['target'] if column_info else None

        print(f"DEBUG: Feature column: {feature_col}, Target column: {target_col}")
        print(f"DEBUG: New DataFrame columns: {new_df.columns.tolist()}")
        print(f"DEBUG: New DataFrame shape: {new_df.shape}")

        # Prepare results
        predictions = []

        # Helper function to get input value from a row
        def get_input_value(row, idx):
            """Extract input value from row, trying multiple strategies"""
            # Strategy 1: Use feature column if specified and exists
            if feature_col and feature_col in new_df.columns:
                val = row[feature_col]
                if pd.notna(val):
                    return str(val)

            # Strategy 2: Try to find column with similar name to feature column
            if feature_col:
                for col in new_df.columns:
                    if col.lower() == feature_col.lower() or feature_col.lower() in col.lower():
                        val = row[col]
                        if pd.notna(val):
                            return str(val)

            # Strategy 3: Use first column
            first_col = new_df.columns[0]
            val = row[first_col]
            if pd.notna(val):
                return str(val)

            # Strategy 4: Use row index as fallback
            return str(idx)

        if model_type == 'timeseries':
            # For time series, each row might be a date or numeric value
            for idx, row in new_df.iterrows():
                input_value = get_input_value(row, idx)

                try:
                    # Try to parse as date first
                    try:
                        selected_date = pd.to_datetime(input_value)

                        df = reference_data['df'].copy()
                        if feature_col and feature_col in df.columns:
                            df[feature_col] = pd.to_datetime(df[feature_col], errors='coerce')
                            valid_dates = df[feature_col].dropna()
                            if len(valid_dates) > 0:
                                last_date = valid_dates.max()

                                last_date_pd = pd.Timestamp(last_date)
                                selected_date_pd = pd.Timestamp(selected_date)

                                months_diff = (selected_date_pd.year - last_date_pd.year) * 12 + (
                                        selected_date_pd.month - last_date_pd.month)

                                if selected_date_pd.day < last_date_pd.day:
                                    months_diff -= 1

                                if months_diff <= 0:
                                    date_diffs = (df[feature_col] - selected_date).abs()
                                    nearest_idx = date_diffs.idxmin()
                                    prediction = float(df[target_col].iloc[nearest_idx])
                                    prediction_type = "historical"
                                else:
                                    steps = max(1, months_diff)
                                    all_predictions = predict_time_series(prediction_model, steps=steps)
                                    prediction = float(all_predictions[-1])
                                    prediction_type = "future"
                            else:
                                steps = 1
                                all_predictions = predict_time_series(prediction_model, steps=steps)
                                prediction = float(all_predictions[-1])
                                prediction_type = "future"
                        else:
                            steps = 1
                            all_predictions = predict_time_series(prediction_model, steps=steps)
                            prediction = float(all_predictions[-1])
                            prediction_type = "future"

                        predictions.append({
                            'input': input_value,
                            'prediction': prediction,
                            'type': prediction_type,
                            'formatted': f"{prediction:.2f}"
                        })

                    except:
                        # If not a date, try as numeric
                        try:
                            feature_num = extract_numeric_value(input_value, idx)
                            if scaler:
                                X_input = scaler.transform([[feature_num]])
                            else:
                                X_input = [[feature_num]]

                            prediction = float(prediction_model.predict(X_input)[0])
                            predictions.append({
                                'input': input_value,
                                'prediction': prediction,
                                'type': 'regression',
                                'formatted': f"{prediction:.2f}"
                            })
                        except Exception as e2:
                            print(f"Error processing numeric value '{input_value}': {e2}")
                            predictions.append({
                                'input': input_value,
                                'prediction': None,
                                'error': 'Invalid input format',
                                'type': 'error'
                            })

                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    predictions.append({
                        'input': input_value,
                        'prediction': None,
                        'error': str(e),
                        'type': 'error'
                    })

        elif model_type == 'classification':
            # For classification
            for idx, row in new_df.iterrows():
                input_value = get_input_value(row, idx)

                try:
                    feature_num = extract_numeric_value(input_value, idx)

                    if scaler:
                        X_input = scaler.transform([[feature_num]])
                    else:
                        X_input = [[feature_num]]

                    y_pred = prediction_model.predict(X_input)[0]
                    prediction = int(y_pred)

                    if hasattr(prediction_model, 'predict_proba'):
                        y_proba = prediction_model.predict_proba(X_input)[0]
                        confidence = float(max(y_proba))
                        predictions.append({
                            'input': input_value,
                            'prediction': prediction,
                            'confidence': f"{confidence * 100:.1f}%",
                            'formatted': f"Class {prediction} ({confidence * 100:.1f}%)"
                        })
                    else:
                        predictions.append({
                            'input': input_value,
                            'prediction': prediction,
                            'formatted': f"Class {prediction}"
                        })

                except Exception as e:
                    print(f"Error processing row {idx} with input '{input_value}': {e}")
                    predictions.append({
                        'input': input_value,
                        'prediction': None,
                        'error': str(e),
                        'type': 'error'
                    })

        else:  # regression
            for idx, row in new_df.iterrows():
                input_value = get_input_value(row, idx)

                try:
                    feature_num = extract_numeric_value(input_value, idx)

                    if scaler:
                        X_input = scaler.transform([[feature_num]])
                    else:
                        X_input = [[feature_num]]

                    y_pred = prediction_model.predict(X_input)[0]
                    prediction = float(y_pred)

                    predictions.append({
                        'input': input_value,
                        'prediction': prediction,
                        'formatted': f"{prediction:.2f}"
                    })

                except Exception as e:
                    print(f"Error processing row {idx} with input '{input_value}': {e}")
                    predictions.append({
                        'input': input_value,
                        'prediction': None,
                        'error': str(e),
                        'type': 'error'
                    })

        # Create results dataframe
        results_df = pd.DataFrame(predictions)

        # Save results to CSV
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'batch_predictions.csv')
        results_df.to_csv(results_path, index=False)

        # Calculate statistics
        stats = {}
        if predictions:
            pred_values = [p['prediction'] for p in predictions if p['prediction'] is not None]
            if pred_values:
                stats = {
                    'total_samples': len(predictions),
                    'successful_predictions': len(pred_values),
                    'failed_predictions': len(predictions) - len(pred_values),
                    'average_prediction': f"{np.mean(pred_values):.4f}",
                    'min_prediction': f"{np.min(pred_values):.4f}",
                    'max_prediction': f"{np.max(pred_values):.4f}",
                    'std_prediction': f"{np.std(pred_values):.4f}"
                }

                if model_type == 'classification':
                    # Add class distribution for classification
                    class_counts = {}
                    for p in predictions:
                        if p['prediction'] is not None:
                            cls = p['prediction']
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                    stats['class_counts'] = class_counts

        # Create preview HTML - show all rows if less than 20, otherwise first 10
        preview_count = min(10, len(results_df))
        preview_df = results_df.head(preview_count).copy()
        preview_html = preview_df.to_html(
            classes='table table-striped table-bordered',
            index=False,
            na_rep='NaN'
        )

        return jsonify({
            'success': True,
            'model_type': model_type,
            'predictions': predictions[:preview_count],
            'statistics': stats,
            'preview': preview_html,
            'total_rows': len(results_df),
            'download_url': '/download_batch_results',
            'message': f'Processed {len(predictions)} rows. {stats.get("successful_predictions", 0)} successful predictions.'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in predict_batch: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Batch prediction error: {str(e)}'})


@app.route('/download_batch_results', methods=['GET'])
def download_batch_results():
    """Download batch prediction results"""
    try:
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'batch_predictions.csv')
        if os.path.exists(results_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'batch_predictions.csv', as_attachment=True)
        else:
            return jsonify({'success': False, 'error': 'No batch results available'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Download error: {str(e)}'})


@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    """Analyze the uploaded data and provide insights"""
    global reference_data

    try:
        if reference_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded. Please upload data first.'})

        df = reference_data['df'].copy()

        # Basic analysis
        analysis = {
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'column_stats': {},
            'data_types': {}
        }

        # Data types
        for col in df.columns:
            analysis['data_types'][col] = str(df[col].dtype)

        # Column statistics
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    clean_col = clean_numeric_column(df[col])
                    if clean_col.notna().sum() > 0:
                        analysis['column_stats'][col] = {
                            'mean': f"{clean_col.mean():.2f}",
                            'std': f"{clean_col.std():.2f}",
                            'min': f"{clean_col.min():.2f}",
                            'max': f"{clean_col.max():.2f}",
                            'median': f"{clean_col.median():.2f}",
                            'missing': int(clean_col.isna().sum())
                        }
                else:
                    analysis['column_stats'][col] = {
                        'unique_values': int(df[col].nunique()),
                        'missing': int(df[col].isna().sum())
                    }
            except:
                analysis['column_stats'][col] = {
                    'error': 'Cannot calculate statistics'
                }

        return jsonify({
            'success': True,
            'analysis': analysis,
            'message': f'Data analysis completed for {len(df)} rows and {len(df.columns)} columns'
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in analyze_data: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})


@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all loaded data and models"""
    global reference_data, prediction_model, model_type, model_trained, column_info, time_series_info, scaler, prediction_context

    reference_data = None
    prediction_model = None
    model_type = None
    model_trained = False
    column_info = None
    time_series_info = None
    scaler = None
    prediction_context = {}

    return jsonify({
        'success': True,
        'message': 'All data and models cleared successfully'
    })


@app.route('/download_template', methods=['GET'])
def download_template():
    """Download a sample template file"""
    try:
        # Create a sample template
        sample_data = {
            'Date': pd.date_range(start='2023-01-01', periods=12, freq='MS').strftime('%m/%d/%Y').tolist(),
            'Sales': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],
            'Price': [10.5, 11.0, 10.8, 11.2, 11.5, 11.8, 12.0, 12.2, 12.5, 12.8, 13.0, 13.2],
            'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        }

        df = pd.DataFrame(sample_data)
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_template.csv')
        df.to_csv(template_path, index=False)

        return send_from_directory(app.config['UPLOAD_FOLDER'], 'prediction_template.csv', as_attachment=True)

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error creating template: {str(e)}'})


if __name__ == '__main__':
    print("=" * 60)
    print("🌐 Universal Prediction System")
    print("=" * 60)
    print("Features:")
    print("  1. Upload CSV/Excel/JSON data")
    print("  2. Automatic model type detection")
    print("  3. Time series, classification, and regression")
    print("  4. Single predictions")
    print("  5. Future predictions (time series)")
    print("  6. Batch predictions from files (with or without headers)")
    print("  7. Data analysis and visualization")
    print("\n🚀 Access: http://127.0.0.1:5001")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5001)