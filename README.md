# Universal Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![StatsModels](https://img.shields.io/badge/StatsModels-Time_Series-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An intelligent, multi-domain prediction platform supporting Time Series Forecasting, Classification, and Regression — all in one unified system.

---

## Overview

**Live Project Repository:** [[ ai-auto-analyzer ](https://github.com/Hamzah-20/ai-auto-analyzer)]

Universal Prediction System is a full-stack machine learning platform that automatically detects your data type and applies the appropriate prediction model. Whether you're forecasting sales, classifying customer behavior, or predicting numerical values — this system handles it all.

The project combines:
- Automatic model type detection
- Time Series Models ( Trend-based forecasting || Moving average smoothing || Custom seasonal adjustment logic )
- Classification models (Logistic Regression, Random Forest)
- Regression models (Ridge Regression)
- Batch prediction capabilities
- Interactive visualizations
- Flexible file handling (CSV, Excel, JSON)

The goal is to provide a **universal, easy-to-use prediction system** that works with any structured data.

---

## Why This Project Matters

This project demonstrates real-world Machine Learning engineering by combining:

- End-to-end ML pipeline development (from raw data to prediction)
- Real-world data preprocessing and flexible file handling
- Automated model selection (Classification / Regression / Time Series)
- Production-ready Flask web deployment
- Full-stack AI system design with interactive UI
- Scalable architecture suitable for real business use cases

---

## Key Features

### AI & Machine Learning
- **Automatic Model Type Detection** — Identifies if your data is time series, classification, or regression
- **Time Series Models** — ARIMA, Exponential Smoothing (Holt-Winters)
- **Classification Models** — Logistic Regression, Random Forest Classifier
- **Regression Models** — Ridge Regression
- **Ensemble Approaches** — Automatic model selection based on data characteristics
- **Feature Scaling** — Automatic StandardScaler application

### Time Series Forecasting
- Automatic date column detection
- Monthly trend calculation
- Future period predictions (1-100 months)
- Seasonal pattern recognition
- Moving average with trend adjustment

### Web Application
- Drag-and-drop file upload
- Dynamic form generation
- Real-time predictions
- Batch CSV/Excel prediction
- Interactive data preview
- Responsive UI/UX with Bootstrap

### Visualization
- Historical data visualization
- Prediction overlays
- Future forecast plotting
- Regression line visualization
- Classification decision boundaries

### Data Flexibility
- CSV (with or without headers)
- Excel files (.xlsx, .xls)
- JSON files
- Automatic column detection
- Intelligent header detection

---

## How Model Detection Works

The system automatically determines your data type:

### Time Series Detection
- Looks for date columns (Date, Time, Month, Year, etc.)
- Checks for date patterns (MM/DD/YYYY, YYYY-MM-DD, etc.)
- If date column found → Time Series model

### Classification Detection
- Target column has 2 unique values → Binary Classification
- Target column has ≤10 unique values (and is string) → Multi-class Classification
- Target column contains binary patterns (Yes/No, True/False, 0/1) → Classification

### Regression Detection
- All other numeric targets → Regression model

---

## Model Performance Comparison

### Time Series Models

| Model Type | Best For | Strengths | Limitations |
|------------|----------|-----------|--------------|
| ARIMA (1,1,1) with seasonal | 24+ months data | Handles seasonality and trends | Requires more data |
| Holt-Winters | 12+ months data | Good for seasonal patterns | Can overfit with noise |
| Trend + Moving Average | 3-12 months data | Works with limited data | Less accurate for complex patterns |

### Classification Models

| Model Type | Best For | Output |
|------------|----------|--------|
| Logistic Regression | Binary classification | Class + Probabilities |
| Random Forest | Multi-class classification | Class + Confidence |

### Regression Models

| Model Type | Best For | Output |
|------------|----------|--------|
| Ridge Regression | Linear relationships | Continuous value prediction |

---

## Technologies Used

### Backend
- Python 3.8+
- Flask (Web Framework)
- Pandas (Data manipulation)
- NumPy (Numerical operations)

### Machine Learning
- Scikit-learn (ML models, preprocessing)
- StatsModels (ARIMA, time series)
- Imbalanced-learn (future enhancement)

### Visualization
- Matplotlib
- Seaborn

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- Font Awesome

---

## Project Structure

```bash
universal-prediction-system/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── templates/
│   └── index.html             # Web interface
│
├── uploads/                   # Uploaded files directory
│   ├── [user uploaded files]
│   └── batch_predictions.csv  # Batch prediction results
│
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/universal-prediction-system.git
cd universal-prediction-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Then open in your browser:

```
http://127.0.0.1:5001
```

---

## Requirements

```
flask==2.3.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
statsmodels==0.14.0
matplotlib==3.7.2
seaborn==0.12.2
openpyxl==3.1.2
werkzeug==2.3.0
```

---

## Usage Guide

### Step 1: Upload Your Data
- Supported formats: CSV, Excel, JSON
- The system automatically detects headers
- Preview shows first 10 rows with all columns

### Step 2: Configure Model
- Select **Target Column** (what you want to predict)
- Select **Feature Column** (optional, for time series select date column)
- System automatically detects model type

### Step 3: Train Model
- Click "Train Prediction Model"
- Model trains based on your data
- Visualizations and metrics are displayed

### Step 4: Make Predictions

#### Single Prediction
- **Time Series:** Enter a future date
- **Classification/Regression:** Enter a numeric value

#### Future Predictions (Time Series only)
- Enter number of periods (1-100)
- View all predictions with statistics

#### Batch Predictions
- Upload a file with new data
- System processes all rows
- Download results as CSV

---

## Example Use Cases

###  Sales Forecasting (Time Series)
```
File: monthly_sales.csv
Columns: Date, Sales
Target: Sales
Feature: Date
→ Predicts future monthly sales
```

###  Customer Churn (Classification)
```
File: customer_data.csv
Columns: Tenure, MonthlyCharges, TotalCharges, Churn
Target: Churn (Yes/No)
Feature: MonthlyCharges
→ Predicts if customer will churn
```

###  House Price Prediction (Regression)
```
File: housing.csv
Columns: SqFt, Bedrooms, Price
Target: Price
Feature: SqFt
→ Predicts house price based on size
```

###  Stock Price Analysis (Time Series)
```
File: stock_prices.csv
Columns: Date, Close, Volume
Target: Close
Feature: Date
→ Predicts future stock prices
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload_data` | POST | Upload data file |
| `/setup_model` | POST | Configure model columns |
| `/train_model` | POST | Train prediction model |
| `/predict_single` | POST | Make single prediction |
| `/predict_future` | POST | Generate future predictions |
| `/predict_batch` | POST | Batch predictions from file |
| `/analyze_data` | POST | Get data statistics |
| `/clear_data` | POST | Reset all data |
| `/download_batch_results` | GET | Download batch predictions |
| `/download_template` | GET | Download sample template |

---

## Data Format Requirements

### Time Series Data
- Must have a date column (recognizes: Date, Time, Month, Year, etc.)
- Date formats supported: MM/DD/YYYY, YYYY-MM-DD, DD-MM-YYYY
- Target column should be numeric

### Classification Data
- Target column should have limited unique values (2-10)
- Binary examples: Yes/No, True/False, 0/1, Pass/Fail
- Feature column can be numeric or categorical

### Regression Data
- Target column must be numeric
- Feature column should be numeric

---

## Important Engineering Practices

- **Automatic Header Detection** — Works with or without CSV headers
- **Flexible Column Matching** — Finds columns even with different names
- **Error Handling** — Graceful fallbacks for malformed data
- **Data Validation** — Comprehensive input validation
- **JSON Serialization** — Proper handling of numpy/pandas types
- **File Security** — Secure filename handling with werkzeug

---

##  System Preview

###  Dashboard
<img width="1350" height="1412" alt="127 0 0 1_5001_ (2)" src="https://github.com/user-attachments/assets/aea058c6-123e-44f2-afc4-555b1fd73dc9" />

###  Data Upload & Preview
<img width="1350" height="2332" alt="127 0 0 1_5001_ (3)" src="https://github.com/user-attachments/assets/b8e5c7da-278f-45f2-80ab-d9371e61d117" />

### Data Analysis
<img width="1143" height="653" alt="image" src="https://github.com/user-attachments/assets/402e890e-cf01-45a4-abfc-cb57588104a7" />

###  Model Training
<img width="1162" height="662" alt="image" src="https://github.com/user-attachments/assets/e037d92a-9492-4ab1-b248-bb23ff74f85e" />

###  Prediction Results
<img width="1153" height="1630" alt="127 0 0 1_5001_ (4)" src="https://github.com/user-attachments/assets/c46b8985-282a-498a-bc66-ff52e4d71d05" />

### Batch Predictions
<img width="659" height="752" alt="image" src="https://github.com/user-attachments/assets/cdef4b12-118c-45a5-b996-06c01163b726" />

###  Future Forecasting (Time Series)
<img width="1350" height="4440" alt="127 0 0 1_5001_ (1)" src="https://github.com/user-attachments/assets/98e12450-77b6-4594-8df9-8716948c998b" />

##  System Workflow

1. Upload dataset
2. Automatic data analysis
3. Model type detection (Regression / Classification / Time Series)
4. Train ML model
5. Single prediction OR batch prediction
6. Forecast future values (if time series)

---

## Key Insights from Implementation

- **Automatic detection** saves users from complex configuration
- **Time series forecasting** works well with as few as 6 months of data
- **Batch prediction** enables processing hundreds of records at once
- **Flexible file handling** supports real-world messy data
- **Visual feedback** helps users understand model behavior

---

## Limitations & Future Improvements

### Current Limitations
- Time series models require at least 3 data points
- Classification only works with numeric or binary categorical targets
- Seasonal patterns need at least 12 months of data for ARIMA

### Future Improvements
- [ ] Deep Learning integration (LSTM for time series)
- [ ] Hyperparameter optimization
- [ ] More classification algorithms (SVM, XGBoost)
- [ ] Export models as pickle files
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Real-time API streaming
- [ ] Database integration (PostgreSQL, MongoDB)
- [ ] User authentication and saved models
- [ ] Automated report generation
- [ ] Email/Slack notifications for predictions

---

## Author

**Your Name**

AI & Machine Learning Engineer  
Focused on Applied AI, Predictive Analytics, and Intelligent Systems

GitHub: [Hamzah-20](https://github.com/Hamzah-20)

LinkedIn: [Hamzah Al-basyouni](https://www.linkedin.com/in/hamzah-al-basyouni-967122369)

---

## License

MIT License - feel free to use for educational, research, and commercial purposes.

---

## Acknowledgments

- Scikit-learn team for excellent ML tools
- StatsModels for time series capabilities
- Flask community for web framework
- Bootstrap for responsive design

---

## Support

For issues, feature requests, or contributions:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the maintainer directly

---

## Version History

- **v1.0.0** (2024) — Initial release
  - Automatic model detection
  - Time series, classification, regression support
  - Batch predictions
  - Interactive visualizations

---

##  Star the Project

If you find this useful, please star the repository on GitHub!

```bash
https://github.com/Hamzah-20/ai-auto-analyzer
```
