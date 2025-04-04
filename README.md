# Car MPG Prediction Project

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-1.30.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning pipeline to predict vehicle fuel efficiency (MPG) using the classic Auto MPG dataset, following CRISP-DM methodology.

## Features

- **End-to-End ML Pipeline**:
  - Data understanding and visualization
  - Feature engineering and selection
  - Multiple regression models
  - Hyperparameter tuning with MLflow tracking

- **Deployment Options**:
  - FastAPI REST endpoint
  - Streamlit web interface
  - Docker containerization

## Quick Start

### Prerequisites
- Python 3.10+
- pip 23.0+

### Installation
```bash
git clone https://github.com/yourusername/car_MPG_prediction.git
cd car_MPG_prediction
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python main.py
```

### Using the Web Interface
```bash
streamlit run main.py 
```

## Key Components

### 1. Data Processing:
- Handles missing values and outliers
- Visualizes feature distributions
```bash
from src.data_processing.data_understanding import explore_data
explore_data('data/raw_data.csv')
```

### 2. Model Training:
20+ regression models including:
- Linear models (Ridge, Lasso)
- Tree-based models (XGBoost, LightGBM)
- Neural networks
```bash
from src.modeling.regressors import get_regressors
models = get_regressors()
```

### Model Tracking:
MLflow integration for experiment tracking
```bash
mlflow ui  # View at http://localhost:5000
```

## Deployment
### Docker Build
```bash
docker build -t mpg-predictor .
docker run -p 8000:8000 mpg-predictor
```
### API Documentation
Access Swagger docs at http://localhost:8000/docs

## Contributing
1. Fork the repository
2. Create your feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request

## License
Distributed under the MIT License. See LICENSE for more information.




