import pandas as pd
from mlflow.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import r2_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y

def hyperparameter_tuning(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return grid_search, best_params, best_model


def log_metrics_and_model(model, X_train,y_train,params=None):
    with mlflow.start_run():
        # Log parameters if provided
        if params:
            mlflow.log_params(params)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        mlflow.log_metrics({
            'mse': mse,
            'r2_score': r2
        })

        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mpg-predictor",
            signature=signature,
            input_example=X_train.iloc[:5] if hasattr(X_train, 'iloc') else X_train[:5],
            registered_model_name="MPG_Predictor"
        )


def run_unit_tests(X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    y_range = y_train.max() - y_train.min()
    threshold_value = 0.10 * y_range

    print(f"\nUnit Test Debug:")
    print(f"- Model MSE: {mse:.4f}")
    print(f"- Allowed Threshold: {threshold_value:.4f}")
    print(f"- Target Range: {y_range:.4f} (min={y_train.min():.4f}, max={y_train.max():.4f})")

    if mse > threshold_value:
        print(f"WARNING: Model performance below threshold (MSE {mse:.4f} > {threshold_value:.4f})")
    else:
        print("Unit test passed!")

def deploy_fastapi_app():
    app = FastAPI()

    @app.post("/predict")
    async def predict(data: dict):
        features = pd.DataFrame(data, index=[0])
        prediction = best_model.predict(features)
        return {"prediction": prediction}

    return app


