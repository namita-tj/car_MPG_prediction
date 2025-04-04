from src.data_processing.data_understanding import explore_data
from src.data_processing.data_preparation import save_prepared_data
from src.feature_engineering.feature_selection import perform_feature_selection
from src.feature_engineering.dimensionality_reduction import perform_dimensionality_reduction
from src.modeling.regressors import get_regressors
from src.utils.helpers import save_results_to_excel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing.data_preprocessing import visualize_correlation_matrix
from src.pipeline.model_pipeline import hyperparameter_tuning, log_metrics_and_model, run_unit_tests
import joblib
from sklearn.linear_model import LinearRegression
import streamlit as st

features = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'name']
target = 'mpg'

df = pd.read_csv('data/raw_data.csv')
explore_data(df)
df = pd.DataFrame(df)

for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].replace('?', df[column].mode()[0])

visualize_correlation_matrix(df)
df = df.drop(columns=['car name'])

X = df.iloc[:796, 1:8]
y = df.iloc[:796, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
save_prepared_data(X_train, X_test, y_train, y_test, "data/")

selected_features = perform_feature_selection(X_train, y_train)
X_train_reduced, X_test_reduced = perform_dimensionality_reduction(X_train, X_test)
print(X_train_reduced)

regressors = get_regressors()
evaluation_results = {}

for name, regressor in regressors.items():
    regressor.fit(X_train_reduced, y_train)
    print(f"Evaluating {name}:")
    y_pred = regressor.predict(X_test_reduced)
    mse_value = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse_value}")
    print(f"R-squared: {r2_value}")
    evaluation_results[name] = {"Mean Squared Error": mse_value, "R-squared": r2_value}

save_results_to_excel(evaluation_results, "excel_results/regression_results.xlsx")
grid_search, best_params, best_model = hyperparameter_tuning(X_train_reduced, y_train)
log_metrics_and_model(model=best_model,X_train=X_train_reduced,y_train=y_train,params=best_params)
run_unit_tests(X_test_reduced, y_train, y_test, best_model)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "trained_model.pkl")

def predict_mpg(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    input_data = [[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]]
    prediction = model.predict(input_data)
    return prediction[0]

st.title("Car MPG Prediction App")
st.write("This app predicts the MPG (miles per gallon) of a car based on its features.")

cylinders = st.slider("Cylinders", min_value=3, max_value=8, value=4)
displacement = st.slider("Displacement", min_value=50, max_value=500, value=250)
horsepower = st.slider("Horsepower", min_value=50, max_value=400, value=150)
weight = st.slider("Weight", min_value=1000, max_value=5000, value=3000)
acceleration = st.slider("Acceleration", min_value=5.0, max_value=25.0, value=15.0)
model_year = st.slider("Model Year", min_value=70, max_value=82, value=76)
origin = st.selectbox("Origin", ["USA", "Europe", "Japan"])

origin_map = {"USA": 1, "Europe": 2, "Japan": 3}
origin_numeric = origin_map[origin]

if st.button("Predict"):
    prediction = predict_mpg(cylinders, displacement, horsepower, weight, acceleration, model_year, origin_numeric)
    st.write(f"The predicted MPG is: {prediction:.2f}")

