from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, HuberRegressor, LassoLars, PassiveAggressiveRegressor, \
RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
def get_regressors():

    regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "XGBoost Regressor": XGBRegressor(),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Neural Network Regressor": MLPRegressor(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Huber Regressor": HuberRegressor(),
    "Lasso Lars": LassoLars(),
    "Passive Aggressive Regressor": PassiveAggressiveRegressor(),
    "RANSAC Regressor": RANSACRegressor(),
    "SGD Regressor": SGDRegressor(),
    "TheilSen Regressor": TheilSenRegressor(),
    "Gaussian Process Regressor": GaussianProcessRegressor(),
    "CatBoost Regressor": CatBoostRegressor(),
    "LightGBM Regressor": LGBMRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()
    }
    return regressors