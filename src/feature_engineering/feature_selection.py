from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
def perform_feature_selection(X_train, y_train):
    k_best_selector = SelectKBest(score_func=f_regression, k=5)
    X_train_k_best = k_best_selector.fit_transform(X_train, y_train)

    variance_selector = VarianceThreshold(threshold=0.1)
    X_train_variance = variance_selector.fit_transform(X_train)

    mutual_info_selector = mutual_info_regression(X_train, y_train)
    selected_features_mi = [index for index, feature in enumerate(mutual_info_selector) if feature > 0.05]
    X_train_mutual_info = X_train.iloc[:, selected_features_mi]

    estimator = Lasso(alpha=0.1)
    rfe_selector = RFE(estimator, n_features_to_select=5, step=1)
    X_train_rfe = rfe_selector.fit_transform(X_train, y_train)

    rf_selector = RandomForestRegressor(n_estimators=100)
    rf_selector.fit(X_train, y_train)
    selected_features_rf = rf_selector.feature_importances_.argsort()[-5:][::-1]
    X_train_rf = X_train.iloc[:, selected_features_rf]
    sfs_selector = SequentialFeatureSelector(estimator, n_features_to_select=5, scoring='neg_mean_squared_error')
    sfs_selector.fit(X_train, y_train)
    selected_features_sfs = list(sfs_selector.get_support(indices=True))
    X_train_sfs = X_train.iloc[:, selected_features_sfs]
    return [X_train_k_best, X_train_variance, X_train_mutual_info, X_train_rfe, X_train_rf,X_train_sfs]