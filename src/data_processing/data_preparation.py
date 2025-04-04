from sklearn.model_selection import train_test_split
from src.utils.helpers import save_data
def prepare_data(raw_data, target_column, test_size=0.2, random_state=42):

    X = raw_data.iloc[:796,[col for col in range(len(raw_data.columns)) if col!=target_column]]
    y = raw_data.iloc[:796,target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
def save_prepared_data(X_train, X_test, y_train, y_test, save_path):

    save_data(X_train, "X_train", save_path)
    save_data(X_test, "X_test", save_path)
    save_data(y_train, "y_train", save_path)
    save_data(y_test, "y_test", save_path)