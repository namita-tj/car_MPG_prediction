import pandas as pd
def load_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data
def explore_data(data):
    print("Data Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
if __name__ == "__main__":
    file_path = "/data/raw_data.csv"
    data = load_data(file_path)
    explore_data(data)

