import os
import pandas as pd
def save_data(data, file_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file_name + ".csv")
    data.to_csv(file_path, index=False)

def save_results_to_excel(evaluation_results, file_path):
    df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    df.to_excel(file_path)