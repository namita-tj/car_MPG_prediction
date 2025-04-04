def analyze_variables(data):
    continuous_vars = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    discrete_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return continuous_vars, discrete_vars