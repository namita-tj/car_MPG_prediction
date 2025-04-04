import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceQuestionMark(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[X == '?'] = np.nan
        return X


class ExtractYear(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: int(str(x)[-2:]))


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df.dropna(inplace=True)
    return df


def visualize_correlation_matrix(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def preprocess_data(df, features, target):
    numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['origin', 'name']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', CountVectorizer())
    ])

    text_features = ['name']
    text_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', CountVectorizer())
    ])

    datetime_features = ['model_year']
    datetime_transformer = Pipeline(steps=[
        ('extract_year', ExtractYear()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_features),
            ('datetime', datetime_transformer, datetime_features)
        ])

    pipeline = Pipeline(steps=[
        ('question_mark_handler', ReplaceQuestionMark()),
        ('preprocessor', preprocessor)
    ])

    transformed_data = pipeline.fit_transform(df[features], df[target])
    return transformed_data