import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from impyute.imputation.cs import mice  # Requires installing impyute package
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Tranforming the data

# Binary Encoding
# Gender: Male -> 0, Female -> 1
# Have you ever had suicidal thoughts ?: No -> 0, Yes -> 1

def transform_data(df):

    df['Gender'] = df['Gender'].map({'Male': 0, "Female": 1})
    df['Suicidal_Thoughts'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
    df.drop(columns=['Have you ever had suicidal thoughts ?'], inplace=True)

    # Ordinal Encoding
    diet_map = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)

    sleep_map = {
        'Less than 5 hours': 0,
        '5-6 hours': 1,
        '7-8 hours': 2,
        'More than 8 hours': 3
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

    
    non_numeric_cols = df.select_dtypes(include=['object']).columns.to_list()
    if non_numeric_cols:
        labelEncoder = LabelEncoder()
    for col in non_numeric_cols:
        df[col] = labelEncoder.fit_transform(df[col].astype(str))
        
    return df

def missing_values(df, missing_value_method):
    
    if missing_value_method == "fillna - zero or unknown":
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(0, inplace=True)
            
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna("Unknown", inplace=True)

    elif missing_value_method == "fillna - mean or mode":
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
            
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    elif missing_value_method == "bfill":
        df.fillna(method='bfill', inplace=True)

    elif missing_value_method == "ffill":
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    elif missing_value_method == "interpolate":
        df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    elif missing_value_method == "mice":
        np.float = float
        df_numeric = df.select_dtypes(include=[np.number])
        imputed_df = mice(df_numeric.values)
        df[df_numeric.columns] = pd.DataFrame(imputed_df, columns=df_numeric.columns)

    return df

def outliering_data(df, column, outlier_method):
    if outlier_method == "-":
        return df
    elif outlier_method == "zscore_th2":
        threshold = 2
        df = df[(np.abs(zscore(df[column])) < threshold)]
    elif outlier_method == "zscore_th2.5":
        threshold = 2.5
        df = df[(np.abs(zscore(df[column])) < threshold)]
    elif outlier_method == "zscore_th3":
        threshold = 3
        df = df[(np.abs(zscore(df[column])) < threshold)]
    elif outlier_method == "quantile_1%":
            q_low = df[column].quantile(0.01)
            q_hi  = df[column].quantile(0.99)
            df = df[(df[column] < q_hi) & (df[column] > q_low)]
    return df






















