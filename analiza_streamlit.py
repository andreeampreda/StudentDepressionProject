import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import analiza_python as fun
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

st.title("Student Depression Data Analysis")
st.write("Here are the steps to a detailed analysis of one of the most common and concerning phenomena in student life: mental health challenges.\
         This dataset brings together real student data — including academic pressure, lifestyle habits, sleep patterns, and more — to help uncover \
         patterns that may be linked to stress, depression, and overall well-being. Let's explore what the numbers reveal behind the student experience.\
         Select the preferred method to handle missing values.")


df = pd.read_csv('Student Depression Dataset.csv')
df.set_index('id', inplace=True)

# Missing Values Selection
st.sidebar.subheader("Missing Values Selection")
missing_value_method = st.sidebar.selectbox(
    "Choose a method to handle missing values:",
    ("fillna - zero or unknown", "fillna - mean or mode", "bfill", "ffill", "interpolate", "mice")
)

# Outliers Selection
st.sidebar.subheader("Outliers Selection")
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.to_list()
excluded_columns = ['Gender', 'Depression']

if len(df_numeric) > 0:
    numeric_cols = [col for col in df_numeric.columns if col not in excluded_columns]
    column_box1 = st.sidebar.selectbox("Choose column for box plot:", numeric_cols)

outlier_method = st.sidebar.selectbox(
        "Choose a method to handle outliers:",
        ("-","zscore_th2", "zscore_th2.5", "zscore_th3", "quantile_1%"))

# Sidebar for choosing a scaler
st.sidebar.subheader("Scaler Selection")
scaler_option = st.sidebar.selectbox(
        "Choose a scaler for feature scaling:",
        ("StandardScaler", "MinMaxScaler", "RobustScaler")
    )    

st.subheader("Missing Values in Percentage")

# Calculate the percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]

if not missing_percentage.empty:
    missing_df = pd.DataFrame(missing_percentage, columns=["Percentage of Missing Values"])
    st.table(missing_df)  
else:
    st.write("No missing values in the dataset.")

st.subheader("Missing Values Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap = 'summer', yticklabels=False)
st.pyplot(plt)

# Data transformation
df = fun.transform_data(df)

# Apply the selected missing value handling method
df = fun.missing_values(df, missing_value_method=missing_value_method)
df_numeric = df.select_dtypes(include=[np.number]) 
numeric_cols = df_numeric.columns.to_list()

st.subheader("Outliers")
df_original = df.copy()
if column_box1 and outlier_method:
    df = fun.outliering_data(df, column=column_box1, outlier_method=outlier_method)

    st.write(f"🔸 Original number of rows: {df_original.shape[0]}")
    st.write(f"🔹 Number of rows after outliering: {df.shape[0]}")

    fig, ax = plt.subplots()
    ax.boxplot(df[column_box1].dropna(), vert=False)
    ax.set_xlabel(column_box1)
    ax.set_title(f"Box Plot după tratarea outlierilor")
    st.pyplot(fig)


scaler = None
if scaler_option == "StandardScaler":
    scaler = StandardScaler()
elif scaler_option == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_option == "RobustScaler":
    scaler = RobustScaler()

# if scaler:
#     df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# # Correlation HeatMap 
# st.subheader("Heatmap de corelație")
# plt.figure(figsize=(10, 8))
# numeric_cols = [col for col in df_numeric.columns if col not in excluded_columns]
# sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
# st.pyplot(plt)





































