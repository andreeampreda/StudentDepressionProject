from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from impyute.imputation.cs import mice
from scipy.stats import zscore
import pandas as pd
  
st.title("Student Depression Data Analysis")
st.write("Here are the steps to a detailed analysis of one of the most common and concerning phenomena in student life: mental health challenges.\
         This dataset brings together real student data — including academic pressure, lifestyle habits, sleep patterns, and more — to help uncover \
         patterns that may be linked to stress, depression, and overall well-being. Let's explore what the numbers reveal behind the student experience.\
         Select the preferred method to handle missing values.")


df = pd.read_csv('Student Depression Dataset.csv')
df.set_index('id', inplace=True)
df.drop(columns=['Work Pressure'])

# Data set preprocess
df['Gender'] = df['Gender'].map({'Male': 0, "Female": 1})
df['Suicidal_Thoughts'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
df.drop(columns=['Have you ever had suicidal thoughts ?'], inplace=True)

# Drop the unnecessary columns
df.drop(columns=['Profession', 'Work Pressure', 'Job Satisfaction'], inplace=True)

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


# Apply the selected missing value handling method
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


df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.to_list()

if len(df_numeric) > 0:
    column_box = st.selectbox("Choose column for box plot:", numeric_cols)
    if column_box:
        fig, ax = plt.subplots()
        ax.boxplot(df[column_box].dropna(), vert=False)  # Drop NaN values for the box plot
        ax.set_xlabel(column_box)
        ax.set_title(f"Box Plot of {column_box}")
        st.pyplot(fig)
else:
    st.write("No numeric columns available for creating a box plot.")

st.subheader("Outliers")
if column_box1:
    if outlier_method == "-":
        df
    elif outlier_method == "zscore_th2":
        threshold = 2
        df = df[(np.abs(zscore(df[column_box1])) < threshold)]
    elif outlier_method == "zscore_th2.5":
        threshold = 2.5
        df = df[(np.abs(zscore(df[column_box1])) < threshold)]
    elif outlier_method == "zscore_th3":
        threshold = 3
        df = df[(np.abs(zscore(df[column_box1])) < threshold)]
    elif outlier_method == "quantile_1%":
            q_low = df[column_box1].quantile(0.01)
            q_hi  = df[column_box1].quantile(0.99)
            df = df[(df[column_box1] < q_hi) & (df[column_box1] > q_low)]

non_numeric_cols = df.select_dtypes(include=['object']).columns.to_list()
if non_numeric_cols:
    labelEncoder = LabelEncoder()
    for col in non_numeric_cols:
        df[col] = labelEncoder.fit_transform(df[col].astype(str))
    st.write(f"Transformed columns: {', '.join(non_numeric_cols)}")

# Correlation matrix
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['Age', 'Gender', 'City']).corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# Logistic Regression
st.subheader("Logistic Regression Model")
X = df.drop(columns=["Depression"])
y = df["Depression"]

# Apply the selected scaler
if scaler_option == "StandardScaler":
    scaler = StandardScaler()
elif scaler_option == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_option == "RobustScaler":
    scaler = RobustScaler()

feature_names = X.columns
X = scaler.fit_transform(X)

# Display a message about the selected scaler
st.write(f"Applied {scaler_option} to scale the features.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

st.write("### Classification Report")
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df.style
             .background_gradient(cmap='Greens')
             .format("{:.2f}")
             .set_properties(**{'text-align': 'center'})
             .set_table_styles(
                 [{'selector': 'th', 'props': [('text-align', 'center')]}]
             ))

roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
st.pyplot(plt)

st.write("### Feature Importance (Coefficients)")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", key=lambda x: abs(x), ascending=False)

st.dataframe(coef_df)
































