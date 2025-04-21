
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")

st.title("Analiza produselor Flanco")
data = pd.read_csv("produse.csv")

st.subheader("Set de date - Produse")
st.dataframe(data)

produs_selectat = st.selectbox("Selectează un produs:", data["Denumire produs"].unique())
st.write("Detalii despre produsul selectat:")
st.write(data[data["Denumire produs"] == produs_selectat])
