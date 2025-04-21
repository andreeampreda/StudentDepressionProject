import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('produse.csv')

print(data.head())


print("Prima înregistrare:")
print(data.loc[0])

print("Înregistrarea a treia:")
print(data.iloc[2])


print("Înainte de modificări:")
print(data[['Denumire produs', 'Cantitate', 'Pret fara TVA']])

data.loc[0, 'Cantitate'] = 1000
data['Pret fara TVA'] = data['Pret fara TVA'] * 1.1

print("După modificări:")
print(data[['Denumire produs', 'Cantitate', 'Pret fara TVA']])


total_cantitate = data['Cantitate'].sum()
medie_pret = data['Pret cu TVA'].mean()

print("Magazinul are o cantitate totala de", total_cantitate, "produse.")
print("Pretul mediu al produselor este de", round(medie_pret), "lei.")


data.dropna(inplace=True)
data.drop(columns=['Nr crt', 'TVA (19%)'], inplace=True)
data.drop([0, 2, 4], inplace=True)


total_suma_totala = data['Suma totala'].sum()
grupare_produs = data.groupby('Denumire produs').sum(numeric_only=True)

print("Totalul vânzărilor pe luna este de", total_suma_totala, "lei.")
print(grupare_produs)


plt.figure(figsize=(12,6))
plt.bar(data['Denumire produs'], data['Cantitate'])
plt.xticks(rotation=90)
plt.xlabel("Denumire produs")
plt.ylabel("Cantitate")
plt.title("Cantitatea pe produs")
plt.tight_layout()
plt.show()


X = data[['Pret cu TVA', 'Cantitate']]
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

plt.figure(figsize=(8,6))
plt.scatter(X['Pret cu TVA'], X['Cantitate'], c=data['Cluster'])
plt.xlabel("Pret cu TVA")
plt.ylabel("Cantitate")
plt.title("Clusterizare produse")
plt.show()


y = (data['Cantitate'] > 500).astype(int)
model = LogisticRegression()
model.fit(X, y)
print("Scor regresie logistică:", model.score(X, y))
