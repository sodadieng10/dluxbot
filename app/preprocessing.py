import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier de sortie s’il n’existe pas
os.makedirs("data", exist_ok=True)

# 1. Chargement des données
df = pd.read_csv("data/data.csv", encoding='latin1')

# 2. Nettoyage des données
df.dropna(subset=['Description', 'CustomerID'], inplace=True)
df['Description'] = df['Description'].str.strip().str.lower()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['CustomerID'] = df['CustomerID'].astype(str)

# 3. Filtrage
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df.drop_duplicates(inplace=True)

# 4. Sauvegarde des données nettoyées
df.to_csv("data/data_cleaned.csv", index=False)

# 5. Visualisations
plt.figure(figsize=(10,6))
top_products = df['Description'].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 produits les plus vendus")
plt.xlabel("Nombre de ventes")
plt.ylabel("Produit")
plt.tight_layout()
plt.savefig("data/top_products.png")

plt.figure(figsize=(10,6))
top_countries = df['Country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 pays par volume de commandes")
plt.xlabel("Nombre de commandes")
plt.ylabel("Pays")
plt.tight_layout()
plt.savefig("data/top_countries.png")

# Graphe supplémentaire : répartition des prix
plt.figure(figsize=(8,5))
sns.histplot(df['UnitPrice'], bins=50, kde=True)
plt.title("Distribution des prix unitaires")
plt.xlabel("Prix unitaire")
plt.ylabel("Fréquence")
plt.xlim(0, df['UnitPrice'].quantile(0.95))  # éviter les extrêmes
plt.tight_layout()
plt.savefig("data/unitprice_distribution.png")
