import os
import pandas as pd
from collections import Counter

# === Définir un chemin absolu vers data_cleaned.csv ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # chemin du dossier app/
DATA_PATH = os.path.join(BASE_DIR, "data", "data_cleaned.csv")

# Vérifier si le fichier existe avant de charger
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"⚠️ Le fichier {DATA_PATH} est introuvable. "
                            f"Assurez-vous d’avoir exécuté preprocessing.py.")

# Chargement des données nettoyées
df = pd.read_csv(DATA_PATH)


# === Recommandations ===

# 1. Recommandation par historique client
def recommend_by_customer(customer_id, top_n=5):
    client_history = df[df['CustomerID'] == str(customer_id)]
    if client_history.empty:
        return [f"Aucune donnée trouvée pour le client {customer_id}."]

    top_products = client_history['Description'].value_counts().head(top_n).index.tolist()
    return [f"Basé sur votre historique, vous aimerez peut-être :"] + top_products


# 2. Recommandation par produit (produits fréquemment achetés ensemble)
def recommend_by_product(product_name, top_n=5):
    product_name = product_name.lower().strip()
    if product_name not in df['Description'].values:
        return [f"Produit '{product_name}' non trouvé."]

    invoices = df[df['Description'] == product_name]['InvoiceNo'].unique()
    co_occur = df[df['InvoiceNo'].isin(invoices)]

    co_products = co_occur[co_occur['Description'] != product_name]['Description']
    top_co_products = Counter(co_products).most_common(top_n)

    if not top_co_products:
        return [f"Aucun produit fréquemment acheté avec '{product_name}'."]

    recommendations = [prod for prod, count in top_co_products]
    return [f"Les clients ayant acheté '{product_name}' ont aussi acheté :"] + recommendations


# 3. Produits populaires
def recommend_popular(top_n=5):
    popular = df['Description'].value_counts().head(top_n).index.tolist()
    return ["Voici les produits les plus populaires en ce moment :"] + popular
