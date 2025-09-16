# app/recommender.py
import os
import pandas as pd
from collections import Counter

# === Localisation du CSV nettoyé ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../app
DATA_PATH = os.path.join(BASE_DIR, "data", "data_cleaned.csv")  # .../app/data/data_cleaned.csv

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"⚠️ Fichier introuvable : {DATA_PATH}\n"
        f"Exécute d'abord app/preprocessing.py pour générer data_cleaned.csv."
    )

# === Chargement & normalisation ===
df = pd.read_csv(DATA_PATH)

# Normaliser les colonnes clés
# - CustomerID peut être 17850.0 -> on garde '17850'
# - Description en minuscules pour recherche insensible à la casse
# - InvoiceNo en chaîne (utile pour les co-occurrences)
if "CustomerID" in df.columns:
    df["CustomerID"] = (
        df["CustomerID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)  # retire le .0 final
    )
else:
    df["CustomerID"] = ""

if "Description" in df.columns:
    df["Description"] = df["Description"].astype(str).str.lower().str.strip()
else:
    df["Description"] = ""

if "InvoiceNo" in df.columns:
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
else:
    df["InvoiceNo"] = ""


# ========== 1) Recommandation par historique client ==========
def recommend_by_customer(customer_id: str, top_n: int = 5):
    """
    Retourne les produits les plus fréquents dans l'historique d'un client.
    """
    customer_id = str(customer_id).strip()
    # Normalise l'éventuel ".0"
    customer_id = customer_id.replace(".0", "")

    client_history = df[df["CustomerID"] == customer_id]

    if client_history.empty:
        return [f"Aucune donnée trouvée pour le client {customer_id}."]

    top_products = (
        client_history["Description"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    return [f"Basé sur votre historique, vous aimerez peut-être :"] + top_products


# ========== 2) Recommandation par produit (co-occurences) ==========
def recommend_by_product(product_query: str, top_n: int = 5):
    """
    Recherche partielle insensible à la casse puis co-occurrences dans les mêmes factures.
    """
    q = str(product_query).lower().strip()
    if not q:
        return ["Merci d'indiquer un nom de produit ou un mot-clé."]

    # Tous les produits contenant le mot-clé
    matches = df[df["Description"].str.contains(q, na=False)]
    if matches.empty:
        return [f"Aucun produit correspondant à « {q} »."]

    # Toutes les factures où apparaît au moins un produit matchant
    invoices = matches["InvoiceNo"].unique()
    co_occur = df[df["InvoiceNo"].isin(invoices)]

    # Produits co-achetés (différents de ceux contenant la requête)
    co_products = co_occur[~co_occur["Description"].str.contains(q, na=False)]["Description"]

    top_co = Counter(co_products).most_common(top_n)
    if not top_co:
        return [f"Aucun produit fréquemment acheté avec « {q} »."]

    recos = [p for p, _ in top_co]
    return [f"Les clients ayant acheté « {q} » ont aussi acheté :"] + recos


# ========== 3) Produits populaires globaux ==========
def recommend_popular(top_n: int = 5):
    """
    Retourne les produits les plus fréquents globalement.
    """
    if df["Description"].empty:
        return ["Pas de données disponibles."]

    popular = df["Description"].value_counts().head(top_n).index.tolist()
    return ["Voici les produits les plus populaires en ce moment :"] + popular
