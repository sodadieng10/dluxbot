# app/recommender.py
import os
import pandas as pd
from collections import Counter

# NEW: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

if "CustomerID" in df.columns:
    df["CustomerID"] = (
        df["CustomerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
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

# ======================================
#      INDEX TF-IDF POUR LA RECHERCHE
# ======================================
# On déduplique les descriptions pour accélérer
desc_series = df["Description"].dropna().astype(str)
unique_desc = pd.Series(desc_series.unique())

# Vectoriseur : unigrams + bigrams, stopwords anglais (dataset en anglais)
tfidf = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(unique_desc.values)

# Fréquence globale des produits (pour départager en cas d’égalité)
global_counts = df["Description"].value_counts()

def _top_unique(items, top_n=5):
    """Garde l'ordre et déduplique."""
    out, seen = [], set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= top_n:
            break
    return out

# ========== 1) Recommandation par historique client ==========
def recommend_by_customer(customer_id: str, top_n: int = 5):
    customer_id = str(customer_id).strip().replace(".0", "")
    client_history = df[df["CustomerID"] == customer_id]

    if client_history.empty:
        return [f"Aucune donnée trouvée pour le client {customer_id}."]

    top_products = (
        client_history["Description"].value_counts().head(top_n).index.tolist()
    )
    return [f"Basé sur votre historique, vous aimerez peut-être :"] + top_products

# ========== 2) Recommandation par produit (co-occurences exactes/partielles) ==========
def recommend_by_product(product_query: str, top_n: int = 5):
    q = str(product_query).lower().strip()
    if not q:
        return ["Merci d'indiquer un nom de produit ou un mot-clé."]

    matches = df[df["Description"].str.contains(q, na=False)]
    if matches.empty:
        return [f"Aucun produit correspondant à « {q} »."]

    invoices = matches["InvoiceNo"].unique()
    co_occur = df[df["InvoiceNo"].isin(invoices)]
    co_products = co_occur[~co_occur["Description"].str.contains(q, na=False)]["Description"]
    top_co = Counter(co_products).most_common(top_n)

    if not top_co:
        return [f"Aucun produit fréquemment acheté avec « {q} »."]

    recos = [p for p, _ in top_co]
    return [f"Les clients ayant acheté « {q} » ont aussi acheté :"] + recos

# ========== 2bis) Recommandation sémantique (requête libre) ==========
def recommend_by_query(query: str, top_n: int = 5):
    """
    Recherche sémantique sur les descriptions via TF-IDF + similarité cosinus.
    Renvoie les descriptions les plus proches, pondérées par leur popularité.
    """
    q = str(query).lower().strip()
    if not q:
        return ["Merci d'indiquer votre besoin."]

    q_vec = tfidf.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()

    # on récupère les indices triés par similarité
    top_idx = sims.argsort()[::-1][: max(top_n * 5, 20)]
    candidates = unique_desc.iloc[top_idx].tolist()

    # on trie d'abord par similarité, puis par popularité globale
    candidates_sorted = sorted(
        candidates,
        key=lambda d: (sims[top_idx[candidates.index(d)]], global_counts.get(d, 0)),
        reverse=True,
    )

    final = _top_unique(candidates_sorted, top_n=top_n)
    if not final:
        return ["Je n'ai pas trouvé de produit adapté à cette demande."]
    return [f"Voici des idées qui correspondent à votre demande :"] + final

# ========== 3) Produits populaires globaux ==========
def recommend_popular(top_n: int = 5):
    if df["Description"].empty:
        return ["Pas de données disponibles."]
    popular = df["Description"].value_counts().head(top_n).index.tolist()
    return ["Voici les produits les plus populaires en ce moment :"] + popular

# ========== 4) Vérifier si une catégorie existe ==========
def check_category_existence(category_keywords: list):
    """
    Vérifie si une catégorie de produits (ex: beauté, maison)
    est représentée dans les descriptions du dataset.
    """
    found_products = df[df["Description"].str.contains("|".join(category_keywords), case=False, na=False)]
    if found_products.empty:
        return f"❌ Non, nous n'avons pas de produits liés à {', '.join(category_keywords)} dans notre base."
    else:
        sample = found_products["Description"].value_counts().head(5).index.tolist()
        return f"✅ Oui, nous avons des produits liés à {', '.join(category_keywords)}. Exemple : {', '.join(sample)}"
