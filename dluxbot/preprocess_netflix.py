import pandas as pd
import os

# Chemin relatif depuis ce fichier
csv_input = os.path.join("data", "netflix_titles.csv")
csv_output = os.path.join("data", "netflix_cleaned.csv")

# Charger le fichier brut
df = pd.read_csv(csv_input)

# Supprimer les lignes avec des champs critiques manquants
df.dropna(subset=["title", "listed_in", "release_year"], inplace=True)

# Nettoyage de texte
df['title'] = df['title'].str.strip()
df['listed_in'] = df['listed_in'].str.lower().str.strip()
df['description'] = df['description'].fillna("").str.strip()

# Supprimer les doublons
df.drop_duplicates(subset=["title", "listed_in", "release_year"], inplace=True)

# Sauvegarder dans un nouveau fichier propre
df.to_csv(csv_output, index=False)

print("✅ Données Netflix nettoyées et sauvegardées dans data/netflix_cleaned.csv")
