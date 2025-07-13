import pandas as pd
import os

class NetflixRecommender:
    def __init__(self):
        # Chemin vers le fichier de données nettoyé
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, "data", "netflix_cleaned.csv")
        self.df = pd.read_csv(data_path)

    def recommend_by_genre(self, genre, n=5):
        """
        Recommande des films en fonction du genre.
        """
        filtered = self.df[self.df['listed_in'].str.contains(genre, case=False, na=False)]
        recommended = filtered.sort_values(by='release_year', ascending=False).head(n)
        return recommended[['title', 'release_year']].to_dict(orient="records")

    def recommend_recent(self, n=5):
        """
        Recommande les films les plus récents.
        """
        recent = self.df.sort_values(by='release_year', ascending=False).head(n)
        return recent[['title', 'release_year']].to_dict(orient="records")
