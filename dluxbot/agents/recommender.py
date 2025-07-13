from openai import OpenAI
from config import OPENAI_API_KEY
from agents.recommender_agent import NetflixRecommender

# Initialisation du client OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialisation du moteur de recommandation
recommender = NetflixRecommender()

def get_response(prompt):
    prompt_lower = prompt.lower()

    # 👉 Si l'utilisateur demande des films d'action
    if "film d'action" in prompt_lower or "action" in prompt_lower:
        films = recommender.recommend_by_genre("action")
        return "Voici des films d'action recommandés :\n" + "\n".join(
            [f"- {f['title']} ({f['release_year']})" for f in films]
        )

    # 👉 Si l'utilisateur demande des films récents
    elif "film récent" in prompt_lower or "derniers films" in prompt_lower:
        films = recommender.recommend_recent()
        return "Voici les films les plus récents :\n" + "\n".join(
            [f"- {f['title']} ({f['release_year']})" for f in films]
        )

    # 👉 Sinon, faire appel à GPT
    else:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",  # ou "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "Tu es un assistant intelligent et personnalisé."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return chat_completion.choices[0].message.content.strip()

