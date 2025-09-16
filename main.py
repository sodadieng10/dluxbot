# main.py
import os
import re
from typing import List, Union

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# OpenAI SDK v1.x
from openai import OpenAI

# ==== Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
APP_DIR = os.path.join(BASE_DIR, "app")
DATA_DIR = os.path.join(APP_DIR, "data")

# ==== .env & OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ==== Import moteur de reco
# (assure-toi que recommender.py charge app/data/data_cleaned.csv)
from app.recommender import (
    recommend_by_customer,
    recommend_by_product,
    recommend_popular,
    recommend_by_query,   # <--- sémantique TF-IDF
)

# ==== FastAPI
app = FastAPI(title="DLUXBOT API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Servir l'interface web (templates/)
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/robot.jpg")
def robot_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "robot.jpg"))

@app.get("/user.jpg")
def user_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "user.jpg"))

# (optionnel) exposer tout templates/ sous /assets
app.mount("/assets", StaticFiles(directory=TEMPLATES_DIR), name="assets")

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
#   GPT: INTENTION + STYLE
# =========================
def gpt_understand_intent(prompt: str) -> str:
    """
    Classifie l'intention: client / produit / populaire / aide.
    Retourne une ou plusieurs étiquettes en minuscules.
    """
    if client is None:
        return "fallback::no_api_key"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un classificateur d'intentions pour un chatbot e-commerce. "
                        "Réponds UNIQUEMENT avec des mots-clés parmi: client, produit, populaire, aide."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return "fallback::error"


def gpt_generate_response(user_input: str, recos: Union[List[str], str]) -> str:
    """
    Demande à GPT-4o de transformer la liste brute 'recos' en message fluide,
    utile et en français (ton assistant/conseiller).
    """
    if client is None:
        # Fallback simple si pas d'API
        return "<br>".join(recos) if isinstance(recos, list) else str(recos)

    recos_text = "\n".join(recos) if isinstance(recos, list) else str(recos)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es DLUXBOT 🤖, un assistant e-commerce francophone expert en recommandations. "
                        "Règles:\n"
                        "- Réponds en français, ton chaleureux et pro.\n"
                        "- Ne JAMAIS inventer des produits qui ne figurent pas dans la liste fournie.\n"
                        "- Structure en puces si utile (max ~8).\n"
                        "- Explique brièvement la pertinence.\n"
                        "- Si la demande est vague et la liste vide, pose 1 question de clarification max.\n"
                        "- Termine parfois par une petite suggestion d'action."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Message utilisateur:\n{user_input}\n\n"
                        f"Recommandations (issues du moteur de recherche, ne rien inventer):\n{recos_text}\n\n"
                        "Rédige la meilleure réponse finale à afficher."
                    ),
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # En cas d'erreur GPT, renvoyer la liste brute
        fallback = "(Réponse directe)\n" + (recos_text if recos_text else "Aucun résultat.")
        return fallback + f"\n\n[Note: erreur GPT: {e}]"


# =========================
#   ENDPOINT DE CHAT
# =========================
@app.post("/ask")
async def ask_question(request: Request):
    payload = await request.json()
    user_input: str = payload.get("message", "").strip()

    if not user_input:
        return JSONResponse({"response": "📝 Merci d’entrer un message."})

    # 1) Raccourci: seulement un CustomerID (≥4 chiffres) → reco client directe
    only_digits = re.fullmatch(r"\s*\d{4,}\s*", user_input)
    if only_digits:
        customer_id = re.sub(r"\D", "", user_input)
        recos = recommend_by_customer(customer_id)
        final = gpt_generate_response(user_input, recos)
        return JSONResponse({"response": final})

    # 2) Sinon, détection d'intention via GPT
    intent = gpt_understand_intent(user_input)

    # 3) Fallback si GPT KO / pas de clé → heuristique
    if intent.startswith("fallback::"):
        if re.search(r"\d{4,}", user_input):
            intent = "client"
        elif any(k in user_input.lower() for k in ["populaire", "tendance", "top", "best"]):
            intent = "populaire"
        else:
            intent = "produit"

    # 4) Routage + fallback sémantique si besoin
    try:

        if "client" in intent:
            m = re.search(r"\d{4,}", user_input)
            if m:
                # cas classique : il y a bien un CustomerID → reco perso
                recos = recommend_by_customer(m.group())
            else:
                # pas d’ID trouvé → on ne bloque pas, on tente une recherche sémantique
                recos = recommend_by_query(user_input)

        elif "produit" in intent:
            # D'abord tentative de match par mot-clé exact/partiel dans les descriptions
            recos = recommend_by_product(user_input)
            # Si rien de pertinent, bascule sur la recherche sémantique (TF-IDF)
            if isinstance(recos, list) and recos and recos[0].lower().startswith("aucun produit"):
                recos = recommend_by_query(user_input)

        elif "populaire" in intent:
            recos = recommend_popular()

        else:
            # Intention ambiguë → directement recherche sémantique
            recos = recommend_by_query(user_input)

        final = gpt_generate_response(user_input, recos)
        return JSONResponse({"response": final})

    except Exception as e:
        return JSONResponse({"response": f"❌ Erreur serveur : {e}"}, status_code=500)
