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

# === Chemins ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
APP_DIR = os.path.join(BASE_DIR, "app")
DATA_DIR = os.path.join(APP_DIR, "data")

# === .env & OpenAI ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# === Import du moteur de reco (qui lit app/data/data_cleaned.csv) ===
from app.recommender_v2 import (
    recommend_by_customer,
    recommend_by_product,
    recommend_popular,
)

# === FastAPI ===
app = FastAPI(title="DLUXBOT API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Servir l'interface web ----
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/robot.jpg")
def robot_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "robot.jpg"))

@app.get("/user.jpg")
def user_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "user.jpg"))

# (optionnel) tout le dossier templates sous /assets
app.mount("/assets", StaticFiles(directory=TEMPLATES_DIR), name="assets")

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
#   GPT: INTENT + STYLE
# =========================

def gpt_understand_intent(prompt: str) -> str:
    """
    Classifie rapidement l'intention: client / produit / populaire / aide.
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
    Demande à GPT-4o de transformer la liste brute 'recos' en réponse
    naturelle, utile et en français (ton assistant/conseiller).
    """
    if client is None:
        # Si pas de clé API → formatage fallback simple
        if isinstance(recos, list):
            return "<br>".join(recos)
        return str(recos)

    # Mise au propre du contenu recommandations
    if isinstance(recos, list):
        # Concatène tout en texte brut, GPT fera la mise en forme propre
        recos_text = "\n".join(recos)
    else:
        recos_text = str(recos)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,  # réponse posée et cohérente
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es DLUXBOT 🤖, un assistant e-commerce francophone expert en recommandations. "
                        "Objectif: rédiger des réponses courtes, claires, naturelles et aimables. "
                        "Règles de style:\n"
                        "- Utilise un ton chaleureux et professionnel.\n"
                        "- Résume et structure sous forme de puces si utile (max ~8 éléments).\n"
                        "- Explique brièvement pourquoi ces articles sont pertinents.\n"
                        "- Si la requête le suggère, reformule les noms produits en français SANS inventer des infos.\n"
                        "- Si la liste est vide ou ambiguë, pose 1 question de clarification maximum.\n"
                        "- Termine parfois par une petite suggestion d'action (ex: 'Souhaitez-vous voir d'autres idées ?')."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Message utilisateur:\n{user_input}\n\n"
                        f"Recommandations brutes (issues du moteur):\n{recos_text}\n\n"
                        "Rédige la meilleure réponse finale à afficher."
                    ),
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # En cas d'erreur GPT, on renvoie la liste brute
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

    # 1) Raccourci: message = uniquement un CustomerID (≥4 chiffres) → reco client directe
    only_digits = re.fullmatch(r"\s*\d{4,}\s*", user_input)
    if only_digits:
        customer_id = re.sub(r"\D", "", user_input)
        recos = recommend_by_customer(customer_id)
        final = gpt_generate_response(user_input, recos)
        return JSONResponse({"response": final})

    # 2) Sinon, détection d'intention via GPT
    intent = gpt_understand_intent(user_input)

    # 3) Fallback si GPT KO / pas de clé
    if intent.startswith("fallback::"):
        if re.search(r"\d{4,}", user_input):
            intent = "client"
        elif any(k in user_input.lower() for k in ["populaire", "tendance", "top", "best"]):
            intent = "populaire"
        else:
            intent = "produit"

    # 4) Routage et génération de la réponse finale par GPT
    try:
        if "client" in intent:
            m = re.search(r"\d{4,}", user_input)
            if m:
                recos = recommend_by_customer(m.group())
            else:
                recos = [
                    "Pour une recommandation personnalisée, indique ton identifiant client (CustomerID). "
                    "Exemple : 'Reco client 17850'."
                ]

        elif "produit" in intent:
            # On passe la requête complète: la fonction fait une recherche partielle
            recos = recommend_by_product(user_input)

        elif "populaire" in intent:
            recos = recommend_popular()

        else:
            recos = [
                "Je peux recommander par **client** (CustomerID), par **produit** (mots-clés) "
                "ou lister les articles **populaires**. Que souhaites-tu ?"
            ]

        final = gpt_generate_response(user_input, recos)
        return JSONResponse({"response": final})

    except Exception as e:
        return JSONResponse(
            {"response": f"❌ Erreur serveur : {e}"},
            status_code=500,
        )
