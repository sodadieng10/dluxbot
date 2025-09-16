# main.py
import os
import re
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # pour éviter un crash si la lib n'est pas installée

# ==== Chemins ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
APP_DIR = os.path.join(BASE_DIR, "app")
DATA_DIR = os.path.join(APP_DIR, "data")

# ==== .env & OpenAI ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OpenAI and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Import du moteur de reco ====
# Assure-toi que app/recommender.py existe et charge app/data/data_cleaned.csv
from app.recommender_v2 import (
    recommend_by_customer,
    recommend_by_product,
    recommend_popular,
)

# ==== FastAPI ====
app = FastAPI(title="DLUXBOT API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en prod, restreins aux origines nécessaires
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Servir l'interface web (index.html + images) ----
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/robot.jpg")
def robot_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "robot.jpg"))

@app.get("/user.jpg")
def user_img():
    return FileResponse(os.path.join(TEMPLATES_DIR, "user.jpg"))

# (Optionnel) monter tout templates/ sous /assets
app.mount("/assets", StaticFiles(directory=TEMPLATES_DIR), name="assets")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Compréhension d'intention via GPT (avec fallback) ----
def gpt_understand_intent(prompt: str) -> str:
    """
    Utilise GPT-4o-mini pour classifier l'intention.
    Renvoie uniquement : 'client', 'produit', 'populaire', 'aide' (ou combinaison),
    sinon 'fallback::...'
    """
    if client is None:
        return "fallback::no_api_key"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un classificateur d'intentions pour un chatbot de recommandation de produits. "
                        "Réponds UNIQUEMENT par un ou plusieurs des mots suivants (séparés par des espaces si besoin) : "
                        "client produit populaire aide. Rien d'autre."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return "fallback::error"

# ---- Endpoint de chat utilisé par l'UI ----
@app.post("/ask")
async def ask_question(request: Request):
    payload = await request.json()
    user_input: str = payload.get("message", "").strip()

    if not user_input:
        return JSONResponse({"response": "📝 Merci d’entrer un message."})

    # 1) Raccourci : si l'utilisateur tape uniquement un ID (≥4 chiffres), on route direct vers la reco client
    only_digits = re.fullmatch(r"\s*\d{4,}\s*", user_input)
    if only_digits:
        customer_id = re.sub(r"\D", "", user_input)
        recos = recommend_by_customer(customer_id)
        html = "<br>".join(recos) if isinstance(recos, list) else str(recos)
        return JSONResponse({"response": html})

    # 2) Sinon, on demande à GPT l'intention
    intent = gpt_understand_intent(user_input)

    # 3) Fallback si GPT KO / pas de clé
    if intent.startswith("fallback::"):
        # Heuristique simple
        if re.search(r"\d{4,}", user_input):
            intent = "client"
        elif any(k in user_input.lower() for k in ["populaire", "tendance", "top", "best"]):
            intent = "populaire"
        else:
            intent = "produit"

    # 4) Routage selon l'intention
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
            # On passe la requête complète au moteur produit qui fait une recherche partielle
            recos = recommend_by_product(user_input)

        elif "populaire" in intent:
            recos = recommend_popular()

        else:
            recos = [
                "Je peux recommander par **client** (CustomerID), par **produit** (mots-clés) "
                "ou lister les articles **populaires**. Que souhaites-tu ?"
            ]

        html = "<br>".join(recos) if isinstance(recos, list) else str(recos)
        return JSONResponse({"response": html})

    except Exception as e:
        return JSONResponse(
            {"response": f"❌ Erreur serveur : {e}"},
            status_code=500,
        )
