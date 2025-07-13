from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from memory.memory import MemoryManager
from agents.recommender import get_response
from utils.prompts import build_prompt

app = FastAPI()

# ✅ Autoriser les requêtes depuis le frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Tu peux remplacer "*" par "http://localhost:8000" pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = MemoryManager()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")
    user_id = data.get("user_id", "default_user")

    context = memory.get_context(user_id)
    prompt = build_prompt(user_input, context)

    response = get_response(prompt)
    updated_context = context + f"\nUtilisateur : {user_input}\nDluxbot : {response}"
    memory.save_context(user_id, updated_context)

    return {"response": response}

# 📂 Serveur de fichiers statiques
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))
