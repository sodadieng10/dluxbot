from fastapi import FastAPI, Request
from memory.memory import MemoryManager
from agents.recommender import get_response
from utils.prompts import build_prompt

app = FastAPI()
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
