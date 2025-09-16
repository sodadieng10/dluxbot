import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Dis-moi bonjour en français"}]
)

print(response.choices[0].message.content)

#uvicorn main:app --reload