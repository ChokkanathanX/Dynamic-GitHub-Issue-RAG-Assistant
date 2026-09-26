import os

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. Load API key
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# ==========================================
# 2. Create Gemini client
# ==========================================

gemini = genai.Client(
    api_key=api_key
)


# ==========================================
# 3. Generate answer
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are a GitHub issue assistant.

Answer the user's question using ONLY the
information provided in the context.

Rules:
- Do not invent information.
- If the context does not contain enough information,
  say that the available information is insufficient.
- Keep the answer concise.
- When referring to an issue, include its issue number.
- Include the relevant GitHub URL when available.

USER QUESTION:
{question}

CONTEXT:
{context}
"""

    response = gemini.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text