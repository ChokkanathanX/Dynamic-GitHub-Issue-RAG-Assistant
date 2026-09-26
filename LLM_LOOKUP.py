import os
import json

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

gemini = genai.Client(api_key=api_key)


def llm_route(question):

    prompt = f"""
You are a routing classifier for a GitHub issue assistant.

Choose exactly ONE route:

exact
semantic
latest
hybrid

Definitions:

exact:
Use when the user asks about a specific GitHub issue number.

Example:
"What is issue #6139?"

semantic:
Use when the user asks a conceptual or topic-based question
about GitHub issues.

Example:
"What issues discuss request contexts?"

latest:
Use when the user only wants the newest/recent issues
without a specific topic.

Example:
"What are the latest issues?"

hybrid:
Use when the user asks about a topic AND also asks for
recent/latest/newest information.

Example:
"What are the latest security issues?"

issue_number:
- Extract the GitHub issue number if the user mentions one.
- Otherwise use null.

state:
- Use "open" if the user asks for open issues.
- Use "closed" if the user asks for closed issues.
- Otherwise use null.

Return ONLY valid JSON.

Format:

{{
    "route": "exact",
    "issue_number": null,
    "state": null
}}

USER QUESTION:
{question}
"""

    response = gemini.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    text = response.text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid router response: {text}"
        )

    route = result.get("route")

    valid_routes = {
        "exact",
        "semantic",
        "latest",
        "hybrid"
    }

    if route not in valid_routes:
        raise ValueError(
            f"Invalid route returned by LLM: {route}"
        )

    return result
