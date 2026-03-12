import os

import requests

from utils import load_env, normalize_env_value

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"


def get_groq_response(prompt: str) -> str:
    load_env()
    api_key = normalize_env_value(os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful voice assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code == 401:
            raise ValueError("Unauthorized: check GROQ_API_KEY in .env") from exc
        raise
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
