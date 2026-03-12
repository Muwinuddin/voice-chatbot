import os

import requests

from utils import load_env, normalize_env_value

SERPER_SEARCH_URL = "https://google.serper.dev/search"


def search_web(query: str, num_results: int = 5) -> list[dict]:
    load_env()
    api_key = normalize_env_value(os.getenv("SERPER_API_KEY", ""))
    if not api_key:
        raise ValueError("SERPER_API_KEY is not set")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    response = requests.post(SERPER_SEARCH_URL, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic", []):
        results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    return results
