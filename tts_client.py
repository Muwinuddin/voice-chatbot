import html
import os

import requests

from utils import load_env, normalize_env_value

DEFAULT_AZURE_VOICE = "en-US-AriaNeural"


def _build_tts_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/cognitiveservices/v1"):
        return base
    return f"{base}/cognitiveservices/v1"


def _build_fallback_tts_url(endpoint: str) -> str | None:
    if ".api.cognitive.microsoft.com" not in endpoint:
        return None
    host = endpoint.split("//", 1)[-1].split("/", 1)[0]
    region = host.split(".")[0]
    return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


def speak_text(text: str) -> bytes:
    load_env()
    api_key = normalize_env_value(os.getenv("AZURE_TTS_KEY", ""))
    endpoint = normalize_env_value(os.getenv("AZURE_TTS_ENDPOINT", ""))
    voice_name = normalize_env_value(os.getenv("AZURE_TTS_VOICE", "alloy"))

    if not api_key:
        raise ValueError("AZURE_TTS_KEY is not set")
    if not endpoint:
        raise ValueError("AZURE_TTS_ENDPOINT is not set")

    if voice_name.lower() == "alloy":
        voice_name = DEFAULT_AZURE_VOICE

    url = _build_tts_url(endpoint)
    escaped_text = html.escape(text)

    ssml = (
        "<speak version='1.0' xml:lang='en-US'>"
        f"<voice name='{voice_name}'>{escaped_text}</voice>"
        "</speak>"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        "User-Agent": "voice-chatbot",
    }

    response = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
    if response.status_code in {401, 403, 404}:
        fallback_url = _build_fallback_tts_url(endpoint)
        if fallback_url and fallback_url != url:
            response = requests.post(
                fallback_url,
                headers=headers,
                data=ssml.encode("utf-8"),
                timeout=30,
            )

    response.raise_for_status()
    return response.content
