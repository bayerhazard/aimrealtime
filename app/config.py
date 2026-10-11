"""Environment configuration."""

import os

APP_VERSION = "26.10.6"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Config:
    def __init__(self) -> None:
        self.advertised_model = env("ADVERTISED_MODEL", "aimrealtime")
        self.api_key = env("API_KEY")
        self.stt_url = env("UPSTREAM_STT_URL", "").rstrip("/")
        self.tts_url = env("UPSTREAM_TTS_URL", "").rstrip("/")
        self.llm_url = env("UPSTREAM_LLM_URL", "").rstrip("/")
        self.stt_model = env("STT_MODEL", "Qwen/Qwen3-ASR-1.7B")
        self.tts_model = env("TTS_MODEL", "omnivoice")
        self.llm_model = env("LLM_MODEL", "qwen3.8-27b")
        self.default_voice = env("DEFAULT_VOICE", "female")
        self.vad_threshold = float(env("VAD_THRESHOLD", "0.5"))
        self.max_calls = int(env("MAX_CALLS", "2"))
        self.log_level = env("LOG_LEVEL", "info").upper()

    def masked(self) -> dict:
        return {
            "advertised_model": self.advertised_model,
            "stt": self.stt_url or "(unset)",
            "tts": self.tts_url or "(unset)",
            "llm": self.llm_url or "(unset)",
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "llm_model": self.llm_model,
            "default_voice": self.default_voice,
            "api_key_set": bool(self.api_key),
            "max_calls": self.max_calls,
        }
