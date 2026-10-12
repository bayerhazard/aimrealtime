"""Environment configuration."""

import os

APP_VERSION = "26.10.16"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Config:
    def __init__(self) -> None:
        self.advertised_model = env("ADVERTISED_MODEL", "aimrealtime")
        self.api_key = env("API_KEY")
        # Empty upstream URL means "use the Router of this Olares".
        self.stt_url = env("UPSTREAM_STT_URL", "").rstrip("/")
        self.tts_url = env("UPSTREAM_TTS_URL", "").rstrip("/")
        self.llm_url = env("UPSTREAM_LLM_URL", "").rstrip("/")
        # Router default categories resolve whatever model this machine has.
        self.stt_model = env("STT_MODEL", "default-stt")
        self.tts_model = env("TTS_MODEL", "default-tts")
        self.llm_model = env("LLM_MODEL", "default-chat")
        self.default_voice = env("DEFAULT_VOICE", "female")
        self.vad_threshold = float(env("VAD_THRESHOLD", "0.5"))
        self.max_calls = int(env("MAX_CALLS", "2"))
        self.log_level = env("LOG_LEVEL", "info").upper()

    def masked(self) -> dict:
        return {
            "advertised_model": self.advertised_model,
            "stt": self.stt_url or "(router)",
            "tts": self.tts_url or "(router)",
            "llm": self.llm_url or "(router)",
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "llm_model": self.llm_model,
            "default_voice": self.default_voice,
            "api_key_set": bool(self.api_key),
            "max_calls": self.max_calls,
        }
