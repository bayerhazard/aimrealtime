"""One Open WebUI voice call: the provider side of the Realtime protocol.

The event surface mirrors what Open WebUI's bridge (routers/audio/realtime.py,
v0.12.0) sends and expects. Names and shapes matter more than completeness:
an unknown event type is ignored, a missing required one breaks the call.
"""

import asyncio
import base64
import json
import logging
import re
import time
import uuid

from audio import RATE
from config import Config
from upstream import UpstreamError, Upstreams
from vad import EnergyVad

log = logging.getLogger("aimrealtime.call")

CHUNK = 9600  # 200 ms of PCM16 @ 24 kHz
MAX_HISTORY = 24

VOICE_ALIASES = {
    "alloy": "female", "echo": "male", "fable": "female", "onyx": "male",
    "nova": "female", "shimmer": "female", "marin": "female", "cedar": "male",
}
PRESETS = {"auto", "female", "male", "child", "whisper"}

BASE_PROMPT = (
    "You are a warm, concise voice assistant inside a phone-style call. "
    "Keep spoken replies to one or two short sentences unless asked for more. "
    "Never read formatting characters aloud. "
    "For anything needing reasoning, fresh information, tools or the user's documents, "
    "call generate_chat_completion instead of answering from memory. "
    "Small talk, acknowledgements and clarifications you handle yourself."
)


def speech_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#>|]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        while len(part) > 240:
            out.append(part[:240])
            part = part[240:].strip()
        if part:
            out.append(part)
    return out or ([text.strip()] if text.strip() else [])


class Call:
    def __init__(self, ws, model: str, cfg: Config, upstreams: Upstreams) -> None:
        self.ws = ws
        self.cfg = cfg
        self.upstreams = upstreams
        self.model = model or cfg.advertised_model
        self.session_id = "sess_" + uuid.uuid4().hex[:16]
        self.voice = cfg.default_voice
        self.instructions = ""
        self.transcription_model = cfg.stt_model
        self.tools: list[dict] = []
        self.vad = EnergyVad()
        self.history: list[dict] = []
        self.context: dict[str, str] = {}
        self.responses: dict[str, dict] = {}
        self.function_results: dict[str, dict] = {}
        self.pending: set[asyncio.Task] = set()
        self.utterance: dict | None = None
        self.emitted: dict[tuple[str, int], int] = {}
        self.items = 0

    # -- plumbing ----------------------------------------------------------

    async def send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event, separators=(",", ":"), ensure_ascii=False))

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}})

    def new_item(self, side: str) -> str:
        self.items += 1
        return f"item_{side}_{self.items}_{int(time.time() * 1000) % 1000000}"

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)
        return task

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(
            {
                "type": "session.created",
                "session": {
                    "object": "realtime.session",
                    "id": self.session_id,
                    "model": self.model,
                    "output_modalities": ["audio"],
                },
            }
        )
        log.info("call open session=%s model=%s", self.session_id, self.model)
        while True:
            raw = await self.ws.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self.error("invalid_event", "Malformed JSON event")
                continue
            kind = event.get("type", "")
            handler = getattr(self, "on_" + kind.replace(".", "_"), None)
            if handler is None:
                log.debug("ignored event type=%s", kind)
                continue
            await handler(event)

    async def close(self) -> None:
        for task in list(self.pending):
            task.cancel()
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)

    # -- inbound events ----------------------------------------------------

    async def on_session_update(self, event: dict) -> None:
        session = event.get("session") or {}
        if isinstance(session.get("instructions"), str) and session["instructions"].strip():
            self.instructions = session["instructions"]
        audio = session.get("audio") or {}
        voice = (audio.get("output") or {}).get("voice")
        if isinstance(voice, str) and voice.strip():
            self.voice = self.map_voice(voice)
        tr = (audio.get("input") or {}).get("transcription") or {}
        if isinstance(tr.get("model"), str) and tr["model"].strip():
            self.transcription_model = tr["model"].strip()
        if isinstance(session.get("tools"), list):
            self.tools = [t for t in session["tools"] if isinstance(t, dict)]
        await self.send(
            {
                "type": "session.updated",
                "session": {
                    "object": "realtime.session",
                    "id": self.session_id,
                    "model": self.model,
                    "voice": self.voice,
                    "transcription_model": self.transcription_model,
                    "tools": self.tools,
                },
            }
        )

    def map_voice(self, voice: str) -> str:
        key = voice.strip().lower()
        if key in PRESETS:
            return key
        if key in VOICE_ALIASES:
            return VOICE_ALIASES[key]
        return self.cfg.default_voice

    async def on_input_audio_buffer_append(self, event: dict) -> None:
        try:
            pcm = base64.b64decode(event.get("audio", ""), validate=True)
        except Exception:
            await self.error("invalid_audio", "audio is not valid base64")
            return
        for ev in self.vad.feed(pcm):
            if ev["type"] == "start":
                self.utterance = {"item_id": self.new_item("in"), "start_ms": ev["ms"]}
                await self.send(
                    {"type": "input_audio_buffer.speech_started", "item_id": self.utterance["item_id"], "audio_start_ms": int(ev["ms"])}
                )
                await self.interrupt()
            elif ev["type"] == "end":
                utterance = self.utterance or {"item_id": self.new_item("in")}
                self.utterance = None
                await self.send(
                    {"type": "input_audio_buffer.speech_stopped", "item_id": utterance["item_id"], "audio_end_ms": int(ev["ms"])}
                )
                self.spawn(self._transcribe(utterance["item_id"], ev["pcm"]))

    async def _transcribe(self, item_id: str, pcm: bytes) -> None:
        try:
            text = await self.upstreams.transcribe(pcm, self.transcription_model)
        except UpstreamError as exc:
            log.warning("stt failed session=%s: %s", self.session_id, exc)
            await self.error("transcription_failed", "Speech recognition failed")
            return
        if not text:
            log.info("empty transcript session=%s item=%s", self.session_id, item_id)
            return
        self.history.append({"role": "user", "content": text})
        self.trim_history()
        await self.send(
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": item_id, "transcript": text}
        )

    async def interrupt(self) -> None:
        for rid, state in self.responses.items():
            if state.get("speaking") and not state.get("cancelled"):
                state["cancelled"] = True
                log.info("barge-in cancelled response session=%s rid=%s", self.session_id, rid)

    async def on_input_audio_buffer_clear(self, event: dict) -> None:
        return

    async def on_input_audio_buffer_commit(self, event: dict) -> None:
        return

    async def on_conversation_item_create(self, event: dict) -> None:
        item = event.get("item") or {}
        kind = item.get("type")
        if kind == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            parsed: dict = {"status": "completed", "answer": ""}
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                    if isinstance(data, dict):
                        parsed = {"status": str(data.get("status", "completed")), "answer": str(data.get("answer", ""))}
                except json.JSONDecodeError:
                    parsed = {"status": "completed", "answer": output}
            if call_id:
                self.function_results[call_id] = parsed
        elif kind == "message" and item.get("role") == "system":
            content = item.get("content") or []
            text = ""
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = str(content[0].get("text", ""))
            if item.get("id") and text:
                self.context[item["id"]] = text

    async def on_conversation_item_delete(self, event: dict) -> None:
        self.context.pop(event.get("item_id"), None)

    async def on_conversation_item_truncate(self, event: dict) -> None:
        key = (event.get("item_id"), event.get("content_index", 0))
        if self.emitted.get(key) is not None:
            end = event.get("audio_end_ms")
            log.info("truncate session=%s item=%s end_ms=%s", self.session_id, key[0], end)

    async def on_response_cancel(self, event: dict) -> None:
        state = self.responses.get(event.get("response_id"))
        if state is None:
            await self.error("response_cancel_not_active", "No such active response")
            return
        state["cancelled"] = True

    async def on_response_create(self, event: dict) -> None:
        request = event.get("response") or {}
        rid = "resp_" + uuid.uuid4().hex[:16]
        state = {
            "cancelled": False,
            "speaking": False,
            "done": False,
            "metadata": request.get("metadata") or {},
        }
        self.responses[rid] = state
        await self.send(
            {
                "type": "response.created",
                "response": {"id": rid, "object": "realtime.response", "status": "in_progress", "metadata": state["metadata"]},
            }
        )
        self.spawn(self._respond(rid, request, state))

    # -- outbound response -------------------------------------------------

    async def _respond(self, rid: str, request: dict, state: dict) -> None:
        try:
            metadata = state["metadata"]
            instructions = request.get("instructions")
            if isinstance(instructions, str) and instructions.strip() and not metadata:
                await self._speak(rid, state, self._status_text(instructions))
            elif metadata.get("call_id"):
                await self._speak_function_result(rid, state, metadata["call_id"])
            elif metadata.get("input_item_id"):
                await self._respond_llm(rid, state)
            else:
                await self._speak(rid, state, "Okay.")
        except asyncio.CancelledError:
            raise
        except UpstreamError as exc:
            log.warning("response failed session=%s rid=%s: %s", self.session_id, rid, exc)
            await self.finish(rid, state, "failed")
        except Exception:
            log.exception("response crashed session=%s rid=%s", self.session_id, rid)
            await self.finish(rid, state, "failed")

    @staticmethod
    def _status_text(instructions: str) -> str:
        marker = "language: "
        if marker in instructions:
            return instructions.split(marker, 1)[1].strip()
        return instructions.strip()

    async def _respond_llm(self, rid: str, state: dict) -> None:
        # The Qwen3.8 vLLM stack accepts exactly one system message, at the
        # very beginning, so prompt + instructions + chat snapshots are merged.
        system_parts = [BASE_PROMPT]
        if self.instructions:
            system_parts.append(self.instructions)
        system_parts.extend(list(self.context.values())[-3:])
        messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        messages.extend(self.history[-MAX_HISTORY:])
        message = await self.upstreams.chat(messages, self.tools)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            await self._emit_tool_calls(rid, state, tool_calls)
            return
        content = speech_text(message.get("content") or "")
        if content:
            await self._speak(rid, state, content)
        else:
            await self.finish(rid, state, "completed")

    async def _emit_tool_calls(self, rid: str, state: dict, tool_calls: list) -> None:
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name not in {"generate_chat_completion", "play_animation"}:
                log.info("ignoring unconfigured tool call name=%s", name)
                continue
            item_id = self.new_item("fc")
            await self.send(
                {
                    "type": "response.output_item.done",
                    "response_id": rid,
                    "output_index": index,
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "call_id": str(call.get("id") or item_id),
                        "name": name,
                        "arguments": function.get("arguments") or "{}",
                        "status": "completed",
                    },
                }
            )
        await self.finish(rid, state, "completed")

    async def _speak_function_result(self, rid: str, state: dict, call_id: str) -> None:
        result = self.function_results.pop(call_id, None)
        if result is None:
            await self._speak(rid, state, "The result is not ready yet. Please try again.")
            return
        if result.get("status") == "failed":
            await self._speak(rid, state, "That request failed. You can ask me to retry.")
            return
        answer = speech_text(str(result.get("answer") or ""))
        if not answer:
            answer = "Done."
        self.history.append({"role": "assistant", "content": answer})
        self.trim_history()
        await self._speak(rid, state, answer)

    async def _speak(self, rid: str, state: dict, text: str) -> None:
        state["speaking"] = True
        item_id = self.new_item("out")
        full_text = ""
        for sentence in sentences(text):
            if state["cancelled"]:
                break
            try:
                pcm = await self.upstreams.tts(sentence, self.voice)
            except UpstreamError as exc:
                log.warning("tts failed session=%s: %s", self.session_id, exc)
                break
            full_text = (full_text + " " + sentence).strip()
            await self.send(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": rid,
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": sentence + " ",
                }
            )
            for off in range(0, len(pcm), CHUNK):
                if state["cancelled"]:
                    break
                chunk = pcm[off : off + CHUNK]
                key = (item_id, 0)
                self.emitted[key] = self.emitted.get(key, 0) + len(chunk) // 2
                await self.send(
                    {
                        "type": "response.output_audio.delta",
                        "response_id": rid,
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": base64.b64encode(chunk).decode(),
                    }
                )
                await asyncio.sleep(0.005)
        state["speaking"] = False
        if full_text:
            await self.send(
                {
                    "type": "response.output_audio_transcript.done",
                    "response_id": rid,
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": full_text,
                }
            )
        await self.finish(rid, state, "cancelled" if state["cancelled"] else "completed")

    async def finish(self, rid: str, state: dict, status: str) -> None:
        if state.get("done"):
            return
        state["done"] = True
        state["speaking"] = False
        await self.send(
            {
                "type": "response.done",
                "response": {
                    "id": rid,
                    "object": "realtime.response",
                    "status": status,
                    "metadata": state.get("metadata") or {},
                },
            }
        )
        self.responses.pop(rid, None)

    def trim_history(self) -> None:
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]
