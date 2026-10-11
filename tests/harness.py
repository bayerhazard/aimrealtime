"""Local conformance harness: mimics Open WebUI's bridge (v0.12.0) against a
running aimrealtime server, using a real WAV as the microphone input."""

import asyncio
import base64
import json
import os
import socket
import sys
import time
import wave

import websockets

URL = os.environ.get("AIMRT_URL", "ws://127.0.0.1:8099/v1/realtime?model=aimrealtime")
KEY = os.environ.get("AIMRT_KEY", "testkey")
WAV = sys.argv[1] if len(sys.argv) > 1 else "../.scratch/user_question.wav"
EXPECT_TOOL = "--expect-tool" in sys.argv

CHAT_TOOL = {
    "type": "function",
    "name": "generate_chat_completion",
    "description": (
        "Handle substantive questions and tasks using the selected chat model, conversation history, "
        "and configured tools. Use when new reasoning, information, or actions are needed. Always "
        "use for questions about chat tools, capabilities, permissions, or model identity unless a "
        "previous function result already answers them. Do not use for small talk, acknowledgments, "
        "call status, clarification, repeating or rephrasing an available answer, or duplicating "
        "pending or completed work."
    ),
    "parameters": {
        "type": "object",
        "properties": {"request": {"type": "string"}},
        "required": ["request"],
        "additionalProperties": False,
    },
}


def read_pcm(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes()), w.getframerate()


async def main() -> int:
    pcm, rate = read_pcm(WAV)
    assert rate == 24000, f"need 24k wav, got {rate}"
    async with websockets.connect(
        URL,
        additional_headers={"Authorization": f"Bearer {KEY}"},
        max_size=None,
        family=socket.AF_INET,
    ) as ws:
        ev = json.loads(await ws.recv())
        print(f"[recv] {ev['type']}")
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["audio"],
                        "instructions": "You are a helpful voice assistant.",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": 24000},
                                "transcription": {"model": "Qwen/Qwen3-ASR-1.7B"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "interrupt_response": True,
                                    "create_response": False,
                                },
                            },
                            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "female"},
                        },
                        "tools": [CHAT_TOOL],
                        "tool_choice": "auto",
                    },
                }
            )
        )
        ev = json.loads(await ws.recv())
        print(f"[recv] {ev['type']} voice={ev['session'].get('voice')}")

        # microphone: 100 ms chunks, lightly paced
        chunk = 24000 * 2 // 10
        for off in range(0, len(pcm), chunk):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[off : off + chunk]).decode()}))
            await asyncio.sleep(0.005)
        # trailing silence, like a live microphone keeps sending
        silence = b"\x00" * chunk
        for _ in range(15):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode()}))
            await asyncio.sleep(0.005)
        print(f"[sent] microphone {len(pcm)/2/24000:.2f}s + 1.5s silence")

        t0 = time.time()
        responded = set()
        done_count = 0
        audio_samples = 0
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
            except asyncio.TimeoutError:
                print("[timeout] no events for 20s")
                return 1
            ev = json.loads(raw)
            kind = ev["type"]
            if kind in {"response.output_audio.delta"}:
                audio_samples += len(base64.b64decode(ev["delta"])) // 2
                continue
            if kind == "conversation.item.input_audio_transcription.completed":
                print(f"[recv] transcript: {ev['transcript']!r} (+{time.time()-t0:.1f}s)")
                await ws.send(
                    json.dumps({"type": "response.create", "response": {"metadata": {"input_item_id": ev["item_id"]}}})
                )
                responded.add("input")
                continue
            if kind == "response.output_item.done" and ev["item"]["type"] == "function_call":
                item = ev["item"]
                print(f"[recv] function_call {item['name']} args={item['arguments'][:80]}")
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": item["call_id"],
                                "output": json.dumps({"status": "completed", "answer": "In Berlin wird es morgen sonnig, bis 19 Grad."}),
                            },
                        }
                    )
                )
                await ws.send(json.dumps({"type": "response.create", "response": {"metadata": {"call_id": item["call_id"]}}}))
                responded.add("tool")
                continue
            if kind == "response.output_audio_transcript.done":
                print(f"[recv] spoken: {ev.get('transcript','')!r}")
                continue
            if kind == "response.done":
                done_count += 1
                print(f"[recv] response.done status={ev['response']['status']} (+{time.time()-t0:.1f}s)")
                if done_count >= (2 if EXPECT_TOOL else 1):
                    break
                continue
            if kind == "error":
                print(f"[recv] ERROR {ev['error']}")
                continue
            print(f"[recv] {kind}")
        print(f"[result] audio={audio_samples/24000:.2f}s responded={sorted(responded)}")
        ok = "input" in responded and audio_samples > 24000
        if EXPECT_TOOL:
            ok = ok and "tool" in responded
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 1


sys.exit(asyncio.run(main()))
