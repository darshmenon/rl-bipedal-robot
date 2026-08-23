"""FastAPI server: web UI + websocket voice/chat pipeline for the humanoid.

Pipeline per turn: audio (webm/opus) -> faster-whisper -> Ollama (llama3.2)
-> optional ROS2 execute_behavior action goal -> Kokoro TTS -> audio back.
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import llm_agent
import stt
import tts
from ros_bridge import get_bridge

APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.on_event("startup")
async def warm_up():
    asyncio.get_event_loop().run_in_executor(None, get_bridge)


async def _send_json(ws: WebSocket, payload: dict):
    await ws.send_text(json.dumps(payload))


async def _handle_utterance(ws: WebSocket, user_text: str, history: list[dict]):
    user_text = user_text.strip()
    if not user_text:
        return

    decision = await asyncio.to_thread(llm_agent.decide, user_text, history)
    action = decision["action"]
    reply = decision["reply"]

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    await _send_json(ws, {"type": "assistant", "text": reply, "action": action})

    if action:
        bridge = get_bridge()
        success, message = await asyncio.to_thread(bridge.send_behavior, action)
        await _send_json(
            ws,
            {"type": "action_result", "action": action, "success": success, "message": message},
        )

    audio = await asyncio.to_thread(tts.synthesize, reply)
    await ws.send_bytes(audio)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []
    await _send_json(ws, {"type": "ready"})

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                audio_bytes = message["bytes"]
                text = await asyncio.to_thread(stt.transcribe, audio_bytes)
                await _send_json(ws, {"type": "transcript", "text": text})
                await _handle_utterance(ws, text, history)

            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "text":
                    await _handle_utterance(ws, payload.get("text", ""), history)

    except WebSocketDisconnect:
        pass
