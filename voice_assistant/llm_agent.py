"""Local LLM (Ollama) agent that turns a user utterance into an optional robot
action plus a short spoken reply."""

from __future__ import annotations

import json

import requests

from ros_bridge import VALID_BEHAVIORS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = f"""You are the voice assistant embedded in a walking humanoid robot.
You can chat naturally AND you can command the robot's body.

Available actions (use the exact string, or null for none):
{json.dumps(VALID_BEHAVIORS)}

- "stand": stand still / relax
- "stop": stop moving immediately
- "walk_forward": walk forward
- "walk_backward": walk backward
- "turn_left" / "turn_right": turn in place
- "wave_hand": wave at the person

Rules:
- Only pick an action when the user is clearly asking the robot to move or do something physical.
- If it's just conversation (small talk, questions, jokes), set action to null.
- Keep "reply" short (1-2 sentences), natural, spoken out loud by the robot.
- Respond ONLY with strict JSON: {{"action": <one of the actions above or null>, "reply": "<text>"}}
"""


def decide(user_text: str, history: list[dict] | None = None) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.4},
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    try:
        parsed = json.loads(content)
        action = parsed.get("action")
        reply = str(parsed.get("reply", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        action = None
        reply = content.strip()

    if action not in VALID_BEHAVIORS:
        action = None
    if not reply:
        reply = "Okay."

    return {"action": action, "reply": reply}
