# Voice-controlled humanoid assistant

Talk to the H1 humanoid: speech (or typed text) -> local LLM (Ollama) decides
whether to chat or move the robot -> the existing `humanoid_interface`
`execute_behavior` ROS2 action drives the sim/real robot -> Kokoro speaks the
reply back.

Stack (all local, no cloud APIs):
- STT: faster-whisper (`small.en`, GPU with CPU fallback)
- LLM: Ollama, `llama3.2:3b` (already pulled)
- TTS: Kokoro (ONNX)
- Robot control: existing `humanoid_interface` package's `execute_behavior`
  action (stand, stop, walk_forward, walk_backward, turn_left, turn_right,
  wave_hand)

## One-time setup

Model weights are not committed to git (too large). Download them once:

```bash
mkdir -p voice_assistant/models && cd voice_assistant/models
curl -sL -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
curl -sL -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

Python deps go in the repo's existing `venv`:

```bash
venv/bin/pip install -r voice_assistant/requirements.txt
```

Ollama must be running with the model pulled (`ollama pull llama3.2:3b`).

## Running

1. Launch the sim (or point at the real robot) with the behavior action server:

   ```bash
   source /opt/ros/humble/setup.bash
   source ros2_ws/install/setup.bash
   ros2 launch h1_description h1_gazebo.launch.py controller:=behavior headless:=true
   ```

2. Start the voice server (needs the same ROS2 environment sourced, since it
   talks to `execute_behavior` via rclpy):

   ```bash
   source /opt/ros/humble/setup.bash
   source ros2_ws/install/setup.bash
   venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
   # (run from inside voice_assistant/)
   ```

3. Open `http://localhost:8000` in a browser. Hold the mic button and talk,
   or type into the text box.

## Notes / known limits

- First TTS call per server process is slow (~15-20s, one-time ONNX model
  load); subsequent replies take ~5-7s on this GPU. Good enough for a demo,
  not yet "instant".
- The LLM only ever picks from the fixed behavior list in
  `humanoid_interface/joint_names.py::VALID_BEHAVIORS` — it can't invent new
  motions.
- `ros_bridge.py` locates the built `humanoid_interface` package via a
  relative path from this file; if you move `voice_assistant/` out of the
  repo root, update `REPO_ROOT` there.
