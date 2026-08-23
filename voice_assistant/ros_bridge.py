"""Background rclpy action client that talks to humanoid_interface's execute_behavior server."""

from __future__ import annotations

import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMANOID_INTERFACE_DIST_PACKAGES = os.path.join(
    REPO_ROOT, "ros2_ws", "install", "humanoid_interface", "local", "lib", "python3.10", "dist-packages"
)
if HUMANOID_INTERFACE_DIST_PACKAGES not in sys.path:
    sys.path.insert(0, HUMANOID_INTERFACE_DIST_PACKAGES)

import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from humanoid_interface.action import ExecuteBehavior  # noqa: E402
from humanoid_interface.joint_names import VALID_BEHAVIORS  # noqa: E402


class _BehaviorClientNode(Node):
    def __init__(self):
        super().__init__("voice_assistant_behavior_client")
        self._client = ActionClient(self, ExecuteBehavior, "execute_behavior")


class RosBridge:
    """Runs an rclpy executor on a background thread and exposes a blocking send_behavior()."""

    def __init__(self, server_wait_timeout: float = 3.0):
        self.server_wait_timeout = server_wait_timeout
        self._ready = threading.Event()
        self._node: _BehaviorClientNode | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _spin(self):
        rclpy.init(args=None)
        self._node = _BehaviorClientNode()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._ready.set()
        try:
            self._executor.spin()
        finally:
            self._node.destroy_node()
            rclpy.shutdown()

    def server_available(self) -> bool:
        if self._node is None:
            return False
        return self._node._client.server_is_ready()

    def send_behavior(self, behavior: str, timeout: float = 10.0) -> tuple[bool, str]:
        behavior = behavior.strip().lower()
        if behavior not in VALID_BEHAVIORS:
            return False, f"unknown behavior '{behavior}'"

        client = self._node._client
        if not client.wait_for_server(timeout_sec=self.server_wait_timeout):
            return False, "behavior_action_server not available"

        goal = ExecuteBehavior.Goal()
        goal.behavior = behavior

        done = threading.Event()
        outcome = {"success": False, "message": "timed out"}

        def _on_goal_response(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                outcome["message"] = f"goal rejected: {behavior}"
                done.set()
                return

            result_future = goal_handle.get_result_async()

            def _on_result(rf):
                res = rf.result().result
                outcome["success"] = res.success
                outcome["message"] = res.message
                done.set()

            result_future.add_done_callback(_on_result)

        send_future = client.send_goal_async(goal)
        send_future.add_done_callback(_on_goal_response)

        done.wait(timeout=timeout)
        return outcome["success"], outcome["message"]


_bridge_singleton: RosBridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> RosBridge:
    global _bridge_singleton
    with _bridge_lock:
        if _bridge_singleton is None:
            _bridge_singleton = RosBridge()
            time.sleep(0.3)
        return _bridge_singleton
