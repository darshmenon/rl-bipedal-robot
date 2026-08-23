#!/usr/bin/env python3
"""Unified H1 behavior action server — stand, walk, stop, wave_hand."""

from __future__ import annotations

import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from humanoid_interface.action import ExecuteBehavior
from humanoid_interface.joint_names import (
    ALL_JOINTS,
    BEHAVIOR_CMDS,
    VALID_BEHAVIORS,
)
from humanoid_interface.policy_runner import H1PolicyRunner
from humanoid_interface.standing_controller import StandingController
from humanoid_interface.wave_controller import WaveController

MODE_STAND = "stand"
MODE_WALK = "walk"
MODE_WAVE = "wave"
MODE_STOP = "stop"


class BehaviorActionServer(Node):
    def __init__(self):
        super().__init__("behavior_action_server")
        self._cb_group = ReentrantCallbackGroup()
        self._mode = MODE_STAND
        self._mode_lock = threading.Lock()
        self._active_goal = None
        self._active_handle = None

        self._policy = H1PolicyRunner()
        self._standing = StandingController()
        self._wave = WaveController()

        self._pubs = {
            name: self.create_publisher(Float64, f"/model/h1/joint/{name}/cmd_pos", 10)
            for name in ALL_JOINTS
        }

        self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10, callback_group=self._cb_group
        )
        self.create_subscription(
            Odometry, "/model/h1/odometry", self._odom_cb, 10, callback_group=self._cb_group
        )
        self.create_subscription(
            Twist, "/humanoid/cmd_vel", self._cmd_vel_cb, 10, callback_group=self._cb_group
        )

        self._action_server = ActionServer(
            self,
            ExecuteBehavior,
            "execute_behavior",
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb_group,
        )

        self.create_timer(0.02, self._tick, callback_group=self._cb_group)
        self.get_logger().info(
            f"Behavior action server ready (policy={self._policy.policy_path})"
        )
        self.get_logger().info(f"Valid behaviors: {', '.join(VALID_BEHAVIORS)}")

    def _goal_cb(self, goal_request):
        behavior = goal_request.behavior.strip().lower()
        if behavior not in VALID_BEHAVIORS:
            self.get_logger().warn(f"Rejected unknown behavior: {behavior}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _joint_cb(self, msg: JointState):
        self._policy.update_joints(msg.name, msg.position, msg.velocity)

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        quat = (q.x, q.y, q.z, q.w)
        w = msg.twist.twist.angular
        omega = (w.x, w.y, w.z)
        self._policy.update_odom(quat, omega)
        self._standing.update_odom(quat, omega)

    def _cmd_vel_cb(self, msg: Twist):
        with self._mode_lock:
            if self._mode == MODE_WALK:
                self._policy.set_cmd(
                    [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]
                )

    def _set_mode(self, behavior: str):
        with self._mode_lock:
            if behavior in ("stand", "stop"):
                self._mode = MODE_STAND if behavior == "stand" else MODE_STOP
                self._wave.stop()
                if behavior == "stop":
                    self._policy.set_cmd([0.0, 0.0, 0.0])
            elif behavior == "wave_hand":
                self._mode = MODE_WAVE
                self._wave.start()
            else:
                self._mode = MODE_WALK
                self._wave.stop()
                self._policy.set_cmd(BEHAVIOR_CMDS[behavior])

    def _publish_targets(self, targets: dict[str, float]):
        for name, value in targets.items():
            msg = Float64()
            msg.data = value
            self._pubs[name].publish(msg)

    def _tick(self):
        with self._mode_lock:
            mode = self._mode

        if mode == MODE_WALK:
            targets = self._policy.step()
        elif mode == MODE_WAVE:
            targets, progress, done = self._wave.step(0.02)
            if done and self._active_handle is not None:
                self._finish_active_goal(True, "wave_hand complete", "done")
        else:
            targets = self._standing.compute_targets()

        self._publish_targets(targets)

        if self._active_handle is not None and mode not in (MODE_WAVE,):
            fb = ExecuteBehavior.Feedback()
            fb.status = mode
            fb.progress = 1.0 if mode == MODE_STAND else 0.5
            self._active_handle.publish_feedback(fb)

    async def _execute_cb(self, goal_handle):
        behavior = goal_handle.request.behavior.strip().lower()
        self._active_goal = behavior
        self._active_handle = goal_handle
        self._set_mode(behavior)

        if behavior == "wave_hand":
            # Completion/abort for wave_hand is driven by _tick() ->
            # _finish_active_goal(), which already calls goal_handle.succeed()
            # (rclpy only auto-aborts on return if the goal is still active,
            # so calling succeed()/abort() here too would double-terminate
            # the goal and crash the executor). This loop just blocks the
            # coroutine's worker thread until that happens or is canceled;
            # the periodic timer runs on its own thread via the reentrant
            # callback group, so this sleep does not stall it.
            while rclpy.ok() and self._active_handle is goal_handle:
                if goal_handle.is_cancel_requested:
                    self._wave.stop()
                    self._set_mode("stand")
                    goal_handle.canceled()
                    self._clear_active()
                    result = ExecuteBehavior.Result()
                    result.success = False
                    result.message = "wave_hand canceled"
                    return result
                time.sleep(0.05)

            result = ExecuteBehavior.Result()
            result.success = True
            result.message = "wave_hand complete"
            return result

        fb = ExecuteBehavior.Feedback()
        fb.status = f"executing {behavior}"
        fb.progress = 1.0
        goal_handle.publish_feedback(fb)
        goal_handle.succeed()
        self._clear_active()
        result = ExecuteBehavior.Result()
        result.success = True
        result.message = f"{behavior} active"
        return result

    def _finish_active_goal(self, success: bool, message: str, status: str):
        if self._active_handle is None:
            return
        fb = ExecuteBehavior.Feedback()
        fb.status = status
        fb.progress = 1.0
        self._active_handle.publish_feedback(fb)
        if success:
            self._active_handle.succeed()
        else:
            self._active_handle.abort()
        result = ExecuteBehavior.Result()
        result.success = success
        result.message = message
        self._clear_active()
        self._set_mode("stand")
        return result

    def _clear_active(self):
        self._active_goal = None
        self._active_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
