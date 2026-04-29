#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
================================================================================
PRE-DEVELOPMENT MODULE - NOT CURRENTLY IN USE
================================================================================

This module implements a Client-Server distributed architecture for Agilex
robot control, where:
    - Robot Host (172.16.18.148): Runs ZMQ bridge + Dora nodes
    - Remote PC: Runs this Agilex client + Policy inference

STATUS: Pre-developed but NOT deployed.

REASON: The robot host has an RTX 4090 GPU, so policy inference runs directly
on the robot using Dora dataflow (Method A in ACT_POLICY_DEPLOYMENT_GUIDE.md).
This Client-Server approach (Method B) is kept as a backup for future use.

DEPLOYMENT METHOD IN USE:
    Method A - On-Robot Policy Inference via Dora
    - Policy node runs on robot: ~/Documents/RoboDriver/components/policy/dora-policy-act/
    - Dataflow config: dataflow_policy.yml
    - No remote PC needed for inference

TO ENABLE THIS MODULE IN FUTURE:
    1. Deploy ZMQ bridge node on robot (dora_zmq_bridge)
    2. Update dataflow to include bridge node
    3. Use: lerobot-record --robot.type=agilex --policy.path=...

================================================================================

Agilex Cobot Magic dual-arm robot implementation for LeRobot.

This module provides a LeRobot-compatible Robot class for the Agilex Cobot Magic
dual-arm robot. It communicates with the robot via ZMQ bridge running on the
robot's onboard computer.

Architecture:
    [Robot Host - Dora + ZMQ Bridge] <--> [Remote PC - Agilex Robot Class]
                    |                              |
            Camera/Arm Nodes                  Policy Inference
                    |                              |
              ZMQ PUB/SUB                     ZMQ SUB/PUSH

Action Space (14 DOF):
    action[0:6]   = right_arm joints (radians)
    action[6]     = right_arm gripper (normalized 0-1)
    action[7:13]  = left_arm joints (radians)
    action[13]    = left_arm gripper (normalized 0-1)

Observation Space:
    - Images: 3 cameras (top, right, left)
    - Joint states: 14 DOF (radians)
    - End effector poses: 2x (x, y, z, roll, pitch, yaw)
"""

import json
import time
from typing import Any

import numpy as np
import torch
import zmq

from lerobot.types import RobotAction, RobotObservation

from .config_agilex import AgilexConfig


class Agilex:
    """
    LeRobot Robot class for Agilex Cobot Magic dual-arm robot.

    This class communicates with the robot via ZMQ bridge, receiving
    observations and sending actions over the network.
    """

    config_class = AgilexConfig
    name = "agilex"

    def __init__(self, config: AgilexConfig):
        self.config = config
        self.robot_type = self.name
        self.id = config.id

        # ZMQ context and sockets
        self._ctx: zmq.Context | None = None
        self._state_sock: zmq.Socket | None = None
        self._cmd_sock: zmq.Socket | None = None

        # Connection state
        self._is_connected = False

        # Observation buffer
        self._last_observation: dict[str, Any] = {}

        # Camera configuration
        self._camera_height = 480
        self._camera_width = 640

    @property
    def observation_features(self) -> dict:
        """Return the observation feature specification."""
        features = {
            # Joint states: 14 DOF (right[0:7], left[7:14])
            "joint_state": (14,),
        }
        # Add camera features if configured
        for cam_name in ["top", "right", "left"]:
            features[f"image_{cam_name}"] = (self._camera_height, self._camera_width, 3)
        return features

    @property
    def action_features(self) -> dict:
        """Return the action feature specification."""
        return {
            # 14 DOF action: right[0:7], left[7:14]
            "action": (14,),
        }

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Agilex robot does not require calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot via ZMQ bridge.

        Args:
            calibrate: Ignored for Agilex (no calibration needed).
        """
        if self._is_connected:
            return

        self._ctx = zmq.Context()

        # SUB socket for receiving state
        self._state_sock = self._ctx.socket(zmq.SUB)
        self._state_sock.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message
        self._state_sock.connect(f"tcp://{self.config.robot_ip}:{self.config.state_port}")
        self._state_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        # PUSH socket for sending commands
        self._cmd_sock = self._ctx.socket(zmq.PUSH)
        self._cmd_sock.setsockopt(zmq.CONFLATE, 1)
        self._cmd_sock.connect(f"tcp://{self.config.robot_ip}:{self.config.cmd_port}")

        self._is_connected = True
        print(f"Connected to Agilex robot at {self.config.robot_ip}")

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None
            self._state_sock = None
            self._cmd_sock = None

        self._is_connected = False
        print("Disconnected from Agilex robot")

    def calibrate(self) -> None:
        """No calibration required for Agilex robot."""
        pass

    def configure(self) -> None:
        """No additional configuration required."""
        pass

    def get_observation(self) -> RobotObservation:
        """
        Receive observation from the robot.

        Returns:
            RobotObservation: Dictionary containing:
                - joint_state: numpy array [14] (radians)
                - image_top, image_right, image_left: numpy arrays [H, W, 3]
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected. Call connect() first.")

        # Receive state from ZMQ
        payload = self._state_sock.recv()
        state = json.loads(payload.decode("utf-8"))

        observation = {}

        # Process joint state
        if "joint_state" in state and state["joint_state"] is not None:
            observation["joint_state"] = np.array(state["joint_state"], dtype=np.float32)
        else:
            observation["joint_state"] = np.zeros(14, dtype=np.float32)

        # Process images
        for cam_name in ["top", "right", "left"]:
            key = f"image_{cam_name}"
            if key in state and state[key] is not None:
                img_data = state[key]
                if isinstance(img_data, dict) and "data" in img_data:
                    # Image is sent as dict with metadata
                    img_bytes = bytes(img_data["data"])
                    h = img_data.get("height", self._camera_height)
                    w = img_data.get("width", self._camera_width)
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    arr = arr.reshape(h, w, 3)
                    observation[key] = arr
                else:
                    # Image is sent as raw bytes
                    arr = np.frombuffer(bytes(img_data), dtype=np.uint8)
                    arr = arr.reshape(self._camera_height, self._camera_width, 3)
                    observation[key] = arr
            else:
                # Return zero image if not available
                observation[key] = np.zeros(
                    (self._camera_height, self._camera_width, 3), dtype=np.uint8
                )

        self._last_observation = observation
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action to the robot.

        Args:
            action: Dictionary with 'action' key containing numpy array [14]
                    action[0:6] = right joints, action[6] = right gripper
                    action[7:13] = left joints, action[13] = left gripper

        Returns:
            The action that was sent (may be clipped).
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected. Call connect() first.")

        # Extract action array
        if isinstance(action, dict):
            action_arr = action.get("action", np.zeros(14, dtype=np.float32))
        else:
            action_arr = action

        action_arr = np.array(action_arr, dtype=np.float32)

        # Clip action to safe range
        action_arr = np.clip(action_arr, -np.pi, np.pi)

        # Build command dict
        cmd = {
            "right_joint": action_arr[0:6].tolist(),
            "right_gripper": float(action_arr[6]),
            "left_joint": action_arr[7:13].tolist(),
            "left_gripper": float(action_arr[13]),
        }

        # Send via ZMQ
        payload = json.dumps(cmd).encode("utf-8")
        self._cmd_sock.send(payload)

        return {"action": action_arr}

    def teleop_step(self) -> None:
        """Teleop not supported for remote Agilex client."""
        raise NotImplementedError("Teleop not supported for Agilex remote client")


def make_agilex_from_config(config: AgilexConfig) -> Agilex:
    """Factory function to create Agilex robot from config."""
    return Agilex(config)
