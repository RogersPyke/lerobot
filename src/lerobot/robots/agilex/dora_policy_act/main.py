#!/usr/bin/env python3

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
Dora node for ACT policy inference on Agilex Cobot Magic robot.

This node receives camera images and joint states from other Dora nodes,
runs policy inference, and outputs actions for the arm nodes.

Environment Variables:
    POLICY_PATH: Path to the trained ACT policy checkpoint
    DEVICE: Device for inference (default: "cuda:0")
    CONTROL_FREQ: Control frequency in Hz (default: 30.0)
    IMAGE_HEIGHT: Camera image height (default: 480)
    IMAGE_WIDTH: Camera image width (default: 640)

Inputs:
    image_top: pyarrow array (flattened RGB bytes from top camera)
    image_right: pyarrow array (flattened RGB bytes from right camera)
    image_left: pyarrow array (flattened RGB bytes from left camera)
    robot_state: pyarrow array [26] (full robot state)
        - [0:7] right arm joints + gripper
        - [7:13] right arm end effector pose (x, y, z, rx, ry, rz)
        - [13:20] left arm joints + gripper
        - [20:26] left arm end effector pose

Outputs:
    action_right_joint: pyarrow array [6] (right arm joint commands)
    action_right_gripper: pyarrow scalar (right gripper command 0-1)
    action_left_joint: pyarrow array [6] (left arm joint commands)
    action_left_gripper: pyarrow scalar (left gripper command 0-1)
"""

import os
import time

import numpy as np
import pyarrow as pa
import torch
from dora import Node

from lerobot.common.policies.act.modeling_act import ACTPolicy


class PolicyNode:
    """Dora node for ACT policy inference."""

    def __init__(self):
        # Environment config
        self.policy_path = os.environ.get("POLICY_PATH", "")
        self.device = os.environ.get("DEVICE", "cuda:0")
        self.control_freq = float(os.environ.get("CONTROL_FREQ", "30.0"))
        self.dt = 1.0 / self.control_freq

        # Camera config
        self.camera_height = int(os.environ.get("IMAGE_HEIGHT", "480"))
        self.camera_width = int(os.environ.get("IMAGE_WIDTH", "640"))

        # Load policy
        if not self.policy_path:
            raise ValueError("POLICY_PATH environment variable must be set")

        print(f"Loading policy from: {self.policy_path}")
        self.policy = ACTPolicy.from_pretrained(self.policy_path)
        self.policy.to(self.device)
        self.policy.eval()
        print(f"Policy loaded successfully on device: {self.device}")

        # State buffer
        self.observation = {
            "top": None,
            "left": None,
            "right": None,
            "robot_state": None,
        }
        self.ready = False

        # Timing
        self.last_inference_time = 0.0

    def process_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Convert raw bytes to tensor [C, H, W].

        Args:
            image_bytes: Raw RGB bytes (H * W * 3)

        Returns:
            torch.Tensor: Normalized image tensor [3, H, W]
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        arr = arr.reshape(self.camera_height, self.camera_width, 3)
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr.copy()).float() / 255.0

    def prepare_observation(self) -> dict:
        """
        Prepare observation dict for policy inference.

        Returns:
            dict: Observation dictionary with keys matching policy input format
        """
        obs = {
            "observation.images.image_top": self.observation["top"].unsqueeze(0).to(self.device),
            "observation.images.image_left": self.observation["left"].unsqueeze(0).to(self.device),
            "observation.images.image_right": self.observation["right"].unsqueeze(0).to(self.device),
            "observation.state": self.observation["robot_state"].unsqueeze(0).to(self.device),
        }
        return obs

    def run(self):
        """Main loop: receive inputs, run inference, send outputs."""
        node = Node()

        print("PolicyNode started, waiting for inputs...")

        for event in node:
            if event["type"] == "INPUT":
                input_id = event["id"]
                data = event["value"]

                # Update observation buffer
                if input_id == "image_top":
                    self.observation["top"] = self.process_image(data.to_pybytes())
                elif input_id == "image_left":
                    self.observation["left"] = self.process_image(data.to_pybytes())
                elif input_id == "image_right":
                    self.observation["right"] = self.process_image(data.to_pybytes())
                elif input_id == "robot_state":
                    state_arr = np.array(data.to_pylist(), dtype=np.float32)
                    self.observation["robot_state"] = torch.from_numpy(state_arr)

                # Check if all observations ready
                self.ready = all(v is not None for v in self.observation.values())

            elif event["type"] == "STOP":
                print("PolicyNode stopped")
                break

            # Inference at control frequency
            if self.ready:
                now = time.time()
                if now - self.last_inference_time >= self.dt:
                    try:
                        obs = self.prepare_observation()
                        with torch.no_grad():
                            action = self.policy.select_action(obs)

                        # action shape: [batch, action_dim] -> [14]
                        action = action.squeeze(0).cpu().numpy()

                        # Clip to safe range
                        action = np.clip(action, -np.pi, np.pi)

                        # Split into right and left arm actions
                        right_joint = action[0:6]
                        right_gripper = float(action[6])
                        left_joint = action[7:13]
                        left_gripper = float(action[13])

                        # Send to arm nodes
                        node.send_output("action_right_joint", pa.array(right_joint.tolist()))
                        node.send_output("action_right_gripper", pa.scalar(right_gripper))
                        node.send_output("action_left_joint", pa.array(left_joint.tolist()))
                        node.send_output("action_left_gripper", pa.scalar(left_gripper))

                        self.last_inference_time = now

                    except Exception as e:
                        print(f"Error during inference: {e}")
                        import traceback
                        traceback.print_exc()
                        continue


if __name__ == "__main__":
    policy_node = PolicyNode()
    policy_node.run()
