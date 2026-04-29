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
Dora node for merging robot state from dual arms.

This node receives joint states and end effector poses from right and left arm nodes,
concatenates them into a single 26-dimensional state vector.

State Structure (26 DOF):
    [0:7]   - Right arm joints [0:6] + gripper [6]
    [7:13]  - Right arm end effector pose (x, y, z, rx, ry, rz)
    [13:20] - Left arm joints [0:6] + gripper [6]
    [20:26] - Left arm end effector pose (x, y, z, rx, ry, rz)

Inputs:
    right_joint: pyarrow array [6] (right arm joint positions in radians)
    right_gripper: pyarrow scalar (right gripper position 0-1)
    right_endpose: pyarrow array [6] (right arm end effector pose)
    left_joint: pyarrow array [6] (left arm joint positions in radians)
    left_gripper: pyarrow scalar (left gripper position 0-1)
    left_endpose: pyarrow array [6] (left arm end effector pose)

Outputs:
    robot_state: pyarrow array [26] (merged robot state)
"""

import numpy as np
import pyarrow as pa
from dora import Node


class MergerNode:
    """Dora node for merging robot state from dual arms."""

    def __init__(self):
        self.right_joint = None
        self.right_gripper = 0.0
        self.right_endpose = None
        self.left_joint = None
        self.left_gripper = 0.0
        self.left_endpose = None

    def run(self):
        """Main loop: receive states, merge, and output."""
        node = Node()

        print("MergerNode started, waiting for robot states...")

        for event in node:
            if event["type"] == "INPUT":
                input_id = event["id"]
                data = event["value"]

                if input_id == "right_joint":
                    self.right_joint = np.array(data.to_pylist(), dtype=np.float32)
                elif input_id == "right_gripper":
                    self.right_gripper = float(data.as_py())
                elif input_id == "right_endpose":
                    self.right_endpose = np.array(data.to_pylist(), dtype=np.float32)
                elif input_id == "left_joint":
                    self.left_joint = np.array(data.to_pylist(), dtype=np.float32)
                elif input_id == "left_gripper":
                    self.left_gripper = float(data.as_py())
                elif input_id == "left_endpose":
                    self.left_endpose = np.array(data.to_pylist(), dtype=np.float32)

                # Check if we have all required data
                have_right = (
                    self.right_joint is not None
                    and self.right_endpose is not None
                )
                have_left = (
                    self.left_joint is not None
                    and self.left_endpose is not None
                )

                if have_right and have_left:
                    # Build 26-dimensional state vector
                    state = np.zeros(26, dtype=np.float32)

                    # Right arm: joints [0:6], gripper [6], endpose [7:13]
                    state[0:6] = self.right_joint
                    state[6] = self.right_gripper
                    state[7:13] = self.right_endpose

                    # Left arm: joints [13:19], gripper [19], endpose [20:26]
                    state[13:19] = self.left_joint
                    state[19] = self.left_gripper
                    state[20:26] = self.left_endpose

                    node.send_output("robot_state", pa.array(state.tolist()))

            elif event["type"] == "STOP":
                print("MergerNode stopped")
                break


if __name__ == "__main__":
    merger = MergerNode()
    merger.run()
