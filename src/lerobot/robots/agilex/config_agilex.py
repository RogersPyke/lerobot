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

This configuration is for the Client-Server distributed architecture (Method B).
Currently NOT deployed because the robot host has RTX 4090 GPU and uses Method A
(On-Robot Policy Inference via Dora).

See agilex.py for detailed documentation.
================================================================================
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("agilex")
@dataclass
class AgilexConfig(RobotConfig):
    """Configuration for Agilex Cobot Magic dual-arm robot."""

    # Robot IP for remote connection
    robot_ip: str = "172.16.18.148"

    # ZMQ ports for communication
    state_port: int = 5555
    cmd_port: int = 5556

    # Control frequency
    control_dt: float = 1.0 / 30.0  # 30Hz

    # Cameras (remote via ZMQ)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Number of joints per arm
    joints_per_arm: int = 6

    # Gripper enabled
    gripper_enabled: bool = True
