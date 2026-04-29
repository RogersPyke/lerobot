# ACT Policy Deployment Guide for Agilex Cobot Magic

## 1. System Overview

### 1.1 Hardware Configuration
- **Robot**: Agilex Cobot Magic (Dual-Arm)
- **Arms**: 2x Piper 6-DOF arms with grippers
- **Cameras**: 3x Intel RealSense (1x D455 top, 2x D435 side)
- **Host PC**: 172.16.18.148 (user: agilex, password: agx)

### 1.2 Interface Stack
- **Framework**: Dora (distributed dataflow)
- **Arm SDK**: piper_sdk (CAN bus communication)
- **Camera SDK**: pyrealsense2
- **CAN Buses**: can_right, can_left

### 1.3 Action Space (14 DOF)
```
action[0:7]  = right_arm [joint_0..joint_5, gripper]
action[7:14] = left_arm  [joint_0..joint_5, gripper]
```
- Joints: radians
- Gripper: normalized [0, 1] -> open to close

### 1.4 Observation Space
- **Images**: 3 cameras (top, right, left) - RGB
- **Joint States**: 14 DOF (radians)
- **End Effector Poses**: 2x (x, y, z, roll, pitch, yaw)

---

## 2. Key File Paths (Robot Host)

### 2.1 Piper SDK
```
~/piper_sdk/piper_sdk/interface/piper_interface_v2.py
```

### 2.2 Dora Components
```
~/Documents/RoboDriver/components/arms/dora-arm-piper/dora_arm_piper/main.py
~/Documents/RoboDriver/components/cameras/dora-camera-realsense/dora_camera_realsense/main.py
```

### 2.3 Dora Dataflow Configs
```
~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/dataflow_realsense.yml
~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/dataflow.yml
```

### 2.4 Robot Environment
```
~/robot-backend/robot/robot_env.py
```

---

## 3. Piper SDK Quick Reference

### 3.1 Conversion Factors
```python
FACTOR = 57324.840764  # radians -> piper integer (based on 0.001 degrees)
GRIPPER_FACTOR = 1000 * 100 * 10  # normalized -> piper integer
```

### 3.2 Key Methods
```python
# Connection
ConnectPort(can_port: str) -> bool
DisconnectPort() -> None
EnableArm(enable: int) -> None  # 7 = enable, 0 = disable

# Motion Control
MotionCtrl_2(ctrl_mode: int, move_mode: int, move_spd_rate_ctrl: int, ...)
JointCtrl(joint_0: int, joint_1: int, joint_2: int, joint_3: int, joint_4: int, joint_5: int)
GripperCtrl(gripper_angle: int, gripper_effort: int, gripper_code: int, set_zero: int)

# State Reading
GetArmJointMsgs() -> tuple[joint_0..joint_5, velocity, torque]
GetArmGripperMsgs() -> tuple[angle, effort, code]
GetArmEndPoseMsgs() -> tuple[x, y, z, rx, ry, rz]
```

### 3.3 Joint Encoding
```python
# Position command (radians -> piper integer)
joint_piper = round(position_radians * FACTOR)

# Gripper command (normalized -> piper integer)
gripper_piper = round(gripper_normalized * GRIPPER_FACTOR)
```

---

## 4. Dora Node I/O Reference

### 4.1 dora-arm-piper
**Environment Variables:**
- TEACH_MODE: "0" or "1"
- CAN_BUS: "can_left" or "can_right"

**Outputs:**
- follower_jointstate: pyarrow array [joint_0..joint_5] (radians)
- leader_jointstate: same as follower (for teleop)
- follower_endpose: pyarrow array [x, y, z, rx, ry, rz]

**Inputs:**
- action_joint: pyarrow array [joint_0..joint_5] (radians)
- action_gripper: pyarrow scalar (normalized 0-1)
- action_endpose: pyarrow array [x, y, z, rx, ry, rz]

### 4.2 dora-camera-realsense
**Environment Variables:**
- DEVICE_SERIAL: camera serial number
- IMAGE_HEIGHT: default 480
- IMAGE_WIDTH: default 640
- ENCODING: "rgb8" or "bgr8"
- FLIP: "0" or "1"

**Outputs:**
- image: pyarrow array (flattened RGB bytes)
- image_depth: pyarrow array (flattened depth bytes)

---

## 5. Method A: On-Robot Policy Inference via Dora

### 5.1 Architecture
```
[Camera Nodes] -> [Policy Node] -> [Arm Nodes]
     |                  |              |
     v                  v              v
  images           action         joint_cmd
```

### 5.2 PolicyNode Implementation

**File: ~/Documents/RoboDriver/components/policy/dora-policy-act/dora_policy_act/main.py**

```python
#!/usr/bin/env python3
"""Dora node for ACT policy inference on Agilex robot."""

import os
import time
import pyarrow as pa
from dora import Node

import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy


class PolicyNode:
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
        self.policy = ACTPolicy.from_pretrained(self.policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        # State buffer
        self.observation = {
            "top": None,
            "left": None,
            "right": None,
            "joint_state": None,
        }
        self.ready = False

    def process_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw bytes to tensor [C, H, W]."""
        import numpy as np
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        arr = arr.reshape(self.camera_height, self.camera_width, 3)
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float() / 255.0

    def prepare_observation(self) -> dict:
        """Prepare observation dict for policy inference."""
        obs = {
            "observation.images.top": self.observation["top"].unsqueeze(0).to(self.device),
            "observation.images.left": self.observation["left"].unsqueeze(0).to(self.device),
            "observation.images.right": self.observation["right"].unsqueeze(0).to(self.device),
            "observation.state": self.observation["joint_state"].unsqueeze(0).to(self.device),
        }
        return obs

    def run(self):
        node = Node()
        last_inference_time = 0.0

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
                elif input_id == "joint_state":
                    import numpy as np
                    joint_arr = np.array(data.to_pylist(), dtype=np.float32)
                    self.observation["joint_state"] = torch.from_numpy(joint_arr)

                # Check if all observations ready
                self.ready = all(v is not None for v in self.observation.values())

            elif event["type"] == "STOP":
                break

            # Inference at control frequency
            if self.ready:
                now = time.time()
                if now - last_inference_time >= self.dt:
                    obs = self.prepare_observation()
                    with torch.no_grad():
                        action = self.policy.select_action(obs)

                    # action shape: [batch, action_dim] -> [14]
                    action = action.squeeze(0).cpu().numpy()

                    # Split into right and left arm actions
                    right_action = action[0:7]
                    left_action = action[7:14]

                    # Send to arm nodes
                    node.send_output("action_right_joint", pa.array(right_action[0:6]))
                    node.send_output("action_right_gripper", pa.scalar(right_action[6]))
                    node.send_output("action_left_joint", pa.array(left_action[0:6]))
                    node.send_output("action_left_gripper", pa.scalar(left_action[6]))

                    last_inference_time = now


if __name__ == "__main__":
    policy_node = PolicyNode()
    policy_node.run()
```

### 5.3 Dataflow Configuration

**File: ~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/dataflow_policy.yml**

```yaml
name: agilex_aloha_policy
version: "1.0"

nodes:
  # Camera nodes
  - id: camera_top
    path: dora-camera-realsense
    inputs:
      - tick: dora/timer_100ms
    outputs:
      - image
    env:
      DEVICE_SERIAL: "344422071988"
      IMAGE_HEIGHT: "480"
      IMAGE_WIDTH: "640"
      ENCODING: "rgb8"
      FLIP: "0"

  - id: camera_right
    path: dora-camera-realsense
    inputs:
      - tick: dora/timer_100ms
    outputs:
      - image
    env:
      DEVICE_SERIAL: "338622071868"
      IMAGE_HEIGHT: "480"
      IMAGE_WIDTH: "640"
      ENCODING: "rgb8"
      FLIP: "0"

  - id: camera_left
    path: dora-camera-realsense
    inputs:
      - tick: dora/timer_100ms
    outputs:
      - image
    env:
      DEVICE_SERIAL: "339522300665"
      IMAGE_HEIGHT: "480"
      IMAGE_WIDTH: "640"
      ENCODING: "rgb8"
      FLIP: "0"

  # Arm nodes
  - id: arm_right
    path: dora-arm-piper
    inputs:
      action_joint: policy/action_right_joint
      action_gripper: policy/action_right_gripper
      tick: dora/timer_50ms
    outputs:
      - follower_jointstate
      - follower_endpose
    env:
      TEACH_MODE: "0"
      CAN_BUS: "can_right"

  - id: arm_left
    path: dora-arm-piper
    inputs:
      action_joint: policy/action_left_joint
      action_gripper: policy/action_left_gripper
      tick: dora/timer_50ms
    outputs:
      - follower_jointstate
      - follower_endpose
    env:
      TEACH_MODE: "0"
      CAN_BUS: "can_left"

  # Policy node
  - id: policy
    path: dora-policy-act
    inputs:
      image_top: camera_top/image
      image_left: camera_left/image
      image_right: camera_right/image
      joint_state: joint_merger/joint_state
    outputs:
      - action_right_joint
      - action_right_gripper
      - action_left_joint
      - action_left_gripper
    env:
      POLICY_PATH: "/path/to/trained/policy"
      DEVICE: "cuda:0"
      CONTROL_FREQ: "30.0"
      IMAGE_HEIGHT: "480"
      IMAGE_WIDTH: "640"

  # Joint state merger
  - id: joint_merger
    path: dora-merger
    inputs:
      right: arm_right/follower_jointstate
      left: arm_left/follower_jointstate
    outputs:
      - joint_state
```

### 5.4 Deployment Steps
1. Create policy node directory: mkdir -p ~/Documents/RoboDriver/components/policy/dora-policy-act/dora_policy_act
2. Copy main.py and create __init__.py
3. Update pyproject.toml to register the node
4. Copy trained policy to robot: scp -r /path/to/policy agilex@172.16.18.148:~/policies/
5. Update POLICY_PATH in dataflow YAML
6. Run: dora run dataflow_policy.yml

---

## 6. Method B: Client-Server ZMQ Bridge

### 6.1 Architecture
```
[Robot Host]                    [Remote PC]
  |                                  |
  +-- ZmqBridge Node                 +-- AgilexClient
  |     |                            |     |
  |     +-- PUB state (ZMQ) -------->+-- recv observation
  |     |                            |     |
  |     +-- PULL cmd (ZMQ) <---------+-- send action
  |     |                            |     |
  +-- Dora Arm/Camera Nodes          +-- Policy Inference
```

### 6.2 ZmqBridge Node (Robot Side)

**File: ~/Documents/RoboDriver/components/bridge/dora-zmq-bridge/dora_zmq_bridge/main.py**

```python
#!/usr/bin/env python3
"""ZMQ bridge node for remote policy inference."""

import os
import json
import time
import pyarrow as pa
from dora import Node
import zmq
import numpy as np


class ZmqBridgeNode:
    def __init__(self):
        # ZMQ config
        self.state_port = int(os.environ.get("STATE_PORT", "5555"))
        self.cmd_port = int(os.environ.get("CMD_PORT", "5556"))

        # Camera config
        self.camera_height = int(os.environ.get("IMAGE_HEIGHT", "480"))
        self.camera_width = int(os.environ.get("IMAGE_WIDTH", "640"))

        # Initialize ZMQ
        self.ctx = zmq.Context()

        # PUB socket for state
        self.state_sock = self.ctx.socket(zmq.PUB)
        self.state_sock.bind(f"tcp://0.0.0.0:{self.state_port}")

        # PULL socket for commands
        self.cmd_sock = self.ctx.socket(zmq.PULL)
        self.cmd_sock.bind(f"tcp://0.0.0.0:{self.cmd_port}")
        self.cmd_sock.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout

        # State buffer
        self.state = {
            "images": {},
            "joint_state": None,
            "timestamp": 0.0,
        }

    def run(self):
        node = Node()

        for event in node:
            if event["type"] == "INPUT":
                input_id = event["id"]
                data = event["value"]

                # Update state buffer
                if input_id.startswith("image_"):
                    camera_name = input_id.replace("image_", "")
                    image_bytes = data.to_pybytes()
                    self.state["images"][camera_name] = {
                        "data": list(image_bytes),
                        "height": self.camera_height,
                        "width": self.camera_width,
                    }
                elif input_id == "joint_state":
                    joint_list = data.to_pylist()
                    self.state["joint_state"] = joint_list

                self.state["timestamp"] = time.time()

                # Publish state
                payload = json.dumps(self.state).encode("utf-8")
                self.state_sock.send(payload)

            elif event["type"] == "STOP":
                break

            # Check for commands (non-blocking)
            try:
                cmd_payload = self.cmd_sock.recv()
                cmd = json.loads(cmd_payload.decode("utf-8"))

                # Send commands to arm nodes
                if "right_joint" in cmd:
                    node.send_output("action_right_joint", pa.array(cmd["right_joint"]))
                if "right_gripper" in cmd:
                    node.send_output("action_right_gripper", pa.scalar(cmd["right_gripper"]))
                if "left_joint" in cmd:
                    node.send_output("action_left_joint", pa.array(cmd["left_joint"]))
                if "left_gripper" in cmd:
                    node.send_output("action_left_gripper", pa.scalar(cmd["left_gripper"]))

            except zmq.Again:
                pass  # No command received


if __name__ == "__main__":
    bridge = ZmqBridgeNode()
    bridge.run()
```

### 6.3 AgilexClient (Remote PC - LeRobot Robot Class)

**File: src/lerobot/robots/agilex/agilex_client.py**

```python
#!/usr/bin/env python3
"""LeRobot Robot client for Agilex via ZMQ bridge."""

import json
import time
import zmq
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@dataclass
class AgilexClientConfig(RobotConfig):
    """Configuration for Agilex client."""
    remote_ip: str = "172.16.18.148"
    state_port: int = 5555
    cmd_port: int = 5556
    control_freq: float = 30.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("agilex_client")
class AgilexClient(RobotConfig):
    """LeRobot Robot client for Agilex dual-arm robot via ZMQ bridge."""

    def __init__(self, config: AgilexClientConfig | None = None):
        self.config = config or AgilexClientConfig()
        self.ctx = None
        self.state_sock = None
        self.cmd_sock = None
        self.connected = False

    def connect(self) -> None:
        """Connect to robot ZMQ bridge."""
        self.ctx = zmq.Context()

        # SUB socket for state
        self.state_sock = self.ctx.socket(zmq.SUB)
        self.state_sock.connect(f"tcp://{self.config.remote_ip}:{self.config.state_port}")
        self.state_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        # PUSH socket for commands
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.connect(f"tcp://{self.config.remote_ip}:{self.config.cmd_port}")

        self.connected = True

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if self.ctx:
            self.ctx.term()
        self.connected = False

    def get_observation(self) -> dict[str, Any]:
        """Receive observation from robot."""
        if not self.connected:
            raise RuntimeError("Not connected to robot")

        payload = self.state_sock.recv()
        state = json.loads(payload.decode("utf-8"))

        # Convert images to tensors
        images = {}
        for cam_name, cam_data in state.get("images", {}).items():
            img_bytes = bytes(cam_data["data"])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            arr = arr.reshape(cam_data["height"], cam_data["width"], 3)
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
            images[cam_name] = torch.from_numpy(arr).float() / 255.0

        # Convert joint state to tensor
        joint_state = None
        if state.get("joint_state") is not None:
            joint_state = torch.tensor(state["joint_state"], dtype=torch.float32)

        return {
            "images": images,
            "joint_state": joint_state,
            "timestamp": state.get("timestamp", 0.0),
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send action to robot.

        Args:
            action: numpy array [14] - right[0:7], left[7:14]
        """
        if not self.connected:
            raise RuntimeError("Not connected to robot")

        cmd = {
            "right_joint": action[0:6].tolist(),
            "right_gripper": float(action[6]),
            "left_joint": action[7:13].tolist(),
            "left_gripper": float(action[13]),
        }

        payload = json.dumps(cmd).encode("utf-8")
        self.cmd_sock.send(payload)

    def teleop_step(self) -> None:
        """Not implemented for remote client."""
        raise NotImplementedError("Teleop not supported for remote client")
```

### 6.4 Deployment Steps (Method B)
1. Create bridge node on robot: mkdir -p ~/Documents/RoboDriver/components/bridge/dora-zmq-bridge/dora_zmq_bridge
2. Copy main.py and create __init__.py
3. Update dataflow to include bridge node
4. On remote PC, add AgilexClient to LeRobot
5. Run inference loop on remote PC

---

## 7. Camera Serial Numbers

| Camera | Serial Number | Model |
|--------|---------------|-------|
| Top    | 344422071988  | D455  |
| Right  | 338622071868  | D435  |
| Left   | 339522300665  | D435  |

---

## 8. CAN Bus Configuration

| Arm    | CAN Bus   |
|--------|-----------|
| Right  | can_right |
| Left   | can_left  |

---

## 9. Timer Frequencies

| Timer         | Period | Frequency |
|---------------|--------|-----------|
| Camera tick   | 100ms  | 10 Hz     |
| Arm tick      | 50ms   | 20 Hz     |
| Policy tick   | 33ms   | 30 Hz     |

---

## 10. Troubleshooting

### 10.1 CAN Bus Issues
```bash
# Check CAN interfaces
ip link show can_left
ip link show can_right

# Bring up CAN interface
sudo ip link set can_left up type can bitrate 1000000
```

### 10.2 RealSense Issues
```bash
# Check connected cameras
rs-enumerate-devices

# Reset USB device
sudo usbreset "Intel RealSense D455"
```

### 10.3 Dora Debug
```bash
# Check node logs
dora logs <node_id>

# Visualize dataflow
dora visualize dataflow.yml
```
