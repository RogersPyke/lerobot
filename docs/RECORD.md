# RECORD.md - ACT Policy Deployment Changes Log

This document records all changes made for deploying ACT policy on Agilex Cobot Magic robot.

---

## 2026-04-29 13:10 UTC+8 - Initial Setup

### 1. Created Agilex Robot Support in LeRobot

**Directory Created:**
```
/home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/
```

**Files Created:**

1. `__init__.py` - Module initialization
   - Exports: Agilex, AgilexConfig

2. `config_agilex.py` - Robot configuration
   - AgilexConfig dataclass with:
     - robot_ip: str = "172.16.18.148"
     - state_port: int = 5555
     - cmd_port: int = 5556
     - control_dt: float = 1.0/30.0
     - joints_per_arm: int = 6
     - gripper_enabled: bool = True

3. `agilex.py` - Robot implementation
   - Agilex class implementing LeRobot Robot interface
   - ZMQ-based communication (SUB for state, PUSH for commands)
   - 14 DOF action space: right[0:7], left[7:14]
   - Supports 3 cameras (top, right, left)

---

### 2. Created Dora Policy Node

**Directory Created:**
```
/home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dora_policy_act/
```

**Files Created:**

1. `__init__.py` - Module initialization

2. `main.py` - PolicyNode implementation
   - Environment variables:
     - POLICY_PATH: Path to trained ACT policy
     - DEVICE: cuda:0 or cpu
     - CONTROL_FREQ: 30.0 Hz
     - IMAGE_HEIGHT: 480
     - IMAGE_WIDTH: 640
   - Inputs:
     - image_top, image_left, image_right: RGB bytes
     - robot_state: 26-dimensional state vector
   - Outputs:
     - action_right_joint: [6] joint commands
     - action_right_gripper: scalar
     - action_left_joint: [6] joint commands
     - action_left_gripper: scalar

---

### 3. Created Dora State Merger Node

**Directory Created:**
```
/home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dora_merger/
```

**Files Created:**

1. `__init__.py` - Module initialization

2. `main.py` - MergerNode implementation
   - Inputs:
     - right_joint: [6] right arm joints
     - right_endpose: [6] right arm end effector pose
     - left_joint: [6] left arm joints
     - left_endpose: [6] left arm end effector pose
   - Outputs:
     - robot_state: [26] merged state vector
   - State structure:
     - [0:7] right arm (joints + gripper)
     - [7:13] right endpose
     - [13:20] left arm (joints + gripper)
     - [20:26] left endpose

---

### 4. Created Dataflow Configuration

**File Created:**
```
/home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dataflow_policy.yml
```

**Configuration:**
- 3 RealSense cameras (top, right, left)
- 2 Piper arm controllers (right, left)
- 1 state merger node
- 1 policy inference node
- Camera serials: 344422071988 (D455), 338622071868 (D435), 339522300665 (D435)
- CAN buses: can_right, can_left
- Timer frequencies: camera 100ms, arm 50ms

---

### 5. Downloaded Trained Model

**Model Source:**
```
modelscope download --model rogerspyke/TRaDA-models --local_dir /home/rogerspyke/projects/TRaDA/models/act_policy
```

**Model Variants Available:**
- pure_fwd_act_20260429_001915
- pure_bwdrev_act_20260429_001915
- mixed_fwd_bwdrev_1_9_act_20260429_095526
- mixed_fwd_bwdrev_3_7_act_20260429_095526
- mixed_fwd_bwdrev_5_5_act_20260429_001915
- mixed_fwd_bwdrev_7_3_act_20260429_095526
- mixed_fwd_bwdrev_9_1_act_20260429_095526

**Model Configuration:**
- Type: ACT
- Input state: 26 dimensions
- Output action: 14 dimensions
- Cameras: image_top, image_right, image_left (480x640x3)
- Chunk size: 100
- VAE enabled: true

---

## Next Steps (To Be Done on Robot)

1. Copy policy node files to robot:
   ```bash
   scp -r /home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dora_policy_act agilex@172.16.18.148:~/Documents/RoboDriver/components/policy/
   ```

2. Copy merger node files to robot:
   ```bash
   scp -r /home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dora_merger agilex@172.16.18.148:~/Documents/RoboDriver/components/merger/
   ```

3. Copy trained model to robot:
   ```bash
   scp -r /home/rogerspyke/projects/TRaDA/models/act_policy/pure_fwd_act_20260429_001915/checkpoints/030000/pretrained_model agilex@172.16.18.148:~/policies/act_policy
   ```

4. Copy dataflow config to robot:
   ```bash
   scp /home/rogerspyke/projects/TRaDA/third_party/lerobot/src/lerobot/robots/agilex/dataflow_policy.yml agilex@172.16.18.148:~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/
   ```

5. Register Dora nodes on robot (update pyproject.toml)

6. Run dataflow:
   ```bash
   dora run dataflow_policy.yml
   ```

---

## File Summary

| File | Purpose | Status |
|------|---------|--------|
| agilex/__init__.py | Module init | Created |
| agilex/config_agilex.py | Robot config | Created |
| agilex/agilex.py | Robot implementation | Created |
| agilex/dora_policy_act/__init__.py | Policy node init | Created |
| agilex/dora_policy_act/main.py | Policy node implementation | Created |
| agilex/dora_merger/__init__.py | Merger node init | Created |
| agilex/dora_merger/main.py | Merger node implementation | Created |
| agilex/dataflow_policy.yml | Dataflow configuration | Created |
| models/act_policy/ | Trained models | Downloaded |
