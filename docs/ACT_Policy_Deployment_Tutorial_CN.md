# ACT Policy Deployment Tutorial for Agilex Cobot Magic

本文档提供在 Agilex Cobot Magic 双臂机器人上部署 ACT 策略的详细步骤指南。

---

## 目录

1. [系统概述](#1-系统概述)
2. [方案A：机载策略推理（推荐）](#2-方案a机载策略推理推荐)
3. [方案B：客户端-服务器架构](#3-方案b客户端-服务器架构)
4. [常见问题排查](#4-常见问题排查)

---

## 1. 系统概述

### 1.1 硬件配置

| 组件 | 规格 |
|------|------|
| 机器人 | Agilex Cobot Magic 双臂 |
| 机械臂 | 2x Piper 6自由度 + 夹爪 |
| 相机 | 3x Intel RealSense (1x D455 顶部, 2x D435 侧面) |
| 机载电脑 | IP: 172.16.18.148, 用户: agilex, 密码: agx |

### 1.2 软件架构

- **框架**: Dora 分布式数据流
- **机械臂 SDK**: piper_sdk (CAN 总线通信)
- **相机 SDK**: pyrealsense2
- **CAN 总线**: can_right (右臂), can_left (左臂)

### 1.3 动作空间 (14 DOF)

```
action[0:7]  = 右臂 [关节0..关节5, 夹爪]
action[7:14] = 左臂 [关节0..关节5, 夹爪]
```

- 关节角度单位：弧度 (radians)
- 夹爪：归一化值 [0, 1]，0=张开，1=闭合

---

## 2. 方案A：机载策略推理（推荐）

### 2.1 方案说明

将训练好的策略部署到机器人机载电脑上，通过 Dora 数据流框架实现端到端的策略推理。

**优点**：
- 低延迟：所有计算在本地完成
- 稳定性高：不依赖网络连接
- 实时性好：适合高频控制

**架构图**：
```
[相机节点] ──┐
             ├──> [策略节点] ──> [机械臂节点]
[关节状态] ──┘
```

### 2.2 准备工作

#### 步骤 1：SSH 连接到机器人

```bash
ssh agilex@172.16.18.148
# 密码: agx
```

#### 步骤 2：检查环境

```bash
# 检查 CAN 总线
ip link show can_left
ip link show can_right

# 检查相机
rs-enumerate-devices

# 检查 Dora
dora --version
```

#### 步骤 3：创建策略节点目录

```bash
mkdir -p ~/Documents/RoboDriver/components/policy/dora-policy-act/dora_policy_act
cd ~/Documents/RoboDriver/components/policy/dora-policy-act
```

### 2.3 创建策略节点

#### 步骤 4：创建 `main.py`

在 `~/Documents/RoboDriver/components/policy/dora-policy-act/dora_policy_act/` 目录下创建 `main.py` 文件，内容参考 `ACT_POLICY_DEPLOYMENT_GUIDE.md` 第 5.2 节。

#### 步骤 5：创建 `__init__.py`

```bash
touch ~/Documents/RoboDriver/components/policy/dora-policy-act/dora_policy_act/__init__.py
```

#### 步骤 6：注册 Dora 节点

编辑 `~/Documents/RoboDriver/components/policy/dora-policy-act/pyproject.toml`：

```toml
[project]
name = "dora-policy-act"
version = "0.1.0"

[project.entry-points."dora.plugins"]
dora-policy-act = "dora_policy_act.main:PolicyNode"
```

### 2.4 传输训练好的策略

#### 步骤 7：从远程 PC 复制策略到机器人

```bash
# 在远程 PC 上执行
scp -r /path/to/your/trained/policy agilex@172.16.18.148:~/policies/act_policy
```

### 2.5 配置数据流

#### 步骤 8：创建数据流配置文件

在 `~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/` 目录下创建 `dataflow_policy.yml`，内容参考 `ACT_POLICY_DEPLOYMENT_GUIDE.md` 第 5.3 节。

**重要配置项**：
- `POLICY_PATH`: 策略文件路径，如 `/home/agilex/policies/act_policy`
- `DEVICE`: 推理设备，`cuda:0` 或 `cpu`
- `CONTROL_FREQ`: 控制频率，建议 30 Hz

### 2.6 运行策略

#### 步骤 9：启动机器人

```bash
# 确保机械臂已上电
# 确保急停按钮已释放
```

#### 步骤 10：运行数据流

```bash
cd ~/Documents/RoboDriver/robodriver/robots/robodriver-robot-agilex-aloha-aio-dora/dora/
dora run dataflow_policy.yml
```

#### 步骤 11：停止运行

按 `Ctrl+C` 停止数据流。

---

## 3. 方案B：客户端-服务器架构

### 3.1 方案说明

在远程 PC 上运行策略推理，通过网络将观测数据传输到远程 PC，再将动作指令传回机器人。

**优点**：
- 灵活性高：可以使用更强大的 GPU
- 调试方便：便于监控和调试
- 资源占用低：机器人端计算负载小

**缺点**：
- 网络延迟：可能影响实时性
- 稳定性依赖网络

**架构图**：
```
[机器人端]                          [远程 PC]
    |                                   |
    +-- ZmqBridge 节点                  +-- AgilexClient
    |     |                             |     |
    |     +-- PUB 状态 (ZMQ) ---------> | -- 接收观测
    |     |                             |     |
    |     +-- PULL 指令 (ZMQ) <-------- | -- 发送动作
    |     |                             |     |
    +-- Dora 机械臂/相机节点            +-- 策略推理
```

### 3.2 机器人端配置

#### 步骤 1：创建 ZMQ 桥接节点

```bash
mkdir -p ~/Documents/RoboDriver/components/bridge/dora-zmq-bridge/dora_zmq_bridge
```

#### 步骤 2：创建桥接节点代码

在上述目录创建 `main.py`，内容参考 `ACT_POLICY_DEPLOYMENT_GUIDE.md` 第 6.2 节。

#### 步骤 3：创建 `__init__.py`

```bash
touch ~/Documents/RoboDriver/components/bridge/dora-zmq-bridge/dora_zmq_bridge/__init__.py
```

#### 步骤 4：更新数据流配置

在数据流配置中添加桥接节点：

```yaml
- id: zmq_bridge
  path: dora-zmq-bridge
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
    STATE_PORT: "5555"
    CMD_PORT: "5556"
    IMAGE_HEIGHT: "480"
    IMAGE_WIDTH: "640"
```

#### 步骤 5：运行桥接数据流

```bash
dora run dataflow_bridge.yml
```

### 3.3 远程 PC 配置

#### 步骤 6：添加 AgilexClient 到 LeRobot

在 LeRobot 项目中创建文件 `src/lerobot/robots/agilex/agilex_client.py`，内容参考 `ACT_POLICY_DEPLOYMENT_GUIDE.md` 第 6.3 节。

#### 步骤 7：注册 Robot 配置

在 `src/lerobot/robots/agilex/__init__.py` 中添加：

```python
from .agilex_client import AgilexClient, AgilexClientConfig
```

#### 步骤 8：创建推理脚本

```python
#!/usr/bin/env python3
"""Remote policy inference script."""

import torch
import numpy as np
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.robots.agilex.agilex_client import AgilexClient, AgilexClientConfig

# Configuration
config = AgilexClientConfig(
    remote_ip="172.16.18.148",
    state_port=5555,
    cmd_port=5556,
    control_freq=30.0,
)

# Load policy
policy = ACTPolicy.from_pretrained("/path/to/your/policy")
policy.to("cuda:0")
policy.eval()

# Connect to robot
client = AgilexClient(config)
client.connect()

try:
    while True:
        # Get observation
        obs = client.get_observation()

        # Prepare observation for policy
        policy_obs = {
            "observation.images.top": obs["images"]["top"].unsqueeze(0).cuda(),
            "observation.images.left": obs["images"]["left"].unsqueeze(0).cuda(),
            "observation.images.right": obs["images"]["right"].unsqueeze(0).cuda(),
            "observation.state": obs["joint_state"].unsqueeze(0).cuda(),
        }

        # Inference
        with torch.no_grad():
            action = policy.select_action(policy_obs)

        # Send action
        action_np = action.squeeze(0).cpu().numpy()
        client.send_action(action_np)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    client.disconnect()
```

#### 步骤 9：运行推理

```bash
python remote_inference.py
```

---

## 4. 常见问题排查

### 4.1 CAN 总线问题

**症状**：机械臂无响应

**排查步骤**：

```bash
# 检查 CAN 接口状态
ip link show can_left
ip link show can_right

# 如果接口未启动，手动启动
sudo ip link set can_left up type can bitrate 1000000
sudo ip link set can_right up type can bitrate 1000000
```

### 4.2 相机问题

**症状**：无法获取图像

**排查步骤**：

```bash
# 检查连接的相机
rs-enumerate-devices

# 重置 USB 设备
sudo usbreset "Intel RealSense D455"
sudo usbreset "Intel RealSense D435"
```

### 4.3 Dora 问题

**症状**：节点无法启动或通信失败

**排查步骤**：

```bash
# 查看节点日志
dora logs <node_id>

# 可视化数据流
dora visualize dataflow.yml

# 检查 Dora 版本
dora --version
```

### 4.4 策略推理问题

**症状**：推理速度慢或报错

**排查步骤**：

1. 检查 GPU 内存：`nvidia-smi`
2. 检查 CUDA 版本：`nvcc --version`
3. 降低图像分辨率
4. 降低控制频率

### 4.5 网络问题（方案B）

**症状**：连接超时或数据丢失

**排查步骤**：

```bash
# 检查网络连通性
ping 172.16.18.148

# 检查端口是否开放
netstat -tlnp | grep 5555
netstat -tlnp | grep 5556

# 检查防火墙
sudo ufw status
```

---

## 5. 相机序列号参考

| 相机位置 | 序列号 | 型号 |
|----------|--------|------|
| 顶部 | 344422071988 | D455 |
| 右侧 | 338622071868 | D435 |
| 左侧 | 339522300665 | D435 |

---

## 6. CAN 总线配置参考

| 机械臂 | CAN 总线 |
|--------|----------|
| 右臂 | can_right |
| 左臂 | can_left |

---

## 7. 控制频率参考

| 节点类型 | 周期 | 频率 |
|----------|------|------|
| 相机 | 100ms | 10 Hz |
| 机械臂 | 50ms | 20 Hz |
| 策略 | 33ms | 30 Hz |

---

## 8. 安全注意事项

1. **急停按钮**：确保急停按钮在操作范围内，随时可以按下
2. **工作空间**：确保机械臂工作空间内无障碍物
3. **速度限制**：首次运行时建议降低控制频率和速度
4. **监控**：运行过程中始终有人监控
5. **备份**：修改配置前备份原始文件

---

## 9. 相关文档

- 技术参考文档：`ACT_POLICY_DEPLOYMENT_GUIDE.md`
- LeRobot 官方文档：https://huggingface.co/lerobot
- Dora 框架文档：https://dora.carsmos.ai
- Piper SDK 文档：联系 Agilex 技术支持

---

**文档版本**: 1.0
**最后更新**: 2026-04-29
