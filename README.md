# dog3_sim2sim

在**第二个仿真器**里跑 Isaac 训练出来的 dog3 策略，验证跨仿真器迁移（sim2sim）。

栈来自 [`ddt_ros2_control`](https://github.com/DDTRobot/ddt_ros2_control)（ROS 2 Humble + `ros2_control`
+ `mujoco_ros2_control` + `rl_controller`），在此基础上删掉 tita/d1/d1h、新增 dog3 的描述与配置，
并加了一套 MuJoCo 原生训练/导出工具（`sim2sim_rl/`）。

训练端在另一个仓库：[`dog3lab`](https://github.com/1SUCCESE1/dog3lab)。

## 目录结构

```
dog3_sim2sim/
├── 本地2sim流程.md                     # ★ Gazebo 全流程（编译/启动/键位/导入策略/踩坑）
├── controller/rl_controller/           # ros2_control 控制器（FSM + ONNX 推理）
│   ├── config/dog3/controllers.yaml    #   dog3 策略配置（观测/动作/增益/策略槽位）
│   └── config/dog3/dog3_flat_01.onnx   #   策略（由 dog3lab 导出）
├── simulation/mujoco_bridge/           # mujoco_ros2_control + mujoco_sim_ros2
├── simulation/gazebo_bridge/           # Gazebo 桥接（含 kp/kd 命令接口的 GazeboBridge）
├── interaction/keyboard_controller/    # 键盘交互
├── ros_utils/                          # 话题名
├── urdfs/dog3_description/             # URDF + meshes + xacro + MuJoCo MJCF
├── third_party/mujoco/                 # 预编译 MuJoCo 3.3.0（build 时走 CMAKE_PREFIX_PATH）
└── sim2sim_rl/                         # MuJoCo 原生：向量环境 / 微调 / 导出 / 评测
```

## 编译与运行

见 [`本地2sim流程.md`](./本地2sim流程.md)。要点：

```bash
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin     # 绕开 miniconda
export CMAKE_PREFIX_PATH=$PWD/third_party/mujoco:$CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
colcon build --packages-up-to dog3_description gazebo_bridge rl_controller keyboard_controller

# Gazebo
ros2 launch rl_controller sim_gazebo.launch.py robot:=dog3
# 键盘（另一个终端，同一个 ROS_DOMAIN_ID）
ros2 run keyboard_controller keyboard_controller_node
```

## 未包含的依赖

- **onnxruntime**（121 MB，未入库）：从 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
  取 `onnxruntime-linux-x64-1.10.0`，放到仓库根目录，并
  `export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.10.0/lib:$LD_LIBRARY_PATH`。

## 对 dog3 模型做的改动（与训练端保持一致）

| 改动 | 原因 |
|---|---|
| URDF 关节 origin rpy 平移（thigh +0.7、calf −1.44），限位同步平移 | 使**关节零位 == 默认站姿**；Gazebo 出生即站立，不再依赖起身动作 |
| 后足 link（`LH/RH_FOOT`）后移 0.035 m | 后腿着地点后移，四腿均衡受力（否则后腿在自重下塌陷） |
| Gazebo link `selfCollide=false` | Isaac 训练时自碰撞是关的；Gazebo 默认开会把腿夹住 |
| `joint_pd` 目标 = 略下蹲的对称姿态 | 原来是"冻结当前姿势"，瘫倒后冻在瘫姿 |

> 训练端 `dog3lab` 的 URDF 已同步为同一套原点/限位/后足偏移，两边模型一致。

## 状态

- **Gazebo 可用**：策略能站、能走（命令 0.5 m/s → 实测 ~0.37 m/s，约 75% 跟踪）
- **步态有退化**：训练端学到的是对角 trot（1.38 Hz、对角腿同相），Gazebo 里四腿不再同频
  （前 ~0.85 Hz / 后 ~0.23 Hz）——接触模型（ODE 三角面 vs PhysX）与后足偏移的影响叠加
- MuJoCo 路线已验证但未走通（凸包碰撞 → 已换图元，仍只慢速移动），`sim2sim_rl/` 里的向量环境、
  微调脚本、ONNX 导出（与参考导出逐位一致）可复用
