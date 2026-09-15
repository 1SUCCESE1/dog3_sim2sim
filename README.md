# dog3_sim2sim

把 [dog3lab](https://github.com/1SUCCESE1/dog3lab) 训练出来的 dog3 策略部署到 **Gazebo** 里跑，
用 ROS 2 `ros2_control` 做控制器框架，验证策略在第二个仿真器上的表现。

栈来自 [`ddt_ros2_control`](https://github.com/DDTRobot/ddt_ros2_control)（ROS 2 Humble + `ros2_control`
+ `mujoco_ros2_control` + `rl_controller`），在此基础上保留 dog3 的描述与配置。

## 效果

策略在 Gazebo 中可正常站立、行走、转向，站立姿态左右对称（与 Isaac 训练端一致）。

## 目录结构

```
dog3_sim2sim/
├── 本地2sim流程.md                     # 完整操作流程（编译/启动/键位/导入策略/排查）
├── controller/rl_controller/
│   ├── config/dog3/controllers.yaml    # dog3 策略配置（观测/动作/增益/策略槽位）
│   └── config/dog3/dog3_flat_01.onnx   # 策略（由 dog3lab 导出）
├── simulation/gazebo_bridge/           # Gazebo 桥接（含 kp/kd 命令接口的 GazeboBridge）
├── simulation/mujoco_bridge/           # MuJoCo 桥接（备选）
├── interaction/keyboard_controller/    # 键盘交互
├── ros_utils/                          # 话题名
├── urdfs/dog3_description/             # URDF + meshes + xacro + MuJoCo MJCF
├── third_party/mujoco/                 # 预编译 MuJoCo 3.3.0（编译时经 CMAKE_PREFIX_PATH）
└── tools/bag2csv.py                    # rosbag2 → CSV（给 PlotJuggler 用）
```

## 依赖

- ROS 2 Humble（`gazebo_ros`、`gazebo_ros2_control`、`controller_manager`、`robot_state_publisher`）
- **onnxruntime**（仓库未含，121 MB）：从 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
  取 `onnxruntime-linux-x64-1.10.0` 放到仓库根目录

## 编译

```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin     # 绕开 miniconda
export CPLUS_INCLUDE_PATH=$PWD/onnxruntime-linux-x64-1.10.0/include
export LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.10.0/lib
export CMAKE_PREFIX_PATH=$PWD/third_party/mujoco:$CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
colcon build --packages-up-to dog3_description gazebo_bridge rl_controller keyboard_controller
```

## 运行

**终端 1 —— 启动 Gazebo**
```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin
export ROS_DOMAIN_ID=1
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.10.0/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rl_controller sim_gazebo.launch.py robot:=dog3
```

**终端 2 —— 键盘控制**
```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin
export ROS_DOMAIN_ID=1                    # ★ 必须与终端 1 一致
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run keyboard_controller keyboard_controller_node
```

### 键位

| 键 | 动作 |
|---|---|
| `0` | 切到策略（`dog3_flat`，行走） |
| `7` | transform_up（起身） |
| `9` | joint_pd（趴下 / 保持默认姿态） |
| `8` | transform_down（折叠） |
| `6` | idle（软） |
| `w` / `s` | 前进 / 后退 |
| `a` / `d` | 左转 / 右转 |
| `r` | 速度归零 |

**流程**：启动后机器人**趴在地上**（`joint_pd` 保持趴姿，与真机上电状态一致）→ 按 **`7`** 起身 → 按 **`0`** 切策略 → 按 **`w`** 走。

## 部署新策略

训练端导出 ONNX 后：

```bash
# 1) 拷进本仓库
cp ~/dog3lab/logs/instinct_rl/dog3_locomotion_flat/<run>/exported/actor.onnx \
   controller/rl_controller/config/dog3/dog3_flat_01.onnx

# 2) 同步到 install 并重编
cp controller/rl_controller/config/dog3/* install/rl_controller/share/rl_controller/config/dog3/
colcon build --packages-select rl_controller
source install/setup.bash
# 重启仿真
```

策略配置在 `controller/rl_controller/config/dog3/controllers.yaml`，关键项：

| 项 | 值 | 必须与训练端一致 |
|---|---|---|
| `num_obs` | **51** | 45 本体 + 6 维步态相位 |
| `observations_name` | `[... "last_actions", "phases"]` | 顺序与训练端 obs 一致 |
| `gait_period` | **1.0 s** | 训练端 `GAIT_CYCLE_TIME` |
| `joint_kp` / `joint_kd` | **40 / 1.0** | 训练端执行器增益 |
| `action_scales` | hip 0.125、thigh/calf 0.25 | 训练端动作缩放 |
| `default_joint_angles` | 全 0 | 关节零位 = 站姿 |

**步态相位**：`rl_controller` 的 `phases` 观测量按 `φ = 2πt/gait_period` 计算，输出
`[sin φ, cos φ, sin(φ/2), cos(φ/2), sin(φ/4), cos(φ/4)]`，与训练端 `gait_phase` 逐位对应，
因此部署时无需改动 C++。

## 模型约定（与训练端共享）

URDF 的关节 origin 已整体平移，使**关节零位 = 默认站姿**（thigh +0.7、calf −1.44 的偏移
写进 origin 的 rpy），关节限位同步平移；因此站立时关节角读到 0。
`dog3lab` 与 `dog3_sim2sim` 两侧的 URDF 保持完全一致。

Gazebo 侧额外设置：link `selfCollide=false`（与 Isaac 的 `enabled_self_collisions=False` 对齐）。

## 数据记录与分析

```bash
# 录包
ros2 bag record -o /tmp/dog3_run /joint_states /imu_sensor_broadcaster/imu /model_states

# 转 CSV（PlotJuggler 的 rosbag2 加载器对这些包不稳定，用 CSV 更可靠）
source /opt/ros/humble/setup.bash
python3 tools/bag2csv.py /tmp/dog3_run
# → /tmp/dog3_run_csv/{joint_states,imu,model_pose}.csv

# PlotJuggler：File → Load data → 选 CSV
ros2 run plotjuggler plotjuggler
```
