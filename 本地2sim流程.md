# dog3 本地 sim2sim（MuJoCo）流程

> 在 `~/dog3_sim2sim` 里用 MuJoCo 跑 dog3 的 InstinctLab 策略（Flat 行走）。
> 参考 `~/work/ddt_ros2_control`（ROS2 Humble + mujoco_ros2_control + rl_controller）。
> 环境：Ubuntu 22.04、ROS2 Humble、系统 Python（miniconda 会干扰，见下）。

---

## 一、编译

```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin        # 绕开 miniconda python
export CPLUS_INCLUDE_PATH=$PWD/onnxruntime-linux-x64-1.10.0/include
export LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.10.0/lib
export CMAKE_PREFIX_PATH=$PWD/third_party/mujoco:$CMAKE_PREFIX_PATH   # vendored MuJoCo 3.3.0
export LD_LIBRARY_PATH=$PWD/third_party/mujoco/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash

colcon build --packages-up-to dog3_description mujoco_bridge rl_controller keyboard_controller
```

`third_party/mujoco/` 是预编译的 MuJoCo（含 `lib/libmujoco.so` + `lib/cmake/mujoco/mujocoConfig.cmake`），
所以 `mujoco_ros2_control` 的 `<depend>mujoco</depend>` 已注释掉，靠 `CMAKE_PREFIX_PATH` 找到。

---

## 二、启动仿真

**终端 1**
```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:$PATH
export ROS_DOMAIN_ID=1
export LD_LIBRARY_PATH=$PWD/third_party/mujoco/lib:$PWD/onnxruntime-linux-x64-1.10.0/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rl_controller sim_mujoco.launch.py robot:=dog3
```

**终端 2 — 键盘**
```bash
cd ~/dog3_sim2sim
export PATH=/usr/bin:$PATH
export ROS_DOMAIN_ID=1          # 必须与终端 1 相同
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run keyboard_controller keyboard_controller_node
```

## 三、键位

| 键 | 动作 |
|---|---|
| `9` | joint_pd（保持站姿） |
| `0` | dog3_flat（RL 行走策略 = `rl_policy_names[0]`） |
| `7` | transform_up（PD 起身） |
| `8` | transform_down（趴下） |
| `6` | idle（软，kp=0 会瘫倒） |
| `w`/`s` | 前进/后退 |
| `a`/`d` | 左转/右转 |
| `r` | 速度归零 |

**标准流程**：启动 → 等 `FSM state now: joint_pd` → 按 `0` 切 dog3_flat → `w` 走。

> **本仓库改动**：FSM 初始状态由 `idle` 改成 `joint_pd`（`FSM.cpp::initialize`）。
> 原因：`idle` 是纯阻尼（kp=0），控制器启动的那几秒机器人会瘫倒，而 kp=20 拉不起来。
> 且 `rc_data->fsm_name_` 默认是 `idle`，不改就会被立刻拉回去，所以一并钉成 `joint_pd`。

---

## 四、导入新策略（训练 run → sim）

```bash
# 1) 训练端导出 ONNX（归一化已烘焙进图）
cd ~/dog3lab
./run.sh play --task=Instinct-Locomotion-Flat-Dog3-Play-v0 --headless \
  --load_run <run_id> --checkpoint model_XXXX.pt --exportonnx
# 产物：logs/instinct_rl/dog3_locomotion_flat/<run>/exported/actor.onnx

# 2) 拷进本仓库（覆盖当前的占位 onnx）
cp <run>/exported/actor.onnx controller/rl_controller/config/dog3/dog3_flat_01.onnx

# 3) 同步 install + 重编
cp controller/rl_controller/config/dog3/* install/rl_controller/share/rl_controller/config/dog3/
colcon build --packages-select rl_controller
source install/setup.bash
```

> ⚠️ **当前 `config/dog3/dog3_flat_01.onnx` 是占位模型**（MatMul 全零，输入 1×45 输出 1×12），
> 只用于打通链路：输出恒为 0 → 目标关节角 = 默认站姿。跑真策略前必须用上面的步骤覆盖它。

---

## 五、dog3 关键对齐参数（改训练端时必须同步改 `config/dog3/controllers.yaml`）

| 项 | 值 |
|---|---|
| obs（45，顺序固定） | ang_vel×0.25 → gravity×1 → cmd×[2,2,0.25] → dof_pos_rel×1 → dof_vel×0.05 → last_action×1 |
| action（12） | hip 0.125 / thigh·calf 0.25，`use_default_offset=True` |
| 关节顺序 | FL,FR,RL,RR × hip,thigh,calf |
| 默认站姿 | hip 0 / thigh 0.7 / calf −1.44 |
| kp / kd | **20 / 0.5（占位值，未标定）** |
| time_interval | **0.02**（50 Hz；MJCF timestep 0.005 → 控制器 200 Hz） |
| torque_limit | 20 N·m |

**IMU 命名约定**（`mujoco_system.cpp`）：ros2_control 的 `<sensor name>` 必须**含 `_imu`**，
去掉最后的 `_xxx` 前缀后去找 MJCF 里 `<前缀>_quat` / `<前缀>_gyro` / `<前缀>_accel`。
例：`trunk_imu` → `trunk_quat`/`trunk_gyro`/`trunk_accel`（MJCF 里挂在 site `trunk_imu` 上）。

---

## 六、常见问题

| 症状 | 原因 | 解决 |
|---|---|---|
| `find_package(mujoco)` 失败 | 没设 CMAKE_PREFIX_PATH | 见步骤一 |
| `libmujoco.so` / `libonnxruntime.so` 找不到（运行） | LD_LIBRARY_PATH 没设 | 见步骤二 |
| `Package 'rl_controller' not found` | 没 source install / 没编译 | 编译 + source |
| `Failed to find IMU sensor ... sensor name: base_imu` | ros2_control sensor 名与 MJCF 不匹配 | 用 `*_imu` 名 + MJCF `<前缀>_quat/_gyro/_accel` |
| `State interface ... trunk_imu/orientation.x does not exist` | xacro 里的 sensor 名和 yaml `sensors:` 不一致 | 两边都用 `trunk_imu` |
| `Material [orange] has malformed color rgba` | URDF 颜色用了 0–255 | xacro 里改成 0–1（`robot.xacro` 的 orange） |
| MuJoCo `inertia must satisfy A+B>=C` | CAD 惯量非物理 | 转换前对 URDF 做惯量平衡（或在 MJCF 加 `<compiler balanceinertia="true"/>`） |
| MuJoCo `STL faces > 200000` | 网格太密 | 抽稀 STL（base_link 已 70 万→3 万面） |
| **机器人站不住 / 策略一进去就倒** | MuJoCo 把碰撞 mesh 变成**凸包**，弯曲的大腿/小腿凸包远大于真实外形，折叠时**大腿凸包先着地**，逼出蜷缩姿态 | 腿/躯干碰撞**换成图元**：大腿/小腿 capsule、髋/躯干 box、足端 sphere（见 `robot.xml`） |
| `urdf2mjcf` 报 `not in the subpath of urdf_dir` | mesh 在 urdf 目录外 | 把 urdf+meshes 放到同一临时目录再转 |
| 躯干/腿看着**透明或闪烁** | 同一 mesh 既作 visual(group 1) 又作 collision(group 0)，两个重合面 **z-fighting** | collision geom 放进 **group 3**（默认不渲染）：`robot.xml` 的 `<default><geom ... group="3"/>`，visual 仍显式 `group="1"` |
| **地面/环境消失** | `scene.xml` 的 floor 是 include 之后定义的，**继承了 robot.xml 里 default 的 `group=3`** | 给 floor 显式写 `group="0"` |
| 环境**黑白** | urdf2mjcf 生成的地面是 builtin checker，`rgb1=".0 .0 .0" rgb2=".8 .8 .8"` | 改 `texplane` 的 `rgb1/rgb2` 为彩色（本仓库已改成蓝灰） |

---

## 七、与训练端的已知差异（待 M4 对齐）

- **站高不同**：MuJoCo 里 kp/kd=20/0.5 的平衡站高 ≈0.256 m，训练端 ≈0.30 m；髋/大腿角也比默认更蜷（thigh≈0.35–0.4 vs 0.7）。
  需调 `armature / frictionloss / geom friction / solref,solimp` 或增益，或在两端口径统一后重训。
- 碰撞几何已从 mesh 凸包换成图元（大腿/小腿 capsule、髋/躯干 box、足端 sphere）；图元尺寸是**按 mesh bbox 估的**，未精调，接触参数（friction/solref/solimp）仍是默认，待与 Isaac 对齐。
- **速度跟踪仍偏差**（MuJoCo 里跟不动命令速度），站得住、能大致朝命令方向移动，但精度不够——下一步继续对齐动力学。- kp/kd 为占位值：sim2sim 与训练必须同值，标定要两边同步改。
