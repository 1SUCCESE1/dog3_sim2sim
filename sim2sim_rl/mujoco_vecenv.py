"""MuJoCo vectorized env for dog3, matching the Isaac Lab (instinctlab) Flat task.

Implements the same observation layout / action mapping / command sampling / reward
terms as dog3lab's Dog3FlatEnvCfg, so an InstinctLab-trained policy can be
fine-tuned here (sim2sim-in-the-loop) and exported back for the ROS2 sim.

Interface matches instinct_rl's VecEnv (see instinctlab vecenv_wrapper.py).
"""

from __future__ import annotations

import numpy as np
import torch
import mujoco

MJCF = "/home/you/dog3_sim2sim/urdfs/dog3_description/mujoco/scene.xml"

# joint order MUST match the training action order (ActionsCfg)
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT_ANGLES = np.array([0.0, 0.7, -1.44] * 4, dtype=np.float64)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float64)
KP, KD, TAU_LIM = 40.0, 1.0, 20.0   # stiffer than Isaac's 20/0.5: kp=20 sags ~0.27 rad on this 17 kg robot

PHYS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION      # 0.02 s -> 50 Hz
EPISODE_LEN_S = 20.0
MAX_EPISODE_LENGTH = int(EPISODE_LEN_S / CONTROL_DT)   # 1000
NUM_ACTIONS = 12
OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL, OBS_SCALE_DOF_VEL = 2.0, 0.25, 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25])
TARGET_HEIGHT = 0.30

# reward weights (subset of dog3lab RewardsCfg)
W_TRACK_LIN, W_TRACK_ANG = 1.5, 0.7
W_LIN_VEL_Z, W_ANG_VEL_XY, W_FLAT_ORI, W_BASE_HEIGHT = -2.0, -0.05, -2.0, -10.0
W_ACTION_RATE, W_DEFAULT_JOINT, W_JOINT_LIMITS = -0.02, -0.02, -5.0
W_FEET_AIR_TIME = 0.75
TRACK_STD = np.sqrt(0.25)
REWARD_NAMES = [
    "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
    "flat_orientation_l2", "base_height_l2", "action_rate_l2", "default_joint_l2",
    "joint_pos_limits", "feet_air_time",
]


class _Cfg:
    """Minimal cfg object exposed to the runner."""
    def __init__(self):
        self.is_finite_horizon = False
        self.get = lambda k, d=None: d


class Dog3MuJoCoVecEnv:
    def __init__(self, num_envs: int = 128, device: str = "cuda:0", seed: int = 0):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = _Cfg()
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.num_actions = NUM_ACTIONS
        self.num_rewards = 1   # single summed reward (training critic is single-output)
        self.num_obs = 45
        self.num_critic_obs = 48
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(MJCF)
        self.model.opt.timestep = PHYS_DT
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

        # joint / actuator addresses in training order
        self.qa, self.da, self.act = [], [], []
        for n in JOINT_NAMES:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.qa.append(self.model.jnt_qposadr[j])
            self.da.append(self.model.jnt_dofadr[j])
            self.act.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n))
        self.qa = np.array(self.qa); self.da = np.array(self.da); self.act = np.array(self.act)

        # joint limits (for the limits penalty)
        self.jnt_lo, self.jnt_up = [], []
        for n in JOINT_NAMES:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.jnt_lo.append(self.model.jnt_range[j][0]); self.jnt_up.append(self.model.jnt_range[j][1])
        self.jnt_lo = np.array(self.jnt_lo); self.jnt_up = np.array(self.jnt_up)

        gyro = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "trunk_gyro")
        self.gyro_adr = self.model.sensor_adr[gyro]

        # collision geoms attached to base_link (illegal-contact termination)
        base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.base_geoms = set(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == base_bid
            and self.model.geom_contype[g] > 0 and self.model.geom_conaffinity[g] > 0
        )
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # match Isaac (enabled_self_collisions=false): robot<->robot off, robot<->ground on.
        # robot collision geoms: contype=1 conaffinity=2 ; floor: contype=2 conaffinity=1.
        for g in range(self.model.ngeom):
            if g == floor_id:
                continue
            if self.model.geom_contype[g] > 0 and self.model.geom_conaffinity[g] > 0:
                self.model.geom_contype[g] = 1
                self.model.geom_conaffinity[g] = 2
        self.model.geom_contype[floor_id] = 2
        self.model.geom_conaffinity[floor_id] = 1
        # foot collision geoms, in FL,FR,RL,RR order (sphere on each calf)
        self.foot_geoms = []
        for b in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            gs = [g for g in range(self.model.ngeom)
                  if self.model.geom_bodyid[g] == bid
                  and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            self.foot_geoms.append(gs[0] if gs else -1)

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_actions = np.zeros((num_envs, NUM_ACTIONS))
        self.prev_actions = np.zeros((num_envs, NUM_ACTIONS))
        self.commands = np.zeros((num_envs, 3))
        self.cmd_range = 1.0
        self.total_steps = 0
        self.cmd_timer = np.zeros(num_envs, dtype=int)
        self.feet_air_time = np.zeros((num_envs, 4))
        self._policy_obs = torch.zeros(num_envs, self.num_obs, device=self.device)
        self._critic_obs = torch.zeros(num_envs, self.num_critic_obs, device=self.device)
        self._floor_id = floor_id

        for i in range(num_envs):
            self._reset_env(i, full=True)

    # ---------------------------------------------------------------- helpers
    def _reset_env(self, i: int, full: bool = True):
        d = self.datas[i]
        m = self.model
        d.qpos[:] = 0.0
        # free joint: qpos[0:3]=pos, qpos[3:7]=quat(w,x,y,z)
        d.qpos[3] = 1.0
        for k, a in enumerate(self.qa):
            d.qpos[a] = DEFAULT_ANGLES[k] * (1.0 + 0.1 * self.rng.standard_normal())
        d.qpos[0] = self.rng.uniform(-0.3, 0.3)
        d.qpos[1] = self.rng.uniform(-0.3, 0.3)
        d.qpos[2] = 0.33
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.last_actions[i] = 0.0
        self.prev_actions[i] = 0.0
        self.feet_air_time[i] = 0.0
        self._resample_command(i)

    def _resample_command(self, i: int):
        if self.rng.random() < 0.15:      # rel_standing_envs
            self.commands[i] = 0.0
        else:
            r = self.cmd_range
            self.commands[i] = [
                self.rng.uniform(-r, r),
                self.rng.uniform(-r, r),
                self.rng.uniform(-r, r),
            ]
        self.cmd_timer[i] = int(10.0 / CONTROL_DT)   # resample every 10 s

    def _update_cmd_range(self):
        """Command-range curriculum: start small (achievable) and ramp to full.

        Tracking reward is exp(-err^2/std^2); while the achievable speed is far
        below the command the gradient is nearly flat and policies settle into
        standing.  Ramping lets it learn to move first.
        """
        self.cmd_range = min(1.0, 0.20 + self.total_steps * 4.0e-7)

    def _sensors(self, i: int):
        d = self.datas[i]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[3:7]); R = R.reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        lin_b = R.T @ d.qvel[0:3]
        q = d.qpos[self.qa].copy()
        dq = d.qvel[self.da].copy()
        return R, grav, gyro, lin_b, q, dq

    # ---------------------------------------------------------------- VecEnv API
    def get_observations(self):
        self._compute_obs()
        return self._policy_obs, {"observations": {"policy": self._policy_obs, "critic": self._critic_obs}}

    def _compute_obs(self):
        for i in range(self.num_envs):
            _, grav, gyro, lin_b, q, dq = self._sensors(i)
            pos_rel = q - DEFAULT_ANGLES
            cmd = self.commands[i] * CMD_SCALE
            pol = np.concatenate([gyro * OBS_SCALE_ANG_VEL, grav, cmd,
                                  pos_rel, dq * OBS_SCALE_DOF_VEL, self.last_actions[i]])
            cri = np.concatenate([lin_b * OBS_SCALE_LIN_VEL, gyro * OBS_SCALE_ANG_VEL, grav, cmd,
                                  pos_rel, dq * OBS_SCALE_DOF_VEL, self.last_actions[i]])
            self._policy_obs[i] = torch.as_tensor(pol, dtype=torch.float32, device=self.device)
            self._critic_obs[i] = torch.as_tensor(cri, dtype=torch.float32, device=self.device)

    def reset(self):
        for i in range(self.num_envs):
            self._reset_env(i)
        self.episode_length_buf[:] = 0
        self._compute_obs()
        return self._policy_obs, {"observations": {"policy": self._policy_obs, "critic": self._critic_obs}}

    def step(self, actions: torch.Tensor):
        acts = actions.detach().cpu().numpy()
        rew = np.zeros((self.num_envs, self.num_rewards), dtype=np.float64)
        dones = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            d = self.datas[i]
            R, grav, gyro, lin_b, q0, dq0 = self._sensors(i)
            a = np.clip(acts[i], -1.0, 1.0)
            q_des = DEFAULT_ANGLES + a * ACTION_SCALE
            tau = np.clip(KP * (q_des - q0) - KD * dq0, -TAU_LIM, TAU_LIM)
            d.ctrl[self.act] = tau
            for _ in range(DECIMATION):
                mujoco.mj_step(self.model, d)

            R, grav, gyro, lin_b, q, dq = self._sensors(i)
            # reward terms
            v_xy, om_z = lin_b[:2], gyro[2]
            cmd = self.commands[i]
            r = np.zeros(len(REWARD_NAMES))
            r[0] = W_TRACK_LIN * np.exp(-np.sum((cmd[:2] - v_xy) ** 2) / TRACK_STD ** 2)
            r[1] = W_TRACK_ANG * np.exp(-((cmd[2] - om_z) ** 2) / TRACK_STD ** 2)
            r[2] = W_LIN_VEL_Z * lin_b[2] ** 2
            r[3] = W_ANG_VEL_XY * np.sum(gyro[:2] ** 2)
            r[4] = W_FLAT_ORI * np.sum(grav[:2] ** 2)
            r[5] = W_BASE_HEIGHT * (d.qpos[2] - TARGET_HEIGHT) ** 2
            r[6] = W_ACTION_RATE * np.sum((a - self.last_actions[i]) ** 2)
            r[7] = W_DEFAULT_JOINT * np.sum((q - DEFAULT_ANGLES) ** 2)
            margin = 0.05
            lo, up = self.jnt_lo + margin, self.jnt_up - margin
            r[8] = W_JOINT_LIMITS * (np.sum(np.maximum(0, q - up)) + np.sum(np.maximum(0, lo - q)))
            r[9] = 0.0
            # foot contacts -> feet_air_time reward
            foot_contact = [False] * 4
            base_hit = False
            for c in range(d.ncon):
                con = d.contact[c]
                g1, g2 = con.geom1, con.geom2
                if self._floor_id in (g1, g2):
                    other = g2 if g1 == self._floor_id else g1
                    if other in self.base_geoms:
                        base_hit = True
                    for f in range(4):
                        if other == self.foot_geoms[f]:
                            foot_contact[f] = True
            for f in range(4):
                if foot_contact[f]:
                    if self.feet_air_time[i, f] > 0.5:
                        r[9] += W_FEET_AIR_TIME * self.feet_air_time[i, f]
                    self.feet_air_time[i, f] = 0.0
                else:
                    self.feet_air_time[i, f] += CONTROL_DT
            rew[i, 0] = r.sum()

            # termination: base contact with floor, or too low
            base_hit = False
            for c in range(d.ncon):
                con = d.contact[c]
                g1, g2 = con.geom1, con.geom2
                if (g1 in self.base_geoms and g2 == self._floor_id) or \
                   (g2 in self.base_geoms and g1 == self._floor_id):
                    base_hit = True
                    break
            self.prev_actions[i] = self.last_actions[i]
            self.last_actions[i] = a
            self.cmd_timer[i] -= 1
            if self.cmd_timer[i] <= 0:
                self._resample_command(i)

            if base_hit or d.qpos[2] < 0.12:
                dones[i] = True
                self._reset_env(i)
                self.episode_length_buf[i] = -1   # becomes 0 after the += 1 below
            elif self.episode_length_buf[i].item() + 1 >= MAX_EPISODE_LENGTH:
                truncated[i] = True
                self._reset_env(i)
                self.episode_length_buf[i] = -1
        self.episode_length_buf += 1
        self.total_steps += self.num_envs
        self._update_cmd_range()
        self._compute_obs()
        rew_t = torch.as_tensor(rew.sum(axis=1, keepdims=True), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones | truncated, dtype=torch.long, device=self.device)
        extras = {"observations": {"policy": self._policy_obs, "critic": self._critic_obs},
                  "time_outs": torch.as_tensor(truncated, dtype=torch.bool, device=self.device)}
        return self._policy_obs, rew_t, done_t, extras

    def get_obs_format(self):
        return {
            "policy": {"base_ang_vel": (3,), "projected_gravity": (3,), "velocity_commands": (3,),
                       "joint_pos": (12,), "joint_vel": (12,), "actions": (12,)},
            "critic": {"base_lin_vel": (3,), "base_ang_vel": (3,), "projected_gravity": (3,),
                       "velocity_commands": (3,), "joint_pos": (12,), "joint_vel": (12,), "actions": (12,)},
        }

    def get_obs_segments(self, group_name: str = "policy"):
        return self.get_obs_format().get(group_name, {})

    def seed(self, seed: int = -1):
        self.rng = np.random.default_rng(seed if seed >= 0 else 0)
        return seed
