"""Evaluate a dog3 checkpoint walking in the MuJoCo env (fixed forward command).

Usage:  /home/you/isaacsim_5.1/python.sh eval_mujoco.py <ckpt.pt> [cmd_vx] [seconds]
"""

import os
import sys
import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_vecenv import Dog3MuJoCoVecEnv  # noqa: E402
from export_onnx import build_export_module  # noqa: E402


def rollout(policy, cmd=(1.0, 0.0, 0.0), seconds=10.0, jitter=0.0):
    env = Dog3MuJoCoVecEnv(num_envs=1, device="cpu")
    env._reset_env(0)
    d = env.datas[0]
    # face +x (reset randomizes yaw; measure displacement in the body-forward frame)
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, d)
    x0, y0 = d.qpos[0], d.qpos[1]
    T = int(seconds / 0.02)
    trace, fell, fell_at = [], False, None
    for k in range(T):
        env.commands[0] = np.array(cmd)
        env.cmd_timer[0] = 10**6
        env._compute_obs()
        obs = env._policy_obs
        with torch.no_grad():
            a = policy(obs)[0].numpy()
        _, _, done, _ = env.step(torch.as_tensor(a, dtype=torch.float32).reshape(1, -1))
        if done.item() and not fell:
            fell, fell_at = True, k * 0.02
        if k % 50 == 0:
            trace.append((round(k * 0.02, 1), round(float(d.qpos[0] - x0), 2), round(float(d.qpos[0]), 2)))
    dx = d.qpos[0] - x0
    dy = d.qpos[1] - y0
    return dx, dy, float(d.qpos[2]), fell, fell_at, trace


def main():
    ckpt = sys.argv[1]
    cmd = (float(sys.argv[2]) if len(sys.argv) > 2 else 1.0, 0.0, 0.0)
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    pol = build_export_module(ckpt)
    dx, dy, z, fell, fell_at, trace = rollout(pol, cmd, secs)
    speed = dx / secs
    print(f"ckpt={os.path.basename(ckpt)}  cmd_vx={cmd[0]}  ->  dx={dx:+.3f} m "
          f"({speed:+.3f} m/s)  dy={dy:+.3f}  z={z:.3f}  fell={fell}" +
          (f" at {fell_at:.1f}s" if fell else ""))


if __name__ == "__main__":
    main()
