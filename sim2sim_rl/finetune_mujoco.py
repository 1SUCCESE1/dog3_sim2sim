"""Fine-tune the dog3 Flat policy inside MuJoCo (sim2sim-in-the-loop).

Reuses the same PPO (instinct_rl) and the Isaac-trained checkpoint, but with a
MuJoCo backend, so the resulting policy walks under the MuJoCo model used by the
ROS2 sim2sim stack.  Export back to ONNX with export_onnx.py.

Usage:  /home/you/isaacsim_5.1/python.sh finetune_mujoco.py [num_iterations]
"""

import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mujoco_vecenv import Dog3MuJoCoVecEnv  # noqa: E402
from instinct_rl.runners import OnPolicyRunner  # noqa: E402

RUN = "/home/you/dog3lab/logs/instinct_rl/dog3_locomotion_flat/20260910_154219"
CKPT = os.path.join(RUN, "model_6000.pt")
AGENT = os.path.join(RUN, "params", "agent.yaml")
LOG = "/home/you/dog3_sim2sim/sim2sim_rl/logs/finetune_mujoco"


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    from_scratch = len(sys.argv) > 2 and sys.argv[2] == "scratch"
    cfg = yaml.safe_load(open(AGENT))
    cfg["max_iterations"] = iters
    cfg["save_interval"] = max(50, min(iters // 4, 250))
    cfg["log_interval"] = 10
    cfg["experiment_name"] = "finetune_mujoco"
    cfg["algorithm"]["learning_rate"] = 1.0e-4 if not from_scratch else 1.0e-3
    cfg["algorithm"]["desired_kl"] = 0.01
    device = cfg.get("device", "cuda:0")

    env = Dog3MuJoCoVecEnv(num_envs=128, device=device)
    print(f"[finetune] num_envs={env.num_envs} num_obs={env.num_obs} "
          f"num_critic_obs={env.num_critic_obs} num_actions={env.num_actions} "
          f"from_scratch={from_scratch}")

    log_dir = LOG + ("_scratch_curr" if from_scratch else "")
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    if not from_scratch:
        print(f"[finetune] loading checkpoint {CKPT}")
        runner.load(CKPT)
    else:
        print("[finetune] training FROM SCRATCH (no warm start)")
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
    print("[finetune] done. checkpoints in", LOG)


if __name__ == "__main__":
    main()
