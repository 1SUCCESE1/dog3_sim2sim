"""Export an instinct_rl dog3 checkpoint to the ONNX format the ROS2 sim2sim
controller expects: input "input" [1,45], output "output" [1,12], with the policy
normalizer baked in as Sub -> Div (same convention as dog3lab play --exportonnx).

Usage:  /home/you/isaacsim_5.1/python.sh export_onnx.py <ckpt.pt> <out.onnx>
"""

import os
import sys
import torch
import torch.nn as nn

EPS = 1e-2


def build_export_module(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    msd = sd["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(45, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    actor_sd = {k.replace("actor.", ""): v for k, v in msd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd)
    actor.eval()
    nz = sd["policy_normalizer_state_dict"]

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = actor
            self.register_buffer("mean", nz["_mean"].float())
            self.register_buffer("std", nz["_std"].float())

        def forward(self, x):
            return self.actor((x - self.mean) / (self.std + EPS))

    return Wrapped().eval()


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2]
    mod = build_export_module(ckpt)
    dummy = torch.zeros(1, 45)
    torch.onnx.export(mod, dummy, out, input_names=["input"], output_names=["output"],
                      opset_version=12, dynamic_axes=None)
    print("wrote", out)

    # sanity check vs the graph ops
    import onnx
    m = onnx.load(out)
    print("ops:", [n.op_type for n in m.graph.node[:5]],
          "in:", [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim],
          "out:", [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])


if __name__ == "__main__":
    main()
