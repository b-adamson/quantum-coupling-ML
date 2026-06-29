"""
Export trained PyTorch model to ONNX for deployment.

Usage:
    python export.py --weights ../data/model.pt --out ../data/model.onnx --grid_size 64
"""

import argparse
import os
import tempfile

import onnx
import torch

from train import CouplingCNN


def export(weights_path: str, out_path: str, grid_size: int):
    model = CouplingCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, grid_size)

    # Export to a temp dir so external data files don't contaminate the output dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "model.onnx")
        torch.onnx.export(
            model,
            dummy,
            tmp_path,
            input_names=["signal"],
            output_names=["coupling"],
            dynamic_axes={"signal": {0: "batch_size"}},
            opset_version=17,
        )
        # Reload and re-save with all tensors inlined — no external data files
        proto = onnx.load(tmp_path)
        onnx.save_model(proto, out_path, save_as_external_data=False)

    print(f"Exported to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="../data/model.pt")
    parser.add_argument("--out", type=str, default="../data/model.onnx")
    parser.add_argument("--grid_size", type=int, default=64)
    args = parser.parse_args()

    export(args.weights, args.out, args.grid_size)
