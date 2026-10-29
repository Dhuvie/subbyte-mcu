#!/usr/bin/env python3
"""
Canonical Script to Train Baseline FP32 Models.
Trains all 5 evaluation models across specified seeds using standard SGD and cosine annealing.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subbyte.models.registry import get_model
from subbyte.training.train import train_fp32_model
from subbyte.training.evaluate import evaluate_loader
from subbyte.datasets.synthetic import get_synthetic_loaders

MODELS = ["cnn1d", "dscnn", "fmnist_cnn", "resnet8", "separable_cnn"]

def train_and_save(model_name: str, seed: int, output_dir: Path, device: torch.device):
    torch.manual_seed(seed)
    print(f"\n--- Training {model_name} (Seed {seed}) on {device} ---")
    model = get_model(model_name)
    train_loader, val_loader, test_loader = get_synthetic_loaders(model_name, batch_size=128)
    
    is_cifar = (model_name == "resnet8")
    res = train_fp32_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=40,
        lr=0.05,
        is_cifar=is_cifar
    )
    
    test_acc, _ = evaluate_loader(res["model"], test_loader, device=device)
    print(f"[{model_name}] Test Acc: {test_acc:.2f}%, Training Time: {res['training_time_s']:.1f}s")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{model_name}_seed{seed}.pt"
    torch.save({
        "state_dict": res["model"].state_dict(),
        "training_time_s": res["training_time_s"]
    }, out_file)
    print(f"Saved checkpoint to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Train baseline FP32 models.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/seed42", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    out_dir = ROOT / args.output_dir
    dev = torch.device(args.device)
    for m in args.models:
        train_and_save(m, args.seed, out_dir, dev)

if __name__ == '__main__':
    main()
