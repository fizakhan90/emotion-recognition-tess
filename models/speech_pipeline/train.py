"""Speech-only emotion recognition: train.

Example:
  python models/speech_pipeline/train.py --features wavlm
  python models/speech_pipeline/train.py --features logmel --norm speaker      # ablation
Trains one model per (split, seed); best checkpoint chosen on the validation set only.
"""
import argparse, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from common.data import find_feat_dir, load_speech, gpu_speech_loaders, SPLITS
from common.layers import SpeechModel
from common.engine import set_seed, train


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["logmel", "wavlm"], required=True)
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--norm", choices=["global", "speaker"], default="global")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "speech"))
    return p.parse_args()


def run_dir(out, features, norm, split, seed):
    return os.path.join(out, f"{features}_{norm}", split, f"seed{seed}")


def main():
    a = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = find_feat_dir(a.feat_dir)
    meta, X, lengths = load_speech(feat_dir, a.features)
    lr = a.lr or (1e-3 if a.features == "logmel" else 5e-4)
    print(f"features={a.features} {X.shape} norm={a.norm} device={device}")

    for split in a.splits:
        loaders = gpu_speech_loaders(X, lengths, meta, split, a.norm, device, ["train", "val"], a.batch)
        for seed in a.seeds:
            set_seed(seed)
            d = run_dir(a.out, a.features, a.norm, split, seed)
            print(f"\n=== {split} seed {seed} -> {d}")
            model = SpeechModel(a.features, X.shape[2:])
            train(model, loaders["train"], loaders["val"], device, d, epochs=a.epochs, lr=lr)
            json.dump(dict(vars(a), lr=lr, split=split, seed=seed, feat_shape=list(X.shape[2:]),
                           feat_dir=feat_dir), open(f"{d}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
