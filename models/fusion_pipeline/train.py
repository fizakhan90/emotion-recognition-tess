"""Multimodal (speech + text) emotion recognition: train.
  python models/fusion_pipeline/train.py                      # all three fusion types
  python models/fusion_pipeline/train.py --fusion gate
Speech = frozen WavLM frames, text = frozen BERT tokens; both are cached by the earlier pipelines."""
import argparse, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from common.data import (find_feat_dir, load_speech, load_text, _masked_stats, GPUFusionLoader, rows_for, SPLITS)
from common.layers import FusionModel
from common.engine import set_seed, train


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", nargs="+", default=["concat", "gate", "gmu"])
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--train_text", choices=["normal", "zeroed"], default="normal",
                   help="zeroed = control run: identical model, but text blanked during training")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_all(feat_dir, device):
    meta, S, S_len = load_speech(feat_dir, "wavlm")
    _, T, T_len = load_text(feat_dir)
    g = lambda a, dt=None: torch.as_tensor(a, device=device) if dt is None else torch.as_tensor(a, dtype=dt, device=device)
    return meta, S, S_len, dict(S=g(S), S_len=g(S_len, torch.long), T=g(T), T_len=g(T_len, torch.long),
                                labels=g(meta.label.values, torch.long))


def make_loader(meta, S, S_len, gpu, split, part, batch=32, shuffle=False, text_mode="normal"):
    m, s = _masked_stats(S, S_len, rows_for(meta, split, "train"))          # speech stats from training rows only
    dev = gpu["S"].device
    return GPUFusionLoader(gpu["S"], gpu["S_len"], gpu["T"], gpu["T_len"], gpu["labels"],
                           rows_for(meta, split, part), torch.as_tensor(m, device=dev), torch.as_tensor(s, device=dev),
                           batch=batch, shuffle=shuffle, text_mode=text_mode)


def main():
    a = get_args()
    a.out = a.out or os.path.join(ROOT, "Results", "fusion" if a.train_text == "normal" else "fusion_control")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = find_feat_dir(a.feat_dir)
    meta, S, S_len, gpu = load_all(feat_dir, device)
    print(f"speech {tuple(S.shape)} text {tuple(gpu['T'].shape)} on {device}")
    for split in a.splits:
        tr = make_loader(meta, S, S_len, gpu, split, "train", a.batch, shuffle=True, text_mode=a.train_text)
        va = make_loader(meta, S, S_len, gpu, split, "val", 128, text_mode=a.train_text)
        for fusion in a.fusion:
            for seed in a.seeds:
                set_seed(seed)
                d = os.path.join(a.out, fusion, split, f"seed{seed}")
                print(f"\n=== {fusion} {split} seed {seed}")
                model = FusionModel(fusion, S.shape[2:], text_dim=gpu["T"].shape[-1])
                train(model, tr, va, device, d, epochs=a.epochs, lr=a.lr)
                json.dump(dict(vars(a), fusion=fusion, split=split, seed=seed), open(f"{d}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
