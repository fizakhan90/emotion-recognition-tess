"""Text-only emotion recognition: train.  python models/text_pipeline/train.py
Frozen BERT token features -> BiLSTM + attention -> classifier. Best epoch chosen on validation only."""
import argparse, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from common.data import find_feat_dir, load_text, make_normaliser, SpeechDataset, rows_for, SPLITS
from common.layers import TextModel
from common.engine import set_seed, train


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "text"))
    return p.parse_args()


def main():
    a = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = find_feat_dir(a.feat_dir)
    meta, X, lengths = load_text(feat_dir)
    labels = meta.label.values
    print(f"text features {X.shape}, device={device}")
    for split in a.splits:
        norm = make_normaliser(X, lengths, meta, split, "none")      # BERT features used as-is (LayerNorm in model)
        loaders = {part: DataLoader(SpeechDataset(X, lengths, labels, rows_for(meta, split, part), norm),
                                    batch_size=a.batch, shuffle=(part == "train"))
                   for part in ["train", "val"]}
        for seed in a.seeds:
            set_seed(seed)
            d = os.path.join(a.out, "bert", split, f"seed{seed}")
            print(f"\n=== {split} seed {seed} -> {d}")
            train(TextModel(in_dim=X.shape[-1]), loaders["train"], loaders["val"], device, d,
                  epochs=a.epochs, lr=a.lr)
            json.dump(dict(vars(a), split=split, seed=seed), open(f"{d}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
