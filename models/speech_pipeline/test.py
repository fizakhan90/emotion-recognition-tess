"""Speech-only emotion recognition: evaluate saved checkpoints on the held-out test set.

Example:
  python models/speech_pipeline/test.py --features wavlm
Writes per-run test_metrics.json / test_predictions.csv / test_embeddings.npz, plus
Results/speech/summary_<features>_<norm>.csv (mean ± std over seeds) and confusion-matrix plots.
"""
import argparse, glob, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.data import find_feat_dir, load_speech, gpu_speech_loaders, EMOTIONS
from common.layers import SpeechModel
from common.engine import predict, save_test_outputs


def plot_confusion(cm, title, path):
    cm = np.asarray(cm); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45); ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["logmel", "wavlm"], required=True)
    p.add_argument("--norm", choices=["global", "speaker"], default="global")
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "speech"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, X, lengths = load_speech(find_feat_dir(a.feat_dir), a.features)
    base = os.path.join(a.out, f"{a.features}_{a.norm}")
    rows = []
    for split_dir in sorted(glob.glob(f"{base}/split_*")):
        split = os.path.basename(split_dir)
        loader = gpu_speech_loaders(X, lengths, meta, split, a.norm, device, ["test"], 64)["test"]
        cms = []
        for d in sorted(glob.glob(f"{split_dir}/seed*")):
            model = SpeechModel(a.features, X.shape[2:]).to(device)
            model.load_state_dict(torch.load(f"{d}/best.pt", map_location=device))
            out = predict(model, loader, device, with_embeddings=True)
            extra = {}
            if a.features == "wavlm":
                extra["layer_weights"] = model.front.layer_weights().detach().cpu().tolist()
            res = save_test_outputs(out, meta, d, EMOTIONS, extra)
            cms.append(res["confusion"])
            rows.append(dict(model=f"speech_{a.features}", norm=a.norm, split=split,
                             seed=os.path.basename(d), acc=res["acc"], uar=res["uar"], macro_f1=res["macro_f1"]))
            print(split, os.path.basename(d), {k: round(res[k], 4) for k in ["acc", "uar", "macro_f1"]})
        if cms:
            plot_confusion(np.sum(cms, 0), f"speech {a.features} ({a.norm}) - {split}, summed over seeds",
                           f"{base}/confusion_{split}.png")
    df = pd.DataFrame(rows)
    df.to_csv(f"{base}/all_runs.csv", index=False)
    summ = df.groupby(["model", "norm", "split"])[["acc", "uar", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{a.out}/summary_{a.features}_{a.norm}.csv")
    print("\n", summ)


if __name__ == "__main__":
    main()
