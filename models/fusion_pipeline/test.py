"""Multimodal emotion recognition: test.   python models/fusion_pipeline/test.py
For every run: normal test metrics + learned speech weight + two robustness checks
(text shuffled between clips, text zeroed). If accuracy does not move, the model ignores text."""
import argparse, glob, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from common.data import find_feat_dir, EMOTIONS
from common.layers import FusionModel
from common.engine import predict, save_test_outputs, metrics
from models.speech_pipeline.test import plot_confusion
from models.fusion_pipeline.train import load_all, make_loader


@torch.no_grad()
def mean_speech_weight(model, loader, device):
    if model.fusion == "concat":
        return None
    model.eval(); ws = []
    for xs, ls, xt, lt, y, r in loader:
        _, g = model.embed(xs, ls, xt, lt)
        ws.append(g.float().mean(-1).expand(len(y)).cpu() if g.dim() > 0 else g.expand(len(y)).cpu())
    w = torch.cat(ws)
    return dict(speech_weight_mean=w.mean().item(), speech_weight_std=w.std().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "fusion"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, S, S_len, gpu = load_all(find_feat_dir(a.feat_dir), device)
    rows = []
    for fdir in sorted(glob.glob(f"{a.out}/*/")):
        fusion = os.path.basename(fdir.rstrip("/"))
        for split_dir in sorted(glob.glob(f"{fdir}split_*")):
            split = os.path.basename(split_dir)
            loaders = {mode: make_loader(meta, S, S_len, gpu, split, "test", 128, text_mode=mode)
                       for mode in ["normal", "shuffled", "zeroed"]}
            cms = []
            for d in sorted(glob.glob(f"{split_dir}/seed*")):
                model = FusionModel(fusion, S.shape[2:], text_dim=gpu["T"].shape[-1]).to(device)
                model.load_state_dict(torch.load(f"{d}/best.pt", map_location=device))
                extra = mean_speech_weight(model, loaders["normal"], device) or {}
                for mode in ["shuffled", "zeroed"]:
                    o = predict(model, loaders[mode], device)
                    extra[f"acc_text_{mode}"] = metrics(o["y"], o["pred"])["acc"]
                res = save_test_outputs(predict(model, loaders["normal"], device, with_embeddings=True),
                                        meta, d, EMOTIONS, extra)
                cms.append(res["confusion"])
                rows.append(dict(model=f"fusion_{fusion}", split=split, seed=os.path.basename(d),
                                 acc=res["acc"], uar=res["uar"], macro_f1=res["macro_f1"],
                                 speech_weight=extra.get("speech_weight_mean", np.nan),
                                 acc_text_shuffled=extra["acc_text_shuffled"], acc_text_zeroed=extra["acc_text_zeroed"]))
                print(fusion, split, os.path.basename(d), {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
            plot_confusion(np.sum(cms, 0), f"fusion {fusion} - {split}, summed over seeds", f"{fdir}confusion_{split}.png")
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/all_runs.csv", index=False)
    cols = ["acc", "uar", "macro_f1", "speech_weight", "acc_text_shuffled", "acc_text_zeroed"]
    summ = df.groupby(["model", "split"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{a.out}/summary_fusion.csv")
    pd.set_option("display.width", 200)
    print("\n", summ)


if __name__ == "__main__":
    main()
