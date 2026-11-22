"""Text-only emotion recognition: test.  python models/text_pipeline/test.py
Also reports a TF-IDF + logistic-regression sanity baseline and the 95% chance interval."""
import argparse, glob, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common.data import find_feat_dir, load_text, make_normaliser, SpeechDataset, rows_for, EMOTIONS, clean_text
from common.layers import TextModel
from common.engine import predict, save_test_outputs, metrics
from models.speech_pipeline.test import plot_confusion


def chance_interval(n, k=7):
    p = 1 / k; half = 1.96 * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "text"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, X, lengths = load_text(find_feat_dir(a.feat_dir))
    base = os.path.join(a.out, "bert")
    rows = []
    for split_dir in sorted(glob.glob(f"{base}/split_*")):
        split = os.path.basename(split_dir)
        te = rows_for(meta, split, "test")
        loader = DataLoader(SpeechDataset(X, lengths, meta.label.values, te,
                                          make_normaliser(X, lengths, meta, split, "none")), batch_size=128)
        cms = []
        for d in sorted(glob.glob(f"{split_dir}/seed*")):
            model = TextModel(in_dim=X.shape[-1]).to(device)
            model.load_state_dict(torch.load(f"{d}/best.pt", map_location=device))
            res = save_test_outputs(predict(model, loader, device, with_embeddings=True), meta, d, EMOTIONS)
            cms.append(res["confusion"])
            rows.append(dict(model="text_bert", split=split, seed=os.path.basename(d),
                             acc=res["acc"], uar=res["uar"], macro_f1=res["macro_f1"]))
            print(split, os.path.basename(d), {k: round(res[k], 4) for k in ["acc", "uar", "macro_f1"]})
        plot_confusion(np.sum(cms, 0), f"text BERT - {split}, summed over seeds", f"{base}/confusion_{split}.png")
        # sanity baseline: bag-of-words logistic regression on the same split
        tr = rows_for(meta, split, "train")
        vec = TfidfVectorizer().fit(meta.text.iloc[tr].map(clean_text))
        clf = LogisticRegression(max_iter=2000).fit(vec.transform(meta.text.iloc[tr].map(clean_text)), meta.label.values[tr])
        m = metrics(meta.label.values[te], clf.predict(vec.transform(meta.text.iloc[te].map(clean_text))))
        rows.append(dict(model="text_tfidf_logreg", split=split, seed="-", **m))
        lo, hi = chance_interval(len(te))
        print(f"{split}: tfidf+LR acc={m['acc']:.4f} | 95% chance interval for n={len(te)}: [{lo:.3f}, {hi:.3f}]")
    df = pd.DataFrame(rows)
    df.to_csv(f"{base}/all_runs.csv", index=False)
    summ = df.groupby(["model", "split"])[["acc", "uar", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(f"{a.out}/summary_text.csv")
    print("\n", summ)


if __name__ == "__main__":
    main()
