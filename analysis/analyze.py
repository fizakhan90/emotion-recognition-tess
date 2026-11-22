"""Analysis for the report. Run after the speech, text and fusion pipelines (and the control run).
  python analysis/analyze.py
Writes to Results/analysis/:
  main_results.csv / .md        one table with every model (mean ± std over seeds)
  per_class.csv                 recall + precision per emotion (WavLM speech, gate fusion, speaker splits)
  confusion_pairs.csv           most frequent true->pred confusions
  failure_cases.csv             clips misclassified by every seed (candidates for the 3-5 failure cases)
  tsne_<split>.png              t-SNE of temporal (speech), contextual (text) and fusion embeddings,
                                coloured by emotion and by speaker
  separability.csv              silhouette scores behind the t-SNE plots
  wavlm_layer_probe.png         per-layer linear probe on mean-pooled WavLM features
"""
import argparse, glob, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from common.data import (find_feat_dir, load_speech, load_text, make_normaliser, SpeechDataset, rows_for,
                         _masked_stats, GPUFusionLoader, gpu_speech_loaders, EMOTIONS)
from common.layers import SpeechModel, TextModel, FusionModel
from common.engine import predict

SPEAKER_SPLITS = ["split_OAF_to_YAF", "split_YAF_to_OAF"]
COLORS = ["#d62728", "#8c564b", "#9467bd", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]


# ------------------------------------------------------------------ 1. results table
def results_table(R, out):
    frames = []
    spec = [("speech/logmel_global/all_runs.csv", "Speech: log-mel CNN-BiLSTM-attn", "acc"),
            ("speech/logmel_speaker/all_runs.csv", "Speech: log-mel + per-speaker norm (ablation)", "acc"),
            ("speech/wavlm_global/all_runs.csv", "Speech: WavLM + BiLSTM-attn", "acc"),
            ("speech/wavlm_speaker/all_runs.csv", "Speech: WavLM + per-speaker norm (ablation)", "acc"),
            ("text/bert/all_runs.csv", None, "acc"),
            ("fusion/all_runs.csv", None, "acc"),
            ("fusion_control/all_runs.csv", "control", "acc_text_zeroed")]
    names = {"text_bert": "Text: BERT + BiLSTM-attn", "text_tfidf_logreg": "Text: TF-IDF + LogReg (sanity)",
             "fusion_concat": "Fusion: concat", "fusion_gate": "Fusion: scalar gate", "fusion_gmu": "Fusion: GMU"}
    for path, name, col in spec:
        f = f"{R}/{path}"
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if name == "control":
            df["model"] = df.model.map(lambda m: f"Control: {m.split('_')[1]} arch, NO text (speech-only)")
        elif name:
            df["model"] = name
        else:
            df["model"] = df.model.map(names)
        frames.append(df[["model", "split"]].assign(acc=df[col] * 100))
    df = pd.concat(frames)
    g = df.groupby(["model", "split"], sort=False).acc
    t = (g.mean().round(1).astype(str) + " ± " + g.std().fillna(0).round(1).astype(str)).unstack("split")
    t = t[[c for c in ["split_OAF_to_YAF", "split_YAF_to_OAF", "split_random"] if c in t.columns]]
    sp = df[df.split.isin(SPEAKER_SPLITS)].groupby("model", sort=False).acc.mean().round(1)
    t["speaker-split mean"] = sp
    t.to_csv(f"{out}/main_results.csv")
    open(f"{out}/main_results.md", "w").write(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())
    print(t.to_string())


# ------------------------------------------------------------------ 2. per-class + failures
def load_preds(R, sub):
    dfs = []
    for split in SPEAKER_SPLITS:
        for f in sorted(glob.glob(f"{R}/{sub}/{split}/seed*/test_predictions.csv")):
            dfs.append(pd.read_csv(f).assign(split=split, seed=os.path.basename(os.path.dirname(f))))
    return pd.concat(dfs) if dfs else None


def per_class(R, out):
    rows = []
    for sub, name in [("speech/wavlm_global", "speech_wavlm"), ("speech/logmel_global", "speech_logmel"),
                      ("fusion/gate", "fusion_gate")]:
        p = load_preds(R, sub)
        if p is None:
            continue
        for split, g in p.groupby("split"):
            pr, rc, f1, _ = precision_recall_fscore_support(g.emotion, g.pred, labels=EMOTIONS, zero_division=0)
            for e, a, b, c in zip(EMOTIONS, pr, rc, f1):
                rows.append(dict(model=name, split=split, emotion=e, precision=a, recall=b, f1=c))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/per_class.csv", index=False)
    piv = df.groupby(["model", "emotion"])[["recall", "precision", "f1"]].mean().round(3).unstack("model")
    print("\nPer-class (mean of both speaker splits):\n", piv.to_string())
    return piv


def failures(R, out, meta):
    p = load_preds(R, "speech/wavlm_global")
    if p is None:
        return
    pairs = p[p.emotion != p.pred].groupby(["emotion", "pred"]).size().sort_values(ascending=False)
    (pairs / len(p) * 100).round(2).rename("pct_of_all_test_preds").to_csv(f"{out}/confusion_pairs.csv")
    print("\nTop confusions (WavLM, speaker splits):\n", pairs.head(8).to_string())
    agg = p.groupby(["split", "fname"]).agg(emotion=("emotion", "first"), word=("word", "first"),
                                            speaker=("speaker", "first"),
                                            n_wrong=("pred", lambda s: 0), preds=("pred", list),
                                            conf=("confidence", "mean"))
    wrong = p.assign(w=p.emotion != p.pred).groupby(["split", "fname"]).w.sum()
    agg["n_wrong"] = wrong
    n_seeds = p.seed.nunique()
    fails = agg[agg.n_wrong == n_seeds].sort_values("conf", ascending=False).reset_index()
    fails["path"] = fails.fname.map(dict(zip(meta.fname, meta.path)))
    fails.to_csv(f"{out}/failure_cases.csv", index=False)
    print(f"\n{len(fails)} clips wrong in all {n_seeds} seeds. Most confident ones:")
    print(fails[["split", "fname", "emotion", "preds", "conf"]].head(8).to_string(index=False))


# ------------------------------------------------------------------ 3. embeddings + t-SNE
@torch.no_grad()
def embed_all(R, fd, split, device):
    """Embeddings of ALL 2800 clips from the seed0 model of each pipeline trained on `split`."""
    embs = {}
    meta, S, S_len = load_speech(fd, "wavlm")
    _, T, T_len = load_text(fd)
    all_rows = np.arange(len(meta))
    ck = f"{R}/speech/wavlm_global/{split}/seed0/best.pt"
    if os.path.exists(ck):
        m = SpeechModel("wavlm", S.shape[2:]).to(device); m.load_state_dict(torch.load(ck, map_location=device))
        dl = gpu_speech_loaders(S, S_len, meta, split, "global", device, ["all"], 64)["all"]
        embs["Temporal block (speech, WavLM)"] = predict(m, dl, device, True)["emb"]
    ck = f"{R}/text/bert/{split}/seed0/best.pt"
    if os.path.exists(ck):
        m = TextModel(in_dim=T.shape[-1]).to(device); m.load_state_dict(torch.load(ck, map_location=device))
        dl = DataLoader(SpeechDataset(T, T_len, meta.label.values, all_rows,
                                      make_normaliser(T, T_len, meta, split, "none")), batch_size=256)
        embs["Contextual block (text, BERT)"] = predict(m, dl, device, True)["emb"]
    ck = f"{R}/fusion/gate/{split}/seed0/best.pt"
    if os.path.exists(ck):
        m = FusionModel("gate", S.shape[2:], text_dim=T.shape[-1]).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        mean, std = _masked_stats(S, S_len, rows_for(meta, split, "train"))
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
        dl = GPUFusionLoader(t(S), t(S_len, torch.long), t(T), t(T_len, torch.long), t(meta.label.values, torch.long),
                             all_rows, t(mean), t(std), batch=64)
        embs["Fusion block (scalar gate)"] = predict(m, dl, device, True)["emb"]
        del dl
    del S
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta, embs


def tsne_plots(R, fd, out, device, splits):
    rows = []
    for split in splits:
        meta, embs = embed_all(R, fd, split, device)
        if not embs:
            continue
        test = (meta[split] == "test").values
        tr_spk = meta.speaker[meta[split] == "train"].iloc[0]
        fig, axes = plt.subplots(len(embs), 2, figsize=(11, 4.6 * len(embs)), squeeze=False)
        for i, (name, E) in enumerate(embs.items()):
            Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
                StandardScaler().fit_transform(E))
            for e_i, e in enumerate(EMOTIONS):
                m = (meta.emotion == e).values
                axes[i, 0].scatter(Z[m & ~test, 0], Z[m & ~test, 1], s=4, c=COLORS[e_i], alpha=0.35, marker="o")
                axes[i, 0].scatter(Z[m & test, 0], Z[m & test, 1], s=6, c=COLORS[e_i], alpha=0.8, marker="x", label=e)
            for spk, c in [("OAF", "#1f77b4"), ("YAF", "#d62728")]:
                m = (meta.speaker == spk).values
                axes[i, 1].scatter(Z[m, 0], Z[m, 1], s=4, c=c, alpha=0.4,
                                   label=f"{spk} ({'train' if spk == tr_spk else 'test'})")
            axes[i, 0].set_title(f"{name}\ncoloured by emotion (o = train speaker, x = test speaker)", fontsize=9)
            axes[i, 1].set_title(f"{name}\ncoloured by speaker", fontsize=9)
            for ax in axes[i]:
                ax.set_xticks([]); ax.set_yticks([])
            axes[i, 0].legend(markerscale=3, fontsize=7, loc="best"); axes[i, 1].legend(markerscale=3, fontsize=8)
            sil = lambda X, y: silhouette_score(X, y) if len(set(y)) > 1 else np.nan
            rows.append(dict(split=split, block=name,
                             silhouette_emotion_test_speaker=sil(E[test], meta.emotion[test]),
                             silhouette_emotion_train_speaker=sil(E[~test], meta.emotion[~test]),
                             silhouette_speaker_all=sil(E, meta.speaker)))
        fig.suptitle(f"t-SNE of learned representations - models trained on {tr_spk} ({split})", fontsize=11)
        fig.tight_layout(); fig.savefig(f"{out}/tsne_{split}.png", dpi=130); plt.close(fig)
        print("saved", f"{out}/tsne_{split}.png")
    df = pd.DataFrame(rows).round(3)
    df.to_csv(f"{out}/separability.csv", index=False)
    print("\nSilhouette scores (higher = better separated; ~0 = no structure):\n", df.to_string(index=False))


# ------------------------------------------------------------------ 4. WavLM layer probe
def layer_probe(fd, out):
    f = f"{fd}/wavlm_layermeans.npy"
    if not os.path.exists(f):
        return
    meta = pd.read_csv(f"{fd}/meta.csv"); M = np.load(f)
    res = {}
    for split in SPEAKER_SPLITS:
        tr, te = rows_for(meta, split, "train"), rows_for(meta, split, "test")
        accs = []
        for L in range(M.shape[1]):
            sc = StandardScaler().fit(M[tr, L])
            clf = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(M[tr, L]), meta.label.values[tr])
            accs.append((clf.predict(sc.transform(M[te, L])) == meta.label.values[te]).mean())
        res[split] = accs
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for split, accs in res.items():
        ax.plot(range(len(accs)), np.array(accs) * 100, marker="o", ms=3, label=split)
    for L in [6, 9, 12, 15, 18]:
        ax.axvline(L, color="grey", lw=0.6, ls=":")
    ax.axhline(100 / 7, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("WavLM layer (0 = CNN output)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Linear probe per WavLM layer (mean-pooled); dotted = layers used by the model", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{out}/wavlm_layer_probe.png", dpi=150); plt.close(fig)
    pd.DataFrame(res).rename_axis("layer").to_csv(f"{out}/wavlm_layer_probe.csv")
    print("\nLayer probe best layers:", {s: int(np.argmax(a)) for s, a in res.items()})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--results", default=os.path.join(ROOT, "Results"))
    p.add_argument("--tsne_splits", nargs="+", default=SPEAKER_SPLITS)
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fd = find_feat_dir(a.feat_dir); R = a.results
    out = f"{R}/analysis"; os.makedirs(out, exist_ok=True)
    meta = pd.read_csv(f"{fd}/meta.csv")
    results_table(R, out)
    per_class(R, out)
    failures(R, out, meta)
    layer_probe(fd, out)
    tsne_plots(R, fd, out, device, a.tsne_splits)
    wl = [json.load(open(f))["layer_weights"] for f in glob.glob(f"{R}/speech/wavlm_global/split_*/seed*/test_metrics.json")]
    if wl:
        print("\nLearned WavLM layer weights [6, 9, 12, 15, 18]:", np.round(np.mean(wl, 0), 3))


if __name__ == "__main__":
    main()
