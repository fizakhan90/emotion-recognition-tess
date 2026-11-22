"""Loading pre-extracted TESS features + splits (output of data_prep/prepare_tess.py)."""
import glob, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
SPLITS = ["split_OAF_to_YAF", "split_YAF_to_OAF", "split_random"]


def find_feat_dir(feat_dir=None):
    if feat_dir:
        return feat_dir
    for root in [os.environ.get("FEAT_DIR", ""), "/kaggle/working/features", "/kaggle/input", "./features"]:
        if not root:
            continue
        hits = glob.glob(f"{root}/**/meta.csv", recursive=True) + glob.glob(f"{root}/meta.csv")
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("meta.csv not found; pass --feat_dir")


def valid_frames(n_samples, kind):
    n = np.asarray(n_samples)
    if kind == "logmel":   # librosa centred STFT, hop 160
        return 1 + n // 160
    if kind == "wavlm":    # WavLM conv front end: 400-sample field, 320 stride
        return np.maximum(1, (n - 400) // 320 + 1)
    raise ValueError(kind)


def load_speech(feat_dir, kind):
    """Returns meta DataFrame, features array (N, T, ...) float16, valid lengths (N,)."""
    meta = pd.read_csv(f"{feat_dir}/meta.csv")
    n_samples = np.load(f"{feat_dir}/n_samples.npy")
    fname = {"logmel": "logmel.npy", "wavlm": "wavlm_frames.npy"}[kind]
    X = np.load(f"{feat_dir}/{fname}")                      # fully into RAM (wavlm ~4.3 GB fp16)
    lengths = np.minimum(valid_frames(n_samples, kind), X.shape[1]).astype(np.int64)
    return meta, X, lengths


def _masked_stats(X, lengths, idx, chunk=128):
    """Per-feature mean/std over the valid frames of rows idx (streamed, float64 accumulators)."""
    feat_shape = X.shape[2:]
    s = np.zeros(feat_shape); ss = np.zeros(feat_shape); n = 0
    for c in range(0, len(idx), chunk):
        rows = idx[c:c + chunk]
        x = X[rows].astype(np.float32)
        m = (np.arange(X.shape[1])[None, :] < lengths[rows][:, None])
        v = x[m]                                            # (frames, *feat_shape)
        s += v.sum(0); ss += (v.astype(np.float64) ** 2).sum(0); n += len(v)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def make_normaliser(X, lengths, meta, split_col, mode):
    """mode='global': stats from training rows only (strict).
       mode='speaker': stats per speaker from that speaker's audio (no labels used) - ablation."""
    if mode == "global":
        tr = np.where(meta[split_col] == "train")[0]
        mean, std = _masked_stats(X, lengths, tr)
        return lambda x, rows: (x - mean) / std
    if mode == "speaker":
        stats = {s: _masked_stats(X, lengths, np.where(meta.speaker == s)[0]) for s in meta.speaker.unique()}
        spk = meta.speaker.values
        def f(x, rows):
            out = np.empty_like(x)
            for i, r in enumerate(rows):
                m, sd = stats[spk[r]]
                out[i] = (x[i] - m) / sd
            return out
        return f
    if mode == "none":
        return lambda x, rows: x
    raise ValueError(mode)


class SpeechDataset(Dataset):
    def __init__(self, X, lengths, labels, rows, normaliser):
        self.X, self.lengths, self.labels, self.rows, self.norm = X, lengths, labels, np.asarray(rows), normaliser

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.norm(self.X[r:r + 1].astype(np.float32), [r])[0]
        return torch.from_numpy(x), int(self.lengths[r]), int(self.labels[r]), int(r)


def rows_for(meta, split_col, part):
    return np.where(meta[split_col] == part)[0]


BERT_NAME = "bert-base-uncased"


def clean_text(s):
    return " ".join(str(s).lower().strip().split())


def default_text_cache(feat_dir):
    """Kaggle: /kaggle/working/text_features (inputs are read-only). Elsewhere: next to the features folder."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/text_features"
    return os.path.join(os.path.dirname(os.path.abspath(feat_dir)), "text_features")


def load_text(feat_dir, cache_dir=None):
    """Frozen BERT last-layer token features for every clip's transcript.
    Returns meta, X (N, L, 768) float16, lengths (N,). Cached after the first run."""
    cache_dir = cache_dir or default_text_cache(feat_dir)
    meta = pd.read_csv(f"{feat_dir}/meta.csv")
    os.makedirs(cache_dir, exist_ok=True)
    fx, fl = f"{cache_dir}/bert_tokens.npy", f"{cache_dir}/bert_lengths.npy"
    if not os.path.exists(fx):
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(BERT_NAME)
        bert = AutoModel.from_pretrained(BERT_NAME).eval()
        texts = [clean_text(t) for t in meta.text]
        enc = tok(texts, padding=True, return_tensors="pt")          # [CLS] say the word back [SEP] + padding
        with torch.no_grad():
            h = torch.cat([bert(input_ids=enc.input_ids[i:i + 256],
                                attention_mask=enc.attention_mask[i:i + 256]).last_hidden_state
                           for i in range(0, len(texts), 256)])
        np.save(fx, h.numpy().astype(np.float16))
        np.save(fl, enc.attention_mask.sum(1).numpy().astype(np.int64))
        print("BERT features:", h.shape, "example tokens:", tok.convert_ids_to_tokens(enc.input_ids[0]))
    return meta, np.load(fx), np.load(fl)


class GPUFusionLoader:
    """Keeps all features on the GPU (WavLM ~4.3 GB fp16 fits a 16 GB T4) so epochs take seconds.
    Yields (speech_x, speech_len, text_x, text_len, y, rows). `text_perm` lets test.py feed each
    clip the transcript of another clip (robustness check: does the model use text at all?)."""
    def __init__(self, S, S_len, T, T_len, labels, rows, s_mean, s_std, batch=32, shuffle=False,
                 text_mode="normal", seed=0):
        self.S, self.S_len, self.T, self.T_len, self.y = S, S_len, T, T_len, labels
        self.rows = torch.as_tensor(np.asarray(rows), device=S.device)
        self.mean, self.std = s_mean, s_std
        self.batch, self.shuffle, self.text_mode = batch, shuffle, text_mode
        g = torch.Generator().manual_seed(seed)
        self.text_rows = self.rows[torch.randperm(len(self.rows), generator=g).to(S.device)] \
            if text_mode == "shuffled" else self.rows

    def __len__(self):
        return (len(self.rows) + self.batch - 1) // self.batch

    def __iter__(self):
        order = torch.randperm(len(self.rows), device=self.S.device) if self.shuffle else \
            torch.arange(len(self.rows), device=self.S.device)
        for i in range(0, len(order), self.batch):
            o = order[i:i + self.batch]
            r, rt = self.rows[o], self.text_rows[o]
            xs = (self.S[r].float() - self.mean) / self.std
            xt = self.T[rt].float()
            if self.text_mode == "zeroed":
                xt = torch.zeros_like(xt)
            yield xs, self.S_len[r], xt, self.T_len[rt], self.y[r], r.cpu()


class GPUSpeechLoader:
    """Speech-only batches from GPU-resident features (much faster than a CPU DataLoader for WavLM).
    mode 'global': one mean/std (from training rows); 'speaker': per-speaker mean/std (ablation)."""
    def __init__(self, S, S_len, labels, rows, mean, std, spk_idx=None, batch=32, shuffle=False):
        self.S, self.S_len, self.y = S, S_len, labels
        self.rows = torch.as_tensor(np.asarray(rows), device=S.device)
        self.mean, self.std, self.spk = mean, std, spk_idx          # mean/std: (n_spk, ...) if spk_idx given
        self.batch, self.shuffle = batch, shuffle

    def __len__(self):
        return (len(self.rows) + self.batch - 1) // self.batch

    def __iter__(self):
        order = torch.randperm(len(self.rows), device=self.S.device) if self.shuffle else \
            torch.arange(len(self.rows), device=self.S.device)
        for i in range(0, len(order), self.batch):
            r = self.rows[order[i:i + self.batch]]
            x = self.S[r].float()
            if self.spk is None:
                x = (x - self.mean) / self.std
            else:
                k = self.spk[r]
                x = (x - self.mean[k].unsqueeze(1)) / self.std[k].unsqueeze(1)
            yield x, self.S_len[r], self.y[r], r.cpu()


def gpu_speech_loaders(X, lengths, meta, split, norm, device, parts, batch=32):
    """Builds GPUSpeechLoaders for the requested parts. Statistics never use labels;
    'global' uses training rows only, 'speaker' uses each speaker's own audio."""
    t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
    S, L, y = t(X), t(lengths, torch.long), t(meta.label.values, torch.long)
    spk_idx = None
    if norm == "global":
        m, s = _masked_stats(X, lengths, rows_for(meta, split, "train"))
        mean, std = t(m), t(s)
    else:
        spks = sorted(meta.speaker.unique())
        st = [_masked_stats(X, lengths, np.where(meta.speaker == k)[0]) for k in spks]
        mean, std = t(np.stack([a for a, _ in st])), t(np.stack([b for _, b in st]))
        spk_idx = t(meta.speaker.map({k: i for i, k in enumerate(spks)}).values, torch.long)
    return {p: GPUSpeechLoader(S, L, y, rows_for(meta, split, p) if p != "all" else np.arange(len(meta)),
                               mean, std, spk_idx, batch=batch, shuffle=(p == "train")) for p in parts}
