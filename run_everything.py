# =====================================================================
# TESS EVERYTHING: speech, text, fusion, control, analysis - ALL IN ONE (paste into one Kaggle cell)
# Needs: GPU on, and the features notebook output added as Input.
# =====================================================================
import matplotlib
matplotlib.use("Agg")
ROOT = "/kaggle/working"

# %% [markdown]
# # TESS preprocessing + feature extraction
# Run once on a Kaggle GPU notebook. Outputs go to /kaggle/working/features/
#   meta.csv            one row per clip: path, speaker, word, emotion, label, text, splits
#   logmel.npy          (N, 301, 64)  float16   log-mel spectrograms     (Option A)
#   wavlm_frames.npy    (N, 149, 5, 1024) float16  frames of 5 middle WavLM layers (Option B)
#   wavlm_layermeans.npy(N, 25, 1024) float32  mean-pooled every layer (for the layer probe plot)
#   n_samples.npy       (N,) true length in samples at 16 kHz (to mask padding)

# %% ------------------------------------------------------------------ config
import os, glob
import numpy as np
import pandas as pd
import librosa
import torch
from sklearn.model_selection import train_test_split

DATA_ROOT = os.environ.get("TESS_DIR", "/kaggle/input")            # folder containing the TESS wav files
OUT = os.environ.get("FEAT_DIR", "/kaggle/working/features")      # where features are written
SR = 16000                    # WavLM expects 16 kHz
MAX_SEC = 3.0                 # longest trimmed clip is ~2.9 s, so nothing gets cut
MAX_LEN = int(SR * MAX_SEC)   # 48000 samples
TOP_DB = 30                   # trim edges quieter than -30 dB relative to peak
WAVLM_NAME = "microsoft/wavlm-large"
WAVLM_LAYERS = [6, 9, 12, 15, 18]   # middle layers carry most emotion info (Pepino et al. 2021)
BATCH = 16
SEED = 42
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
os.makedirs(OUT, exist_ok=True)

# %% ------------------------------------------------------------------ 1. metadata
def build_metadata(root):
    rows = []
    for w in glob.glob(f"{root}/**/*.wav", recursive=True):
        fname = os.path.basename(w)
        folder = os.path.basename(os.path.dirname(w))
        emo = folder.split("_", 1)[1].lower()
        emo = {"pleasant_surprised": "ps", "pleasant_surprise": "ps"}.get(emo, emo)
        rows.append(dict(path=w, fname=fname,
                         speaker=fname.split("_")[0], word=fname.split("_")[1], emotion=emo))
    df = pd.DataFrame(rows).drop_duplicates(subset="fname")      # Kaggle copy has every file twice
    df.loc[df.speaker == "OA", "speaker"] = "OAF"                 # one mislabelled filename
    df = df.sort_values(["speaker", "emotion", "word"]).reset_index(drop=True)
    assert len(df) == 2800 and set(df.emotion) == set(EMOTIONS) and set(df.speaker) == {"OAF", "YAF"}
    df["label"] = df.emotion.map({e: i for i, e in enumerate(EMOTIONS)})
    df["text"] = "say the word " + df.word.str.lower()           # transcript (identical across emotions)
    return df

def add_splits(df):
    """Each split column holds 'train' / 'val' / 'test'. Val always comes from training data only."""
    # speaker-independent folds: train on one speaker, test on the other
    for tr_spk, te_spk in [("OAF", "YAF"), ("YAF", "OAF")]:
        col = f"split_{tr_spk}_to_{te_spk}"
        df[col] = "test"
        tr_idx = df.index[df.speaker == tr_spk]
        tr, va = train_test_split(tr_idx, test_size=0.15, random_state=SEED,
                                  stratify=df.loc[tr_idx, "emotion"])
        df.loc[tr, col] = "train"; df.loc[va, col] = "val"
    # random split, reference only (both speakers everywhere -> optimistic)
    tr, rest = train_test_split(df.index, test_size=0.30, random_state=SEED,
                                stratify=df.speaker + df.emotion)
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED,
                              stratify=(df.speaker + df.emotion)[rest])
    df["split_random"] = "train"
    df.loc[va, "split_random"] = "val"; df.loc[te, "split_random"] = "test"
    return df

# %% ------------------------------------------------------------------ 2. audio preprocessing
def load_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)        # resample 24414 Hz -> 16 kHz
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)      # trim leading/trailing silence
    y = y[:MAX_LEN]
    n = len(y)
    return np.pad(y, (0, MAX_LEN - n)).astype(np.float32), n

def logmel(y):
    # 25 ms window, 10 ms hop, 64 mel bands -> (301 frames, 64)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=64)
    return librosa.power_to_db(m, ref=1.0).T.astype(np.float16)   # absolute dB: keeps loudness cue

# %% ------------------------------------------------------------------ 3. WavLM extraction
@torch.no_grad()
def extract_wavlm(waves, lengths, model, fe, device, frames_out, means_out, start):
    """waves: list of unpadded 1-D arrays. Writes into memmaps starting at row `start`."""
    inputs = fe([w[:n] for w, n in zip(waves, lengths)], sampling_rate=SR,
                padding="max_length", max_length=MAX_LEN, return_attention_mask=True,
                return_tensors="pt")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(inputs.input_values.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1).float()          # (B, 25, T, 1024)
    T = hs.shape[2]
    # frame-level validity mask from the true sample lengths
    n_frames = torch.tensor([wavlm_valid_frames(n) for n in lengths], device=hs.device).clamp(max=T)
    mask = (torch.arange(T, device=hs.device)[None, :] < n_frames[:, None]).float()   # (B, T)
    means = (hs * mask[:, None, :, None]).sum(2) / n_frames[:, None, None]          # (B, 25, 1024)
    sel = hs[:, WAVLM_LAYERS].permute(0, 2, 1, 3)                                    # (B, T, 5, 1024)
    b = len(lengths)
    frames_out[start:start + b, :T] = sel.cpu().numpy().astype(np.float16)
    means_out[start:start + b] = means.cpu().numpy()

def wavlm_valid_frames(n_samples):
    # WavLM's CNN front end: 25 ms receptive field, 20 ms stride (400 / 320 samples)
    return max(1, (n_samples - 400) // 320 + 1)

# %% ------------------------------------------------------------------ 4. run
def prepare_main(model=None, fe=None, data_root=DATA_ROOT):
    from transformers import AutoFeatureExtractor, WavLMModel
    df = add_splits(build_metadata(data_root))
    N = len(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = fe or AutoFeatureExtractor.from_pretrained(WAVLM_NAME)
    model = (model or WavLMModel.from_pretrained(WAVLM_NAME)).to(device).eval()
    # with layer-norm feature extractors, padding + attention_mask gives the same output as no padding
    # (checked to ~1e-7). Group-norm models (e.g. wavlm-base) would leak padding into the features.
    assert model.config.feat_extract_norm == "layer", "use a layer-norm WavLM (e.g. wavlm-large)"
    n_layers = model.config.num_hidden_layers + 1
    T = wavlm_valid_frames(MAX_LEN)

    lm = np.lib.format.open_memmap(f"{OUT}/logmel.npy", "w+", np.float16, (N, 1 + MAX_LEN // 160, 64))
    fr = np.lib.format.open_memmap(f"{OUT}/wavlm_frames.npy", "w+", np.float16,
                                   (N, T, len(WAVLM_LAYERS), model.config.hidden_size))
    mn = np.lib.format.open_memmap(f"{OUT}/wavlm_layermeans.npy", "w+", np.float32,
                                   (N, n_layers, model.config.hidden_size))
    n_samples = np.zeros(N, dtype=np.int32)

    for s in range(0, N, BATCH):
        waves, lens = zip(*(load_clip(p) for p in df.path[s:s + BATCH]))
        for i, (w, n) in enumerate(zip(waves, lens)):
            lm[s + i] = logmel(w)
            n_samples[s + i] = n
        extract_wavlm(waves, lens, model, fe, device, fr, mn, s)
        if s % (BATCH * 20) == 0:
            print(f"{s}/{N}")

    for a in (lm, fr, mn):
        a.flush()
    np.save(f"{OUT}/n_samples.npy", n_samples)
    df.to_csv(f"{OUT}/meta.csv", index=False)
    print("done:", {k: v.shape for k, v in dict(logmel=lm, frames=fr, layermeans=mn).items()})
    print(df.filter(like="split_").apply(lambda c: c.value_counts()).T)
    return df




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


"""Building blocks shared by the speech, text and fusion pipelines."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def length_mask(lengths, T):
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]      # (B, T) True = valid


class AttentionPool(nn.Module):
    """Additive attention pooling over time (Mirsamadi et al., 2017). Returns pooled vector + weights."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)                   # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        return (a.unsqueeze(-1) * h).sum(1), a


class BiLSTMAttention(nn.Module):
    """Temporal / contextual modelling block: BiLSTM over the sequence, then attention pooling."""
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.pool = AttentionPool(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, lengths):
        T = x.shape[1]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(min=1, max=T).cpu(),
                                                   batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=T)
        z, attn = self.pool(self.drop(h), length_mask(lengths, T))
        return z, attn


class LogMelFrontEnd(nn.Module):
    """Option A feature block: 2-D CNN over (mel, time). Pools 4x in time and 8x in frequency."""
    def __init__(self, n_mels=64, channels=(32, 64, 128), out_dim=256, dropout=0.2):
        super().__init__()
        layers, c_in = [], 1
        pools = [(2, 2), (2, 2), (2, 1)]                     # (freq, time)
        for c, p in zip(channels, pools):
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(p), nn.Dropout(dropout)]
            c_in = c
        self.cnn = nn.Sequential(*layers)
        self.time_pool = 4
        self.proj = nn.Linear(channels[-1] * (n_mels // 8), out_dim)
        self.out_dim = out_dim

    def forward(self, x, lengths):                            # x (B, T, n_mels)
        h = self.cnn(x.transpose(1, 2).unsqueeze(1))          # (B, C, F', T')
        B, C, Fq, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
        new_len = torch.div(lengths, self.time_pool, rounding_mode="floor").clamp(min=1, max=T)
        return F.relu(self.proj(h)), new_len


class WavLMFrontEnd(nn.Module):
    """Option B feature block: learned softmax weights over saved WavLM layers (SUPERB-style),
    per-layer LayerNorm, then projection to a smaller dimension."""
    def __init__(self, n_layers, dim, out_dim=256, dropout=0.2):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def layer_weights(self):
        return torch.softmax(self.layer_logits, 0)

    def forward(self, x, lengths):                            # x (B, T, L, D)
        h = (self.norm(x) * self.layer_weights()[None, None, :, None]).sum(2)
        return self.proj(h), lengths


class SpeechModel(nn.Module):
    """Preprocessed features -> front end -> BiLSTM+attention -> classifier."""
    def __init__(self, kind, feat_shape, n_classes=7, hidden=128, dropout=0.3):
        super().__init__()
        if kind == "logmel":
            self.front = LogMelFrontEnd(n_mels=feat_shape[-1])
        else:
            self.front = WavLMFrontEnd(n_layers=feat_shape[-2], dim=feat_shape[-1])
        self.temporal = BiLSTMAttention(self.front.out_dim, hidden, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.temporal.out_dim, n_classes))
        self.emb_dim = self.temporal.out_dim

    def embed(self, x, lengths):
        h, l = self.front(x, lengths)
        z, attn = self.temporal(h, l)
        return z, attn

    def forward(self, x, lengths):
        z, _ = self.embed(x, lengths)
        return self.head(z)


class TextModel(nn.Module):
    """Frozen BERT token features (tokens x 768) -> projection -> BiLSTM+attention (contextual) -> classifier."""
    def __init__(self, in_dim=768, proj=256, n_classes=7, hidden=128, dropout=0.3):
        super().__init__()
        self.front = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, proj), nn.ReLU())
        self.temporal = BiLSTMAttention(proj, hidden, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.temporal.out_dim, n_classes))
        self.emb_dim = self.temporal.out_dim

    def embed(self, x, lengths):
        return self.temporal(self.front(x), lengths)

    def forward(self, x, lengths):
        z, _ = self.embed(x, lengths)
        return self.head(z)


class FusionModel(nn.Module):
    """Speech branch (WavLM front end + BiLSTM/attention) and text branch (BERT tokens + BiLSTM/attention),
    combined by one of three fusion blocks:
      concat - concatenate both embeddings, then a dense layer (no explicit weighting)
      gate   - one learned scalar w = sigmoid(a):  h = w*h_speech + (1-w)*h_text
      gmu    - Gated Multimodal Unit (Arevalo et al., 2017): per-sample, per-dimension gate z"""
    def __init__(self, fusion, speech_shape, text_dim=768, dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        self.fusion = fusion
        self.s_front = WavLMFrontEnd(n_layers=speech_shape[-2], dim=speech_shape[-1])
        self.s_temporal = BiLSTMAttention(self.s_front.out_dim, 128, dropout)
        self.t_front = nn.Sequential(nn.LayerNorm(text_dim), nn.Dropout(dropout), nn.Linear(text_dim, 256), nn.ReLU())
        self.t_context = BiLSTMAttention(256, 128, dropout)
        ds, dt = self.s_temporal.out_dim, self.t_context.out_dim
        if fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(ds + dt, dim), nn.ReLU())
        else:
            self.hs = nn.Linear(ds, dim)
            self.ht = nn.Linear(dt, dim)
            if fusion == "gate":
                self.gate_logit = nn.Parameter(torch.zeros(1))          # w starts at 0.5 (no preference)
            elif fusion == "gmu":
                self.gate = nn.Linear(ds + dt, dim)
            else:
                raise ValueError(fusion)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.emb_dim = dim
        self._last_gate = None

    def speech_weight(self):
        if self.fusion == "gate":
            return torch.sigmoid(self.gate_logit).item()
        if self.fusion == "gmu" and self._last_gate is not None:
            return self._last_gate
        return None

    def embed(self, xs, ls, xt, lt):
        zs, _ = self.s_temporal(*self.s_front(xs, ls))
        zt, _ = self.t_context(self.t_front(xt), lt)
        if self.fusion == "concat":
            return self.fuse(torch.cat([zs, zt], -1)), None
        hs, ht = torch.tanh(self.hs(zs)), torch.tanh(self.ht(zt))
        if self.fusion == "gate":
            w = torch.sigmoid(self.gate_logit)
            return w * hs + (1 - w) * ht, w
        z = torch.sigmoid(self.gate(torch.cat([zs, zt], -1)))           # (B, dim), 1 = trust speech
        self._last_gate = z.mean().item()
        return z * hs + (1 - z) * ht, z

    def forward(self, xs, ls, xt, lt):
        h, _ = self.embed(xs, ls, xt, lt)
        return self.head(h)

    def extra_log(self):
        w = self.speech_weight()
        return {} if w is None else {"speech_weight": float(w)}


"""Training / evaluation loop shared by all pipelines."""
import json, os, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    return dict(acc=accuracy_score(y_true, y_pred),
                uar=balanced_accuracy_score(y_true, y_pred),           # unweighted average recall
                macro_f1=f1_score(y_true, y_pred, average="macro"))


def to_device(batch, device):
    *inputs, y, rows = batch
    return [t.to(device) if torch.is_tensor(t) else t for t in inputs], y.to(device), rows


@torch.no_grad()
def predict(model, loader, device, with_embeddings=False):
    model.eval()
    ys, ps, probs, rows, embs = [], [], [], [], []
    for batch in loader:
        inputs, y, r = to_device(batch, device)
        if with_embeddings:
            z, _ = model.embed(*inputs)
            logits = model.head(z); embs.append(z.cpu())
        else:
            logits = model(*inputs)
        probs.append(torch.softmax(logits, -1).cpu()); ys.append(y.cpu()); rows.append(r)
    out = dict(y=torch.cat(ys).numpy(), prob=torch.cat(probs).numpy(), rows=torch.cat(rows).numpy())
    out["pred"] = out["prob"].argmax(1)
    if with_embeddings:
        out["emb"] = torch.cat(embs).numpy()
    return out


def train(model, train_loader, val_loader, device, out_dir, epochs=40, lr=1e-3, wd=1e-4, patience=8,
          log_fn=print):
    """Early stopping on validation UAR. Saves best weights to out_dir/best.pt. Test data never seen here."""
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    best, bad, history = -1, 0, []
    for ep in range(1, epochs + 1):
        model.train(); tot, n = 0.0, 0
        for batch in train_loader:
            inputs, y, _ = to_device(batch, device)
            loss = loss_fn(model(*inputs), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        v = predict(model, val_loader, device)
        m = metrics(v["y"], v["pred"]); sched.step(m["uar"])
        rec = dict(epoch=ep, train_loss=tot / n, **{f"val_{k}": x for k, x in m.items()})
        if hasattr(model, "extra_log"):
            rec.update(model.extra_log())
        history.append(rec)
        log_fn(" ".join(f"{k}={x:.4f}" if isinstance(x, float) else f"{k}={x}" for k, x in rec.items()))
        if m["uar"] > best:
            best, bad = m["uar"], 0
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    json.dump(history, open(f"{out_dir}/history.json", "w"), indent=1)
    model.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=device))
    return model, history


def save_test_outputs(out, meta, out_dir, class_names, extra=None):
    m = metrics(out["y"], out["pred"])
    cm = confusion_matrix(out["y"], out["pred"], labels=range(len(class_names)))
    per_class_recall = (cm.diagonal() / cm.sum(1).clip(min=1)).tolist()
    res = dict(**m, per_class_recall=dict(zip(class_names, per_class_recall)), confusion=cm.tolist())
    if extra:
        res.update(extra)
    json.dump(res, open(f"{out_dir}/test_metrics.json", "w"), indent=1)
    pred = meta.iloc[out["rows"]][["fname", "speaker", "word", "emotion"]].copy()
    pred["pred"] = [class_names[i] for i in out["pred"]]
    pred["confidence"] = out["prob"].max(1)
    pred.to_csv(f"{out_dir}/test_predictions.csv", index=False)
    if "emb" in out:
        np.savez(f"{out_dir}/test_embeddings.npz", emb=out["emb"], y=out["y"], rows=out["rows"])
    return res


"""Speech-only emotion recognition: train.

Example:
  python models/speech_pipeline/train.py --features wavlm
  python models/speech_pipeline/train.py --features logmel --norm speaker      # ablation
Trains one model per (split, seed); best checkpoint chosen on the validation set only.
"""
import argparse, json, os, sys

import torch


def speech_train_get_args(argv):
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
    return p.parse_args(argv)


def run_dir(out, features, norm, split, seed):
    return os.path.join(out, f"{features}_{norm}", split, f"seed{seed}")


def speech_train_main(argv=None):
    a = speech_train_get_args(argv)
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




"""Speech-only emotion recognition: evaluate saved checkpoints on the held-out test set.

Example:
  python models/speech_pipeline/test.py --features wavlm
Writes per-run test_metrics.json / test_predictions.csv / test_embeddings.npz, plus
Results/speech/summary_<features>_<norm>.csv (mean ± std over seeds) and confusion-matrix plots.
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def speech_test_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["logmel", "wavlm"], required=True)
    p.add_argument("--norm", choices=["global", "speaker"], default="global")
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "speech"))
    a = p.parse_args(argv)
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




"""Text-only emotion recognition: train.  python models/text_pipeline/train.py
Frozen BERT token features -> BiLSTM + attention -> classifier. Best epoch chosen on validation only."""
import argparse, json, os, sys

import torch
from torch.utils.data import DataLoader


def text_train_get_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "text"))
    return p.parse_args(argv)


def text_train_main(argv=None):
    a = text_train_get_args(argv)
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




"""Text-only emotion recognition: test.  python models/text_pipeline/test.py
Also reports a TF-IDF + logistic-regression sanity baseline and the 95% chance interval."""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def chance_interval(n, k=7):
    p = 1 / k; half = 1.96 * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def text_test_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "text"))
    a = p.parse_args(argv)
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




"""Multimodal (speech + text) emotion recognition: train.
  python models/fusion_pipeline/train.py                      # all three fusion types
  python models/fusion_pipeline/train.py --fusion gate
Speech = frozen WavLM frames, text = frozen BERT tokens; both are cached by the earlier pipelines."""
import argparse, json, os, sys

import numpy as np
import torch


def fusion_train_get_args(argv):
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
    return p.parse_args(argv)


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


def fusion_train_main(argv=None):
    a = fusion_train_get_args(argv)
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




"""Multimodal emotion recognition: test.   python models/fusion_pipeline/test.py
For every run: normal test metrics + learned speech weight + two robustness checks
(text shuffled between clips, text zeroed). If accuracy does not move, the model ignores text."""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
import torch


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


def fusion_test_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "Results", "fusion"))
    a = p.parse_args(argv)
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


def analysis_main(argv=None):
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




# =====================================================================
# RUN
# =====================================================================
import os, glob, gc, shutil, time, torch
if not (glob.glob("/kaggle/input/**/meta.csv", recursive=True) or os.path.exists("/kaggle/working/features/meta.csv")):
    assert glob.glob("/kaggle/input/**/*.wav", recursive=True), \
        "Neither features nor TESS found: Add Input -> search 'toronto emotional speech set' (ejlok1), then rerun."
    print("######## features not found -> building them from TESS (about 15 min) ########")
    prepare_main()
assert glob.glob("/kaggle/input/**/meta.csv", recursive=True) or os.path.exists("/kaggle/working/features/meta.csv"), \
    "Features not found: Add Input -> Your Work -> the notebook that created features/ (meta.csv), then rerun."
EXTRA = []          # e.g. ["--seeds", "0"] for a quick test run
def step(name, fn, args):
    t = time.time(); print(f"\n\n######## {name} ########"); fn(args + EXTRA if "train" in fn.__name__ else args)
    gc.collect(); torch.cuda.empty_cache(); print(f"######## {name} done in {(time.time() - t) / 60:.1f} min")
for feats, norm in [("logmel", "global"), ("wavlm", "global"), ("logmel", "speaker"), ("wavlm", "speaker")]:
    step(f"speech {feats} {norm} - train", speech_train_main, ["--features", feats, "--norm", norm])
    step(f"speech {feats} {norm} - test", speech_test_main, ["--features", feats, "--norm", norm])
step("text - train", text_train_main, []); step("text - test", text_test_main, [])
step("fusion - train", fusion_train_main, []); step("fusion - test", fusion_test_main, [])
step("control (no text) - train", fusion_train_main, ["--fusion", "gate", "concat", "--splits", "split_OAF_to_YAF",
                                                      "split_YAF_to_OAF", "--train_text", "zeroed"])
step("control - test", fusion_test_main, ["--out", os.path.join(ROOT, "Results", "fusion_control")])
step("analysis", analysis_main, [])
shutil.make_archive(os.path.join(ROOT, "Results_backup"), "zip", ROOT, "Results")
print("\nALL DONE. Download Results_backup.zip from the Output panel (right side) NOW.")
