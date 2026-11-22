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
def main(model=None, fe=None, data_root=DATA_ROOT):
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


if __name__ == "__main__":
    main()
