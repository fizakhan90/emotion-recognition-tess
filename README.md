# Multimodal Emotion Recognition on TESS (speech, text, speech + text)

Speech-only, text-only and multimodal emotion classifiers on the Toronto Emotional Speech Set (TESS),
evaluated **speaker-independently** (train on one actress, test on the other) as well as on a random split.

| Model (accuracy %, mean ± std over 3 seeds) | OAF → YAF | YAF → OAF | Random split |
|---|---|---|---|
| Speech: log-mel CNN-BiLSTM-attention | 35.7 ± 4.5 | 31.6 ± 5.1 | 99.4 ± 0.3 |
| Speech: WavLM + BiLSTM-attention | 59.3 ± 2.6 | 61.7 ± 6.0 | 99.9 ± 0.1 |
| Text: BERT + BiLSTM-attention | 14.3 ± 0.0 | 14.3 ± 0.0 | 14.3 ± 0.0 |
| Multimodal: concat | 64.0 ± 3.9 | 60.6 ± 3.7 | 99.9 ± 0.1 |
| Multimodal: scalar gate | 66.7 ± 4.0 | 64.0 ± 4.2 | 99.7 ± 0.4 |
| Multimodal: GMU | 65.5 ± 13.2 | 56.4 ± 5.5 | 99.9 ± 0.1 |
| Control: gate architecture, no text | 68.4 ± 1.0 | 63.5 ± 5.1 | – |

TESS transcripts are identical across emotions ("Say the word ___"), so text is at chance (14.3 %) and the
multimodal models provably ignore it (shuffling or blanking the text changes nothing). Full table, ablations,
per-class results, failure cases and t-SNE plots: `Results/analysis/` and the report.

## Layout

```
data_prep/prepare_tess.py        dedupe, label cleaning, splits, 16 kHz / trim / pad, log-mel + WavLM features
common/                          shared data loading, model blocks (layers.py) and training loop (engine.py)
models/speech_pipeline/          train.py, test.py   (--features logmel | wavlm, --norm global | speaker)
models/text_pipeline/            train.py, test.py   (frozen BERT tokens -> BiLSTM + attention)
models/fusion_pipeline/          train.py, test.py   (--fusion concat gate gmu, --train_text zeroed = control)
analysis/analyze.py              results table, per-class, failure cases, t-SNE, silhouette, WavLM layer probe
run_everything.py                single-file version of all of the above for one Kaggle cell
Results/                         accuracy tables (summary_*.csv, all_runs.csv), confusion matrices, analysis/
requirements.txt
```

## Blocks

| Block | Speech | Text |
|---|---|---|
| Preprocessing | 16 kHz, trim −30 dB, pad to 3 s | lower-case, BERT tokenizer |
| Features | A: log-mel 64 × 301 · B: frozen WavLM-Large layers 6/9/12/15/18 | frozen BERT-base tokens × 768 |
| Temporal / contextual | (CNN or learned layer weights) → BiLSTM → attention pooling | BiLSTM → attention pooling |
| Fusion | concat · learned scalar gate · Gated Multimodal Unit | |
| Classifier | dropout + linear (7 classes) | |

## Setup

```bash
pip install -r requirements.txt
```
A GPU with ≥ 16 GB is recommended (WavLM features, 4.3 GB, are kept on the GPU). Tested on Kaggle (T4).

Data: add the Kaggle dataset `ejlok1/toronto-emotional-speech-set-tess` (on Kaggle: *Add Input*; locally:
download it and pass its folder). The Kaggle copy contains every file twice; `prepare_tess.py` removes duplicates.

## Run

```bash
python data_prep/prepare_tess.py                     # features -> /kaggle/working/features (edit DATA_ROOT/OUT for local runs)

python models/speech_pipeline/train.py --features logmel && python models/speech_pipeline/test.py --features logmel
python models/speech_pipeline/train.py --features wavlm  && python models/speech_pipeline/test.py --features wavlm
python models/speech_pipeline/train.py --features wavlm --norm speaker && python models/speech_pipeline/test.py --features wavlm --norm speaker

python models/text_pipeline/train.py   && python models/text_pipeline/test.py
python models/fusion_pipeline/train.py && python models/fusion_pipeline/test.py
python models/fusion_pipeline/train.py --fusion gate concat --splits split_OAF_to_YAF split_YAF_to_OAF --train_text zeroed
python models/fusion_pipeline/test.py --out Results/fusion_control

python analysis/analyze.py
```

Or on Kaggle: paste `run_everything.py` into one cell (GPU + Internet on, TESS added as input) and use
*Save & Run All*; it builds the features if missing, runs every step above and zips `Results/`.

### Running outside Kaggle

Paths default to Kaggle's folders. On any other machine, set two environment variables:

```bash
export TESS_DIR=/path/to/TESS          # folder that contains the TESS .wav files (any depth)
export FEAT_DIR=$PWD/features          # where extracted features are written and read
python data_prep/prepare_tess.py
python models/speech_pipeline/train.py --features wavlm   # ... then the other commands above
```

(PowerShell: `$env:TESS_DIR="C:\path\to\TESS"`, `$env:FEAT_DIR="$PWD\features"`.)
BERT text features are cached in a `text_features/` folder next to `FEAT_DIR` (on Kaggle: `/kaggle/working/text_features`).
WavLM-Large and BERT-base are downloaded from Hugging Face on first use, so internet access is needed once.

Every script also accepts `--feat_dir` if the features are elsewhere. Test sets are used only by `test.py`;
model selection uses the validation split of the training speaker.

## Splits

| Split | Train | Val | Test |
|---|---|---|---|
| split_OAF_to_YAF | OAF 1,190 | OAF 210 | YAF 1,400 |
| split_YAF_to_OAF | YAF 1,190 | YAF 210 | OAF 1,400 |
| split_random (reference) | 1,960 | 420 | 420 |

Model checkpoints (`best.pt`) and test embeddings are not in the repository (size); they are in the Drive zip.
