| model                                         | split_OAF_to_YAF   | split_YAF_to_OAF   | split_random   |   speaker-split mean |
|:----------------------------------------------|:-------------------|:-------------------|:---------------|---------------------:|
| Speech: log-mel CNN-BiLSTM-attn               | 35.7 ± 4.5         | 31.6 ± 5.1         | 99.4 ± 0.3     |                 33.7 |
| Speech: log-mel + per-speaker norm (ablation) | 64.4 ± 1.2         | 48.9 ± 7.9         | 99.4 ± 0.3     |                 56.6 |
| Speech: WavLM + BiLSTM-attn                   | 59.3 ± 2.6         | 61.7 ± 6.0         | 99.9 ± 0.1     |                 60.5 |
| Speech: WavLM + per-speaker norm (ablation)   | 63.5 ± 5.6         | 62.5 ± 6.2         | 99.8 ± 0.2     |                 63   |
| Text: BERT + BiLSTM-attn                      | 14.3 ± 0.0         | 14.3 ± 0.0         | 14.3 ± 0.0     |                 14.3 |
| Text: TF-IDF + LogReg (sanity)                | 14.3 ± 0.0         | 14.3 ± 0.0         | 0.2 ± 0.0      |                 14.3 |
| Fusion: concat                                | 64.0 ± 3.9         | 60.6 ± 3.7         | 99.9 ± 0.1     |                 62.3 |
| Fusion: scalar gate                           | 66.7 ± 4.0         | 64.0 ± 4.2         | 99.7 ± 0.4     |                 65.4 |
| Fusion: GMU                                   | 65.5 ± 13.2        | 56.4 ± 5.5         | 99.9 ± 0.1     |                 61   |
| Control: concat arch, NO text (speech-only)   | 59.0 ± 5.0         | 56.2 ± 2.7         | nan            |                 57.6 |
| Control: gate arch, NO text (speech-only)     | 68.4 ± 1.0         | 63.5 ± 5.1         | nan            |                 66   |