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
