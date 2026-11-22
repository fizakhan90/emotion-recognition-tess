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
