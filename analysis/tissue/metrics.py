"""Metrics and calibration for the tissue classifier (numpy and pandas only)."""
import numpy as np
import pandas as pd


def combine(p_tumour, p_rest):
    """Two-step output to class probabilities in CLASSES order: [p_tumour, (1 - p_tumour) * p_rest]."""
    p_tumour = np.asarray(p_tumour, dtype=float)
    return np.column_stack([p_tumour, (1 - p_tumour)[:, None] * np.asarray(p_rest, dtype=float)])


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _nll(logits, y, t):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def fit_temperature(logits, y, lo=0.05, hi=20.0, iters=80):
    """Temperature T in [lo, hi] minimising the NLL of softmax(logits / T); golden-section on log T."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=int)
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll(logits, y, np.exp(c)), _nll(logits, y, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll(logits, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll(logits, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def ece(probs, y, bins=15):
    """Expected calibration error of the top-class probability, equal-width bins."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=int)
    conf, correct = probs.max(1), (probs.argmax(1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(y))


def recall_precision(y_true, y_pred, classes):
    """Per class: n_true, recall (NaN when n_true is 0) and precision (NaN when never predicted)."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for c in classes:
        n_true, n_pred, tp = int((t == c).sum()), int((p == c).sum()), int(((t == c) & (p == c)).sum())
        rows.append({"class": c, "n_true": n_true, "recall": tp / n_true if n_true else np.nan,
                     "precision": tp / n_pred if n_pred else np.nan})
    return pd.DataFrame(rows)


def threshold_for_precision(p, is_k, target=0.90, balanced=False):
    """Lowest threshold t with precision of (p >= t) at least target; None if never reached.
    balanced=True computes precision as if classes k and not-k were equally common
    (TPR / (TPR + FPR)), so a validation set dominated by class k cannot make any threshold look precise."""
    p, is_k = np.asarray(p, dtype=float), np.asarray(is_k, dtype=bool)
    if not is_k.any():
        return None
    order = np.argsort(-p, kind="stable")
    ps, ks = p[order], is_k[order]
    tp, fp = np.cumsum(ks), np.cumsum(~ks)
    last = np.r_[ps[1:] != ps[:-1], True]  # a threshold keeps every tie at its value
    ps, tp, fp = ps[last], tp[last], fp[last]
    if balanced:
        tpr, fpr = tp / ks.sum(), fp / max((~ks).sum(), 1)
        prec = tpr / (tpr + fpr)
    else:
        prec = tp / (tp + fp)
    ok = np.nonzero(prec >= target)[0]
    return float(ps[ok[-1]]) if len(ok) else None
