"""Train, calibrate and test the two-step tissue classifier; apply the fixed pass bars (Beast, env pilot, CPU).

  python -I train_eval.py      -> results/pancan/tissue_classifier_{weights.npz,tests.csv,summary.json}

Step 1: tumour or not (TCGA-UT, Kather, BCSS tumour vs GTEx, Kather and BCSS non-tumour).
Step 2: which non-tumour tissue (Kather and BCSS only). Probabilities combined as
[p_tumour, (1 - p_tumour) * p_rest]. Bars on the 64 um crop ('crop64'); 'full' reported alongside.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tissue.bars import (class_bars, merges_for, normal_epithelium_bar,  # noqa: E402
                                  close_merges, premerge_small, tumour_organ_bars,
                                  PATHOMIC_PRECISION)
from analysis.tissue.classes import CLASSES, NONTUMOUR  # noqa: E402
from analysis.tissue.metrics import (combine, ece, fit_temperature, recall_precision,  # noqa: E402
                                     threshold_for_precision)

DATA = Path("/data/scratch/juan/datasets/tissue")
OUT = Path(__file__).resolve().parents[2] / "results" / "pancan"
CS = [0.01, 0.1, 1.0, 10.0]
MIN_TRAIN_TILES = 100  # a step-2 class with fewer 64 um training tiles is merged before training
SEED = 0


def load(feature):
    import torch
    idx = pd.concat([pd.read_csv(DATA / "public_index.csv", keep_default_na=False),
                     pd.read_csv(DATA / "gtex_index.csv", keep_default_na=False)], ignore_index=True)
    X, ids = [], []
    for f in sorted((DATA / "emb").glob("*.pt")):
        o = torch.load(f, map_location="cpu", weights_only=False)
        X.append(o[feature].float().numpy())
        ids += o["tile_id"]
    X = np.concatenate(X)
    pos = {t: i for i, t in enumerate(ids)}
    idx = idx[idx["tile_id"].isin(pos)].copy()
    idx["row"] = idx["tile_id"].map(pos)
    return X, idx


def group_weights(df, y):
    """Each (source, type-or-tissue) group gets equal total weight within its class; classes weigh equally."""
    key = df["source"] + "|" + np.where(df["source"] == "gtex", df.get("tissue", ""), df["type"])
    w = np.zeros(len(df))
    for cls in np.unique(y):
        m = y == cls
        groups = key[m]
        n_groups = groups.nunique()
        sizes = groups.map(groups.value_counts())
        w[m] = 1.0 / (len(np.unique(y)) * n_groups * sizes.values)
    return w * len(df)


def fit_lr(X, y, w, Xv, yv, wv, classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    best = None
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit(X, y, sample_weight=w)
        loss = log_loss(yv, m.predict_proba(Xv), sample_weight=wv, labels=m.classes_)
        print(f"    C={C}: validation log-loss {loss:.4f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, C, m)
    return best[2], best[1]


def logits_of(m, X):
    z = m.decision_function(X)
    return np.column_stack([np.zeros_like(z), z]) if z.ndim == 1 else z


def train(X, idx, merged):
    """Fit both steps on train rows; C by validation log-loss; temperatures on validation rows."""
    lab = idx["label"].map(lambda c: merged.get(c, c))
    tr, va = idx["split"] == "train", idx["split"] == "valid"
    one = idx.drop_duplicates("tile_id")  # GTEx tiles appear once per type they stand in for
    s1_tr = one[(one["split"] == "train")]
    s1_va = one[(one["split"] == "valid") & one["source"].isin(["tcgaut", "bcss", "gtex"])]
    y1_tr, y1_va = (s1_tr["label"] == "tumour").astype(int).values, (s1_va["label"] == "tumour").astype(int).values
    print("  step 1:", len(s1_tr), "train rows,", len(s1_va), "validation rows", flush=True)
    m1, C1 = fit_lr(X[s1_tr["row"]], y1_tr, group_weights(s1_tr, y1_tr), X[s1_va["row"]], y1_va,
                    group_weights(s1_va, y1_va), [0, 1])
    T1 = fit_temperature(logits_of(m1, X[s1_va["row"]]), y1_va)
    nt = idx["source"].isin(["kather100k", "bcss"]) & (idx["label"] != "tumour")
    s2_tr, s2_va = idx[nt & tr], idx[nt & va & (idx["source"] == "bcss")]
    y2_tr, y2_va = lab[s2_tr.index].values, lab[s2_va.index].values
    print("  step 2:", len(s2_tr), "train rows,", len(s2_va), "validation rows", flush=True)
    m2, C2 = fit_lr(X[s2_tr["row"]], y2_tr, group_weights(s2_tr, y2_tr), X[s2_va["row"]], y2_va,
                    group_weights(s2_va, y2_va), sorted(set(y2_tr)))
    T2 = fit_temperature(logits_of(m2, X[s2_va["row"]]), pd.Categorical(y2_va, categories=m2.classes_).codes)
    return {"m1": m1, "C1": C1, "T1": T1, "m2": m2, "C2": C2, "T2": T2, "merged": merged}


def predict(model, X, calibrated=True):
    """(n, 9) probabilities in CLASSES order; merged-away classes stay as zero columns."""
    t1, t2 = (model["T1"], model["T2"]) if calibrated else (1.0, 1.0)
    z1 = logits_of(model["m1"], X) / t1
    p1 = np.exp(z1[:, 1] - np.logaddexp(z1[:, 0], z1[:, 1]))
    z2 = logits_of(model["m2"], X) / t2
    p2 = np.exp(z2 - z2.max(1, keepdims=True))
    p2 /= p2.sum(1, keepdims=True)
    rest = np.zeros((len(X), len(NONTUMOUR)))
    for j, c in enumerate(model["m2"].classes_):
        rest[:, NONTUMOUR.index(c)] = p2[:, j]
    return combine(p1, rest)


def evaluate(model, X, idx, version, feature, round_):
    """Long table of test results for one tile version, plus the bar inputs."""
    lab = idx["label"].map(lambda c: model["merged"].get(c, c))
    te = idx[(idx["split"] == "test") & (idx["version"] == version)]
    P = predict(model, X[te["row"]])
    pred = np.array(CLASSES)[P.argmax(1)]
    te = te.assign(pred=pred, p_tumour=P[:, 0], true=lab[te.index].values)
    rows, tables = [], []
    for s in ["kather7k", "bcss", "tcgaut"]:
        g = te[te["source"] == s].drop_duplicates("tile_id")
        t = recall_precision(g["true"], g["pred"], CLASSES).assign(test_set=s)
        tables.append(t)
        for _, r in t.iterrows():
            rows += [(s, r["class"], m, r[m]) for m in ("n_true", "recall", "precision")]
    table = pd.concat(tables)
    tc = te[te["source"] == "tcgaut"]
    recall_by_type = tc.groupby("type").apply(lambda g: float((g["pred"] == "tumour").mean())).to_dict()
    gt = te[te["source"] == "gtex"]
    false_tumour = gt.groupby("type").apply(lambda g: float((g["pred"] == "tumour").mean())).to_dict()
    called_normal = tc.groupby("type").apply(lambda g: float((g["pred"] == "normal_epithelium").mean())).to_dict()
    shares = tc.groupby("type")["pred"].value_counts(normalize=True).rename("share").reset_index()
    for k, v in recall_by_type.items():
        rows.append(("tcgaut", k, "tumour_recall", v))
    for k, v in false_tumour.items():
        rows.append(("gtex", k, "called_tumour", v))
    for r in shares.itertuples():
        rows.append(("tcgaut_share", f"{r.type}:{r.pred}", "share", r.share))
    bcss_nt = te[(te["source"] == "bcss") & (te["true"] != "tumour")].drop_duplicates("tile_id")
    gtex_breast = gt[gt["tissue"] == "Breast"].drop_duplicates("tile_id")
    shortcut = {"bcss_nontumour_called_tumour": float((bcss_nt["pred"] == "tumour").mean()),
                "gtex_breast_called_tumour": float((gtex_breast["pred"] == "tumour").mean())}
    rows += [("shortcut", k, "share", v) for k, v in shortcut.items()]
    cal = {}
    for s in ["kather7k", "bcss"]:
        g = te[te["source"] == s].drop_duplicates("tile_id")
        y = g["true"].map(CLASSES.index).values
        cal[s] = {"ece_raw": ece(predict(model, X[g["row"]], calibrated=False), y),
                  "ece_calibrated": ece(predict(model, X[g["row"]]), y)}
        rows += [("calibration", s, k, v) for k, v in cal[s].items()]
    long = pd.DataFrame(rows, columns=["test_set", "item", "metric", "value"]).assign(
        version=version, feature=feature, round=round_)
    return long, {"table": table, "recall_by_type": recall_by_type, "false_tumour": false_tumour,
                  "called_normal": called_normal, "shortcut": shortcut, "calibration": cal}


def bars_of(ev):
    cb = class_bars(ev["table"][["test_set", "class", "n_true", "recall", "precision"]])
    organ = tumour_organ_bars(ev["recall_by_type"], ev["false_tumour"])
    normal = normal_epithelium_bar(cb.get("normal_epithelium", False), ev["called_normal"])
    cb["normal_epithelium"] = normal
    return cb, organ


def threshold_rows(idx):
    """Validation tiles for the pathomic thresholds: BCSS only, where tumour and non-tumour come from
    the same TCGA slides (the case closest to ours). TCGA-UT tumour vs GTEx normal is too easy and set
    a tumour threshold near 0; GTEx tiles have no tissue label for the other classes."""
    return idx[(idx["split"] == "valid") & (idx["source"] == "bcss")].drop_duplicates("tile_id")


def thresholds(model, X, idx):
    """Per class, the calibrated probability where validation precision, at equal class prevalence,
    reaches PATHOMIC_PRECISION."""
    g = threshold_rows(idx)
    lab = g["label"].map(lambda x: model["merged"].get(x, x)).values
    P = predict(model, X[g["row"]])
    return {c: (threshold_for_precision(P[:, k], lab == c, PATHOMIC_PRECISION, balanced=True)
                if (lab == c).any() else None) for k, c in enumerate(CLASSES)}


def run_feature(feature):
    X, idx = load(feature)
    print(feature, X.shape, flush=True)
    nt = idx[idx["source"].isin(["kather100k", "bcss"]) & (idx["label"] != "tumour")
             & (idx["split"] == "train") & (idx["version"] == "crop64")]
    merged = premerge_small(nt["label"].value_counts().to_dict(), MIN_TRAIN_TILES)
    print("  merged before training (too few tiles):", merged, flush=True)
    rounds, longs = [], []
    for round_ in (1, 2):
        model = train(X, idx, merged)
        evs = {}
        for version in ("crop64", "full"):
            long, evs[version] = evaluate(model, X, idx, version, feature, round_)
            longs.append(long)
        cb, organ = bars_of(evs["crop64"])
        failed = {c for c, ok in cb.items() if not ok}
        rounds.append({"round": round_, "merged": dict(merged), "C1": model["C1"], "C2": model["C2"],
                       "T1": model["T1"], "T2": model["T2"], "class_bars": cb,
                       "organ_bars": organ.reset_index(names="type").to_dict("records"),
                       "shortcut": evs["crop64"]["shortcut"], "calibration": evs["crop64"]["calibration"],
                       "failed": sorted(failed)})
        new = merges_for(failed - set(merged))
        if round_ == 2 or not new or feature != "conch":
            break
        merged = close_merges({**merged, **new})
        print("  failed:", sorted(failed), "-> merging", new, "and retraining once", flush=True)
    return model, rounds, pd.concat(longs), X, idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, rounds, long_c, X, idx = run_feature("conch")
    thr = thresholds(model, X, idx)
    _, rounds_u, long_u, _, _ = run_feature("uni2")
    save_classifier(OUT / "tissue_classifier_weights.npz", model, thr)
    pd.concat([long_c, long_u]).to_csv(OUT / "tissue_classifier_tests.csv", index=False)
    counts = idx.drop_duplicates("tile_id").groupby(["source", "split", "version"]).size()
    summary = {"conch": rounds, "uni2_reported_only": rounds_u, "pathomic_thresholds": thr,
               "tiles": {"|".join(k): int(v) for k, v in counts.items()}}
    (OUT / "tissue_classifier_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({"conch_final": rounds[-1]["class_bars"], "failed": rounds[-1]["failed"]}, indent=2))


def save_classifier(path, model, thr):
    """Plain numeric and unicode arrays only, so load_classifier works with allow_pickle=False."""
    np.savez(path, coef1=model["m1"].coef_, b1=model["m1"].intercept_, T1=model["T1"],
             coef2=model["m2"].coef_, b2=model["m2"].intercept_, T2=model["T2"],
             classes2=np.array([str(c) for c in model["m2"].classes_], dtype=str), classes=np.array(CLASSES),
             thresholds=np.array([np.nan if thr[c] is None else thr[c] for c in CLASSES], dtype=float),
             merged=json.dumps(model["merged"]), feature="conch")


def load_classifier(path):
    """f(conch (n, 512)) -> (n, 9) calibrated probabilities in CLASSES order."""
    w = np.load(path, allow_pickle=False)
    classes2 = [str(c) for c in w["classes2"]]

    def f(X):
        X = np.asarray(X, dtype=np.float32)
        z1 = (X @ w["coef1"].T + w["b1"]).ravel() / float(w["T1"])
        p1 = 1 / (1 + np.exp(-z1))
        z2 = (X @ w["coef2"].T + w["b2"]) / float(w["T2"])
        if z2.shape[1] == 1:  # binary step 2: one row of coefficients, for the second class
            z2 = np.column_stack([np.zeros(len(X)), z2[:, 0]])
        p2 = np.exp(z2 - z2.max(1, keepdims=True))
        p2 /= p2.sum(1, keepdims=True)
        rest = np.zeros((len(X), len(NONTUMOUR)))
        for j, c in enumerate(classes2):
            rest[:, NONTUMOUR.index(c)] = p2[:, j]
        return combine(p1, rest)
    return f


if __name__ == "__main__":
    main()
