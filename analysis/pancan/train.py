"""Train our per-tissue model or ABMIL on hospital-held-out rounds (Beast, env audit; GPU only after the run ends).

  python -I train.py --model ours|abmil --variant main|stroma --rounds 0:20 [--epochs 30] [--device cuda]
                     [--subset all|<name>] [--batch-size 32] [--workers 8]

Same model and settings as the paper (analysis/region_ablation.py, abmil_compartment_shapley.py), except:
batches come from one cancer type at a time and the Cox loss and every C-index are within type
(analysis/pancan/metrics.py); out-of-bag (OOB) patients are whole hospitals (analysis/pancan/splits.py); inputs are
the per-slide bags of analysis/pancan/bags.py.

Writes /data/processed/juan/pancan/runs/<variant>/<model>/<subset>/round_###/:
  epoch_##.pt (state dict) and oob.npz (patient, time, event, type, risk[epoch, patient], cindex[epoch],
  region_weights[epoch, R] for ours). A round with oob.npz is skipped, so the command is re-runnable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

torch.multiprocessing.set_sharing_strategy("file_system")  # bags hold many small tensors; fd sharing hits ulimit 1024

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from abmil_compartment_shapley import GatedABMIL  # noqa: E402
from region_ablation import IPGGraphFormer, normalize_xy, set_seed  # noqa: E402

from analysis.pancan.metrics import cindex_within, cox_loss_stratified  # noqa: E402
from analysis.pancan.splits import site_rounds, type_batches  # noqa: E402

PANCAN = Path("/data/processed/juan/pancan")
N_ROUNDS = 20  # design decision 10: about 20 rounds for the full cohort
ABMIL_MAX_TILES = 4096  # as in the paper's ABMIL


class BagDataset(Dataset):
    def __init__(self, table):
        self.t = table.reset_index(drop=True)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        o = torch.load(r["bag"], map_location="cpu", weights_only=False)
        feats = [f if len(f) else None for f in o["feats"]]
        pos = [torch.from_numpy(normalize_xy(np.asarray(c))) if len(c) else None for c in o["coords"]]
        return feats, pos, float(r["pfi_time"]), int(r["pfi_event"]), r["type"], r["patient"]


def collate(batch):
    feats, pos, t, e, ty, pid = zip(*batch)
    return list(feats), list(pos), np.array(t), np.array(e), np.array(ty), list(pid)


class TypeBatchSampler(Sampler):
    def __init__(self, types, batch_size, seed):
        self.types, self.bs, self.seed, self.epoch = np.asarray(types), batch_size, seed, 0

    def __iter__(self):
        self.epoch += 1
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        return iter([b.tolist() for b in type_batches(self.types, self.bs, rng)])

    def __len__(self):
        return len(type_batches(self.types, self.bs, np.random.default_rng(0)))


def make_model(kind, R):
    return IPGGraphFormer(num_regions=R) if kind == "ours" else GatedABMIL()


def risks_of(model, kind, feats, pos, device, train):
    if kind == "ours":
        return model.forward_batch(feats, pos)
    return torch.stack([model.forward_bag(f, device, ABMIL_MAX_TILES if train else None) for f in feats])


def optimiser(model, kind, steps_per_epoch, epochs):
    """The paper's settings: ours AdamW 3e-5 (region weights 1e-3), 3-epoch warmup then cosine; ABMIL AdamW 1e-4."""
    if kind == "abmil":
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4), None
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if n != "region_weights"], "weight_decay": 1e-3},
         {"params": [model.region_weights], "lr": 1e-3, "weight_decay": 0.0}], lr=3e-5)
    warm = min(3, max(epochs - 1, 1)) * steps_per_epoch
    total = max(epochs * steps_per_epoch - warm, 1)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.33, 1.0, total_iters=warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=5e-6)], milestones=[warm])
    return opt, sched


@torch.no_grad()
def predict(model, kind, loader, device):
    model.eval()
    out = []
    for feats, pos, *_ in loader:
        out.append(risks_of(model, kind, feats, pos, device, train=False).float().cpu().numpy())
    return np.concatenate(out)


def train_round(kind, R, train_table, oob_table, out_dir, epochs, device, batch_size=32, workers=8, seed=0):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(kind, R).to(device)
    sampler = TypeBatchSampler(train_table["type"].values, batch_size, seed)
    tl = DataLoader(BagDataset(train_table), batch_sampler=sampler, collate_fn=collate, num_workers=workers)
    ol = DataLoader(BagDataset(oob_table), batch_size=16, shuffle=False, collate_fn=collate, num_workers=workers)
    opt, sched = optimiser(model, kind, len(sampler), epochs)
    risk, cidx, weights = [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for feats, pos, t, e, ty, _ in tl:
            if e.sum() == 0:
                if sched is not None:
                    sched.step()
                continue
            r = risks_of(model, kind, feats, pos, device, train=True)
            loss = cox_loss_stratified(r, torch.tensor(t, dtype=torch.float32, device=device),
                                       torch.tensor(e, dtype=torch.float32, device=device), ty)
            if kind == "ours":
                w = torch.relu(model.region_weights) + 0.01
                w = w / w.sum()
                loss = loss + 1e-3 * (-(w * torch.log(w + 1e-8)).sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        p = predict(model, kind, ol, device)
        c = cindex_within(oob_table["pfi_time"].values, oob_table["pfi_event"].values, p, oob_table["type"].values)
        risk.append(p)
        cidx.append(c)
        if kind == "ours":
            w = torch.relu(model.region_weights.detach().cpu()) + 0.01
            weights.append((w / w.sum()).numpy())
        torch.save(model.state_dict(), out_dir / f"epoch_{ep:02d}.pt")
        print(f"  epoch {ep}/{epochs} loss={tot / max(nb, 1):.4f} OOB within-type C={c:.4f}", flush=True)
    np.savez(out_dir / "oob.npz", patient=np.asarray(oob_table["patient"], dtype=str),
             time=oob_table["pfi_time"].values, event=oob_table["pfi_event"].values,
             type=np.asarray(oob_table["type"], dtype=str), risk=np.array(risk), cindex=np.array(cidx),
             region_weights=np.array(weights) if weights else np.zeros((0, R)))
    return cidx


def patient_table(variant, subset="all"):
    """Cohort patients whose bag exists, with their bag path (subset: 'all' or a patient list from curve.py)."""
    c = pd.read_csv(PANCAN / "cohort.csv")
    c["bag"] = [str(PANCAN / f"bags_{variant}" / t / f"{s}.pt") for t, s in zip(c["type"], c["slide"])]
    c = c[[Path(b).exists() for b in c["bag"]]]
    if subset != "all":
        keep = set(pd.read_csv(PANCAN / "curve" / f"{subset}.csv")["patient"])
        c = c[c["patient"].isin(keep)]
    return c.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "abmil"], required=True)
    ap.add_argument("--variant", choices=["main", "stroma"], required=True)
    ap.add_argument("--rounds", default=f"0:{N_ROUNDS}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(PANCAN / "runs"))
    a = ap.parse_args(argv)
    table = patient_table(a.variant, a.subset)
    R = len(torch.load(table["bag"].iat[0], map_location="cpu", weights_only=False)["compartments"])
    cohort = pd.read_csv(PANCAN / "cohort.csv")
    rounds = site_rounds(cohort, N_ROUNDS)  # fixed on the whole cohort, whatever is finished
    lo, hi = (int(x) for x in a.rounds.split(":"))
    print(f"{a.model} {a.variant} {a.subset}: {len(table)} patients, {table['pfi_event'].sum()} events, R={R}")
    for r in range(lo, hi):
        out = Path(a.out) / a.variant / a.model / a.subset / f"round_{r:03d}"
        if (out / "oob.npz").exists():
            print(f"round {r}: done, skipped")
            continue
        tr = set(rounds.loc[(rounds["round"] == r) & rounds["in_train"], "patient"])
        train_t, oob_t = table[table["patient"].isin(tr)], table[~table["patient"].isin(tr)]
        print(f"round {r}: train {len(train_t)} ({train_t['pfi_event'].sum()} events), "
              f"OOB {len(oob_t)} ({oob_t['pfi_event'].sum()} events)", flush=True)
        train_round(a.model, R, train_t, oob_t, out, a.epochs, a.device, a.batch_size, a.workers, seed=r)


if __name__ == "__main__":
    main()
