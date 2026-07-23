import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_conditional_diffusion import distribution_metrics
from evaluate_knn_design_ensemble import FlatKnnIndex
from train_conditional_diffusion import VehiclePointCloudDataset, set_seed
from train_target_latent_retrieval import TargetLatentMapper, build_descriptors, normalize


EPS = 1e-8
BLEND_ALPHAS = (0.25, 0.5, 0.75)


@torch.no_grad()
def chamfer_values(a: torch.Tensor, b: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Return one Chamfer value per paired point cloud without mixing batch items."""
    vals = []
    for start in range(0, a.shape[0], chunk):
        aa = a[start:start + chunk]
        bb = b[start:start + chunk]
        dist = torch.cdist(aa, bb, p=2)
        cd = 0.5 * (dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1))
        vals.append(cd)
    return torch.cat(vals, dim=0)


def collapse_metrics(names: Sequence[str]) -> Dict[str, float]:
    counts = Counter(names)
    n = max(len(names), 1)
    if not counts:
        return {
            "unique_ratio": 0.0,
            "top_frac": 1.0,
            "eff_proto_ratio": 0.0,
            "retrieval_entropy": 0.0,
            "num_unique": 0,
        }
    probs = np.asarray([v / n for v in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + EPS)).sum())
    eff_num = float(np.exp(entropy))
    return {
        "unique_ratio": float(len(counts) / n),
        "top_frac": float(max(counts.values()) / n),
        "eff_proto_ratio": float(eff_num / n),
        "retrieval_entropy": entropy,
        "num_unique": int(len(counts)),
    }


def hard_batch_ok(metrics: Dict[str, float], args) -> bool:
    return (
        metrics["top_frac"] <= args.hard_top_frac
        and metrics["unique_ratio"] >= args.hard_unique_ratio
        and metrics["eff_proto_ratio"] >= args.hard_eff_proto_ratio
    )


def quantiles(values: Sequence[float], qs: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    return sorted({float(np.quantile(arr, q)) for q in qs})


def rule_decision(rule: Dict, row: Dict) -> bool:
    if rule["op"] == "<=":
        return row[rule["key"]] <= rule["value"]
    if rule["op"] == ">=":
        return row[rule["key"]] >= rule["value"]
    raise ValueError(rule["op"])


def sample_policy_decision(policy: Dict, row: Dict) -> bool:
    name = policy["name"]
    if name == "always":
        return True
    if name == "threshold":
        return rule_decision(policy, row)
    if name == "and":
        return all(rule_decision(rule, row) for rule in policy["rules"])
    if name == "or":
        return any(rule_decision(rule, row) for rule in policy["rules"])
    raise ValueError(name)


def selection_rank(row: Dict):
    return (
        row["tlr_nn_dist"],
        row["tlr_proto_frac"],
        row["shape_disagree_cd"],
        -row["tlr_rel_margin"],
        -row["flat_nn_dist"],
    )


def apply_proto_quota(rows: List[Dict], decisions: List[bool], max_frac: float) -> List[bool]:
    if max_frac <= 0:
        return [False] * len(rows)
    cap = max(1, int(math.floor(max_frac * max(len(rows), 1))))
    by_name = defaultdict(list)
    for i, use_tlr in enumerate(decisions):
        if use_tlr:
            by_name[rows[i]["tlr_name"]].append(i)
    out = decisions[:]
    for _, ids in by_name.items():
        if len(ids) <= cap:
            continue
        keep = set(sorted(ids, key=lambda i: selection_rank(rows[i]))[:cap])
        for i in ids:
            if i not in keep:
                out[i] = False
    return out


def candidate_label(candidate: Dict) -> str:
    return json.dumps(candidate, sort_keys=True, ensure_ascii=True)


def alpha_key(alpha: float) -> str:
    return str(int(round(alpha * 100)))


def candidate_alpha(candidate: Dict) -> float:
    return float(candidate.get("alpha", 1.0))


def candidate_cd(row: Dict, use_tlr: bool, alpha: float) -> float:
    if not use_tlr:
        return row["flat_cd"]
    if alpha >= 0.999:
        return row["tlr_cd"]
    return row[f"blend_cd_{alpha_key(alpha)}"]


def candidate_decisions(candidate: Dict, rows: List[Dict], batch: Dict[str, float], args) -> List[bool]:
    n = len(rows)
    if candidate["kind"] == "flat":
        return [False] * n
    if not hard_batch_ok(batch, args):
        return [False] * n

    if candidate["kind"] == "batch_tlr":
        decisions = [True] * n
    elif candidate["kind"] == "batch_guard":
        ok = (
            batch["top_frac"] <= candidate["top_frac"]
            and batch["unique_ratio"] >= candidate["unique_ratio"]
            and batch["eff_proto_ratio"] >= candidate["eff_proto_ratio"]
        )
        decisions = [ok] * n
    elif candidate["kind"] == "sample":
        if "batch_guard" in candidate:
            guard = candidate["batch_guard"]
            ok = (
                batch["top_frac"] <= guard["top_frac"]
                and batch["unique_ratio"] >= guard["unique_ratio"]
                and batch["eff_proto_ratio"] >= guard["eff_proto_ratio"]
            )
            if not ok:
                return [False] * n
        decisions = [sample_policy_decision(candidate["policy"], row) for row in rows]
    else:
        raise ValueError(candidate["kind"])

    quota = candidate.get("proto_quota_frac")
    if quota is not None:
        decisions = apply_proto_quota(rows, decisions, quota)
    return decisions


def train_mapper_rc(
    descs,
    train_idx: Sequence[int],
    device,
    hidden_dim=512,
    epochs=100,
    lr=1e-3,
    lambda_nce=0.1,
    temperature=0.1,
    lambda_spread=0.02,
    seed=42,
):
    set_seed(seed)
    x_train = descs["cond_desc"][train_idx]
    y_train = descs["target_desc"][train_idx]
    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
    x_train_n = normalize(x_train, x_mean, x_std)
    y_train_n = normalize(y_train, y_mean, y_std)
    model = TargetLatentMapper(x_train_n.shape[1], y_train_n.shape[1], hidden_dim, 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_idx)
    batch_size = min(128, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            ids = perm[start:start + batch_size]
            pred = model(x_train_n[ids])
            yb = y_train_n[ids]
            mse = F.mse_loss(pred, yb)
            nce = torch.tensor(0.0, device=device)
            if lambda_nce > 0 and pred.shape[0] > 1:
                logits = F.normalize(pred, dim=-1) @ F.normalize(yb, dim=-1).T / temperature
                labels = torch.arange(pred.shape[0], device=device)
                nce = F.cross_entropy(logits, labels)
            spread = torch.tensor(0.0, device=device)
            if lambda_spread > 0 and pred.shape[0] > 2:
                pred_std = pred.std(dim=0, unbiased=False)
                y_std_batch = yb.std(dim=0, unbiased=False).detach()
                var_floor = 0.35 * y_std_batch
                spread = F.relu(var_floor - pred_std).mean()
                spread = spread + 0.1 * F.mse_loss(pred.mean(dim=0), yb.mean(dim=0).detach())
            loss = mse + lambda_nce * nce + lambda_spread * spread
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, x_mean, x_std, y_mean, y_std


@torch.no_grad()
def flat_candidates(dataset, train_idx: Sequence[int], eval_idx: Sequence[int], descs, device):
    index = FlatKnnIndex(dataset, train_idx, device)
    cond = descs["conds"][eval_idx]
    target = descs["targets"][eval_idx]
    q = cond.reshape(cond.shape[0], -1).to(device)
    dist = torch.cdist(q, index.conds_flat, p=2)
    k = min(2, dist.shape[1])
    vals, pos = dist.topk(k, largest=False, dim=1)
    nn_pos = pos[:, 0]
    pred = index.targets[nn_pos]
    train_tensor = torch.tensor(train_idx, device=device)
    nn_idx = train_tensor[nn_pos]
    cds = chamfer_values(pred, target)
    rows = []
    for i in range(len(eval_idx)):
        margin = float((vals[i, 1] - vals[i, 0]).item()) if k > 1 else 0.0
        rows.append(
            {
                "flat_cd": float(cds[i].item()),
                "flat_nn_dist": float(vals[i, 0].item()),
                "flat_margin": margin,
                "flat_idx": int(nn_idx[i].item()),
                "flat_name": dataset.names[int(nn_idx[i].item())],
            }
        )
    return pred, nn_idx, rows


@torch.no_grad()
def tlr_candidates(dataset, train_idx: Sequence[int], eval_idx: Sequence[int], descs, model, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    train_targets_desc = normalize(descs["target_desc"][train_idx], y_mean, y_std)
    q = normalize(descs["cond_desc"][eval_idx], x_mean, x_std)
    pred_desc = model(q)
    dist = torch.cdist(pred_desc, train_targets_desc, p=2)
    k = min(2, dist.shape[1])
    vals, pos = dist.topk(k, largest=False, dim=1)
    nn_pos = pos[:, 0]
    train_tensor = torch.tensor(train_idx, device=device)
    nn_idx = train_tensor[nn_pos]
    pred = descs["targets"][nn_idx]
    target = descs["targets"][eval_idx]
    names = [dataset.names[int(i.item())] for i in nn_idx]
    counts = Counter(names)
    n = max(len(names), 1)
    cds = chamfer_values(pred, target)
    rows = []
    for i, name in enumerate(names):
        d1 = float(vals[i, 0].item())
        d2 = float(vals[i, 1].item()) if k > 1 else d1
        rows.append(
            {
                "tlr_cd": float(cds[i].item()),
                "tlr_nn_dist": d1,
                "tlr_margin": d2 - d1,
                "tlr_rel_margin": (d2 - d1) / (d1 + EPS),
                "tlr_proto_frac": counts[name] / n,
                "tlr_idx": int(nn_idx[i].item()),
                "tlr_name": name,
            }
        )
    return pred, nn_idx, rows, collapse_metrics(names)


@torch.no_grad()
def build_sample_rows(dataset, descs, train_idx, eval_idx, model, x_mean, x_std, y_mean, y_std, device, group_id):
    flat_pred, flat_idx, flat_rows = flat_candidates(dataset, train_idx, eval_idx, descs, device)
    tlr_pred, tlr_idx, tlr_rows, batch_metrics = tlr_candidates(
        dataset, train_idx, eval_idx, descs, model, x_mean, x_std, y_mean, y_std, device
    )
    disagree_vals = chamfer_values(flat_pred, tlr_pred)
    blend_cds = {}
    for alpha in BLEND_ALPHAS:
        blend = flat_pred * (1.0 - alpha) + tlr_pred * alpha
        blend_cds[alpha_key(alpha)] = chamfer_values(blend, descs["targets"][eval_idx])
    rows = []
    for i, idx in enumerate(eval_idx):
        disagree = float(disagree_vals[i].item())
        row = {
            "name": dataset.names[idx],
            "prefix": dataset.names[idx][:2],
            "group_id": group_id,
            "same_retrieval": int(flat_idx[i].item() == tlr_idx[i].item()),
            "shape_disagree_cd": disagree,
            **flat_rows[i],
            **tlr_rows[i],
        }
        for key, vals in blend_cds.items():
            row[f"blend_cd_{key}"] = float(vals[i].item())
        row["delta_flat_minus_tlr"] = row["flat_cd"] - row["tlr_cd"]
        rows.append(row)
    return rows, flat_pred, tlr_pred, batch_metrics


def make_sample_policies(inner_rows: List[Dict]) -> List[Dict]:
    policies = [{"name": "always"}]
    q_dense = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    low_keys = ["tlr_nn_dist", "tlr_proto_frac", "shape_disagree_cd"]
    high_keys = ["tlr_margin", "tlr_rel_margin", "flat_nn_dist"]
    for key in low_keys:
        for value in quantiles([r[key] for r in inner_rows], q_dense):
            policies.append({"name": "threshold", "key": key, "op": "<=", "value": value})
    for key in high_keys:
        for value in quantiles([r[key] for r in inner_rows], q_dense):
            policies.append({"name": "threshold", "key": key, "op": ">=", "value": value})
    for value in quantiles([r["flat_margin"] for r in inner_rows], q_dense):
        policies.append({"name": "threshold", "key": "flat_margin", "op": "<=", "value": value})
        policies.append({"name": "threshold", "key": "flat_margin", "op": ">=", "value": value})

    flat_uncertain = []
    for value in quantiles([r["flat_nn_dist"] for r in inner_rows], [0.4, 0.5, 0.6, 0.7, 0.8]):
        flat_uncertain.append({"key": "flat_nn_dist", "op": ">=", "value": value})
    for value in quantiles([r["flat_margin"] for r in inner_rows], [0.2, 0.3, 0.4, 0.5]):
        flat_uncertain.append({"key": "flat_margin", "op": "<=", "value": value})

    tlr_conf = []
    for value in quantiles([r["tlr_nn_dist"] for r in inner_rows], [0.2, 0.3, 0.4, 0.5, 0.6]):
        tlr_conf.append({"key": "tlr_nn_dist", "op": "<=", "value": value})
    for value in quantiles([r["tlr_rel_margin"] for r in inner_rows], [0.4, 0.5, 0.6, 0.7, 0.8]):
        tlr_conf.append({"key": "tlr_rel_margin", "op": ">=", "value": value})

    anti_collapse = []
    for value in quantiles([r["tlr_proto_frac"] for r in inner_rows], [0.25, 0.4, 0.5, 0.65, 0.8]):
        anti_collapse.append({"key": "tlr_proto_frac", "op": "<=", "value": value})
    bounded_disagree = []
    for value in quantiles([r["shape_disagree_cd"] for r in inner_rows], [0.4, 0.5, 0.6, 0.75, 0.9]):
        bounded_disagree.append({"key": "shape_disagree_cd", "op": "<=", "value": value})

    for a in flat_uncertain:
        for b in tlr_conf:
            policies.append({"name": "and", "rules": [a, b]})
    for a in tlr_conf:
        for b in anti_collapse:
            policies.append({"name": "and", "rules": [a, b]})
    for a in flat_uncertain[:6]:
        for b in tlr_conf[:8]:
            for c in anti_collapse:
                policies.append({"name": "and", "rules": [a, b, c]})
    for a in flat_uncertain[:6]:
        for b in tlr_conf[:8]:
            for c in bounded_disagree:
                policies.append({"name": "and", "rules": [a, b, c]})
    for a in tlr_conf[:8]:
        for b in anti_collapse:
            for c in bounded_disagree:
                policies.append({"name": "and", "rules": [a, b, c]})

    seen = set()
    unique = []
    for policy in policies:
        key = json.dumps(policy, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    return unique


def make_candidates(inner_rows: List[Dict]) -> List[Dict]:
    alpha_values = [1.0]
    if any(f"blend_cd_{alpha_key(alpha)}" in inner_rows[0] for alpha in BLEND_ALPHAS):
        alpha_values = [0.25, 0.5, 0.75, 1.0]
    candidates = [{"kind": "flat"}]
    for alpha in alpha_values:
        candidates.append({"kind": "batch_tlr", "alpha": alpha})
    for quota in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for alpha in alpha_values:
            candidates.append({"kind": "batch_tlr", "proto_quota_frac": quota, "alpha": alpha})
    for top in [0.35, 0.5, 0.65, 0.75, 0.85, 0.9]:
        for uniq in [0.025, 0.05, 0.075, 0.1]:
            for eff in [0.025, 0.05, 0.075]:
                for alpha in alpha_values:
                    candidates.append({"kind": "batch_guard", "top_frac": top, "unique_ratio": uniq, "eff_proto_ratio": eff, "alpha": alpha})

    sample_policies = make_sample_policies(inner_rows)
    guards = [
        None,
        {"top_frac": 0.9, "unique_ratio": 0.025, "eff_proto_ratio": 0.025},
        {"top_frac": 0.75, "unique_ratio": 0.05, "eff_proto_ratio": 0.05},
        {"top_frac": 0.65, "unique_ratio": 0.075, "eff_proto_ratio": 0.05},
    ]
    quotas = [None, 0.2, 0.35, 0.5, 0.65]
    for policy in sample_policies:
        for guard in guards:
            for quota in quotas:
                for alpha in alpha_values:
                    cand = {"kind": "sample", "policy": policy, "alpha": alpha}
                    if guard is not None:
                        cand["batch_guard"] = guard
                    if quota is not None:
                        cand["proto_quota_frac"] = quota
                    candidates.append(cand)

    seen = set()
    unique = []
    for cand in candidates:
        key = candidate_label(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def group_by(rows: Iterable[Dict], key: str) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


def score_candidate(candidate: Dict, rows_by_group: Dict[str, List[Dict]], batch_by_group: Dict[str, Dict], args) -> Dict:
    group_summaries = []
    all_cd = []
    all_flat = []
    accepted = 0
    num_cases = 0
    selected_names = []
    alpha = candidate_alpha(candidate)
    for group, rows in rows_by_group.items():
        batch = batch_by_group[group]
        decisions = candidate_decisions(candidate, rows, batch, args)
        cds = [candidate_cd(row, use, alpha) for row, use in zip(rows, decisions)]
        flats = [row["flat_cd"] for row in rows]
        tlr_count = int(sum(decisions))
        accepted += tlr_count
        num_cases += len(rows)
        all_cd.extend(cds)
        all_flat.extend(flats)
        selected_names.extend([row["tlr_name"] for row, use in zip(rows, decisions) if use])
        group_summaries.append(
            {
                "group": group,
                "mean_cd": float(np.mean(cds)),
                "flat_cd": float(np.mean(flats)),
                "gain_flat_minus_policy": float(np.mean(flats) - np.mean(cds)),
                "tlr_count": tlr_count,
                "num_cases": len(rows),
                "batch_top_frac": batch["top_frac"],
                "batch_unique_ratio": batch["unique_ratio"],
                "batch_eff_proto_ratio": batch["eff_proto_ratio"],
                "hard_batch_ok": hard_batch_ok(batch, args),
            }
        )
    mean_cd = float(np.mean(all_cd))
    flat_mean = float(np.mean(all_flat))
    gains = [g["gain_flat_minus_policy"] for g in group_summaries]
    harm_excess = [max(0.0, -g - args.max_inner_prefix_harm) for g in gains]
    if 0 < accepted < args.min_inner_accept:
        objective = float("inf")
    else:
        mean_gain = flat_mean - mean_cd
        objective = mean_cd
        objective += args.risk_lambda * float(np.mean(harm_excess))
        objective += args.q10_lambda * max(0.0, -float(np.quantile(gains, 0.1)))
        objective += args.negative_mean_lambda * max(0.0, args.min_mean_gain - mean_gain)
        if candidate["kind"] == "sample":
            objective += args.sample_objective_penalty
    selected_collapse = collapse_metrics(selected_names) if selected_names else collapse_metrics([])
    return {
        "candidate": candidate,
        "objective": objective,
        "inner_mean_cd": mean_cd,
        "inner_flat_mean_cd": flat_mean,
        "inner_mean_gain": float(flat_mean - mean_cd),
        "inner_min_prefix_gain": float(min(gains)) if gains else 0.0,
        "inner_q10_prefix_gain": float(np.quantile(gains, 0.1)) if gains else 0.0,
        "inner_harm_prefixes": int(sum(g < -args.max_inner_prefix_harm for g in gains)),
        "accepted": int(accepted),
        "num_cases": int(num_cases),
        "tlr_fraction": float(accepted / max(num_cases, 1)),
        "selected_collapse": selected_collapse,
        "group_summaries": group_summaries,
    }


def apply_candidate_to_rows(candidate: Dict, rows: List[Dict], batch: Dict, args):
    decisions = candidate_decisions(candidate, rows, batch, args)
    alpha = candidate_alpha(candidate)
    out = []
    for row, use_tlr in zip(rows, decisions):
        cd = candidate_cd(row, use_tlr, alpha)
        source = "flat_l2" if not use_tlr else ("tlr" if alpha >= 0.999 else f"blend_{alpha_key(alpha)}")
        out.append({**row, "source": source, "alpha": alpha if use_tlr else 0.0, "cd": cd})
    return out, decisions


def select_candidate(scored: List[Dict], args) -> Dict:
    finite = [s for s in scored if np.isfinite(s["objective"])]
    if not finite:
        return next(s for s in scored if s["candidate"]["kind"] == "flat")
    finite.sort(key=lambda x: (x["objective"], x["inner_mean_cd"]))
    best = finite[0]
    if best["inner_mean_gain"] < args.min_mean_gain:
        return next(s for s in scored if s["candidate"]["kind"] == "flat")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", default="splits/prefix_all10")
    p.add_argument("--point_dir", default="models_pointcloud_npy")
    p.add_argument("--label_dir", default="models_labels_npy")
    p.add_argument("--output_dir", default="eval/rc_tlr_v2")
    p.add_argument("--inner_epochs", type=int, default=80)
    p.add_argument("--outer_epochs", type=int, default=200)
    p.add_argument("--calibration_epochs", type=int, default=80)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--lambda_nce", type=float, default=0.1)
    p.add_argument("--lambda_spread", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard_top_frac", type=float, default=0.9)
    p.add_argument("--hard_unique_ratio", type=float, default=0.025)
    p.add_argument("--hard_eff_proto_ratio", type=float, default=0.025)
    p.add_argument("--min_inner_accept", type=int, default=20)
    p.add_argument("--min_mean_gain", type=float, default=0.0)
    p.add_argument("--max_inner_prefix_harm", type=float, default=0.0015)
    p.add_argument("--risk_lambda", type=float, default=2.0)
    p.add_argument("--q10_lambda", type=float, default=0.5)
    p.add_argument("--negative_mean_lambda", type=float, default=5.0)
    p.add_argument("--sample_objective_penalty", type=float, default=0.0)
    p.add_argument("--val_repeat", type=int, default=0)
    p.add_argument("--final_include_val", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VehiclePointCloudDataset(args.point_dir, args.label_dir)
    descs = build_descriptors(dataset, device)
    summaries, all_rows = [], []

    split_paths = sorted(Path(args.split_dir).glob("prefix_fold*.json"))
    if not split_paths:
        raise FileNotFoundError(f"No prefix folds found under {args.split_dir}")

    for fold, split_path in enumerate(split_paths):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_idx, test_idx = split["train"], split["test"]
        val_idx = split.get("val", [])
        train_prefixes = sorted({dataset.names[idx][:2] for idx in train_idx})
        inner_rows = []
        batch_by_group = {}

        for pi, prefix in enumerate(train_prefixes):
            inner_val = [idx for idx in train_idx if dataset.names[idx][:2] == prefix]
            inner_train = [idx for idx in train_idx if dataset.names[idx][:2] != prefix]
            model, x_mean, x_std, y_mean, y_std = train_mapper_rc(
                descs,
                inner_train,
                device,
                hidden_dim=args.hidden_dim,
                epochs=args.inner_epochs,
                lambda_nce=args.lambda_nce,
                temperature=args.temperature,
                lambda_spread=args.lambda_spread,
                seed=args.seed + fold * 101 + pi,
            )
            rows, _, _, batch_metrics = build_sample_rows(
                dataset, descs, inner_train, inner_val, model, x_mean, x_std, y_mean, y_std, device, prefix
            )
            inner_rows.extend(rows)
            batch_by_group[prefix] = batch_metrics

        if args.val_repeat > 0 and val_idx:
            val_group = f"val:{split.get('val_prefix', 'heldout')}"
            model, x_mean, x_std, y_mean, y_std = train_mapper_rc(
                descs,
                train_idx,
                device,
                hidden_dim=args.hidden_dim,
                epochs=args.calibration_epochs,
                lambda_nce=args.lambda_nce,
                temperature=args.temperature,
                lambda_spread=args.lambda_spread,
                seed=args.seed + fold * 1543 + 17,
            )
            val_rows, _, _, val_batch = build_sample_rows(
                dataset, descs, train_idx, val_idx, model, x_mean, x_std, y_mean, y_std, device, val_group
            )
            for _ in range(args.val_repeat):
                inner_rows.extend([{**row} for row in val_rows])
            batch_by_group[val_group] = val_batch

        rows_by_group = group_by(inner_rows, "group_id")
        candidates = make_candidates(inner_rows)
        scored = [score_candidate(candidate, rows_by_group, batch_by_group, args) for candidate in candidates]
        best = select_candidate(scored, args)

        final_train_idx = list(train_idx) + list(val_idx) if args.final_include_val and val_idx else train_idx
        model, x_mean, x_std, y_mean, y_std = train_mapper_rc(
            descs,
            final_train_idx,
            device,
            hidden_dim=args.hidden_dim,
            epochs=args.outer_epochs,
            lambda_nce=args.lambda_nce,
            temperature=args.temperature,
            lambda_spread=args.lambda_spread,
            seed=args.seed + fold * 1009,
        )
        outer_rows_raw, flat_pred, tlr_pred, outer_batch = build_sample_rows(
            dataset,
            descs,
            final_train_idx,
            test_idx,
            model,
            x_mean,
            x_std,
            y_mean,
            y_std,
            device,
            split.get("test_prefix", f"fold{fold}"),
        )
        outer_rows, decisions = apply_candidate_to_rows(best["candidate"], outer_rows_raw, outer_batch, args)
        alpha = candidate_alpha(best["candidate"])
        alpha_mask = torch.tensor([alpha if use else 0.0 for use in decisions], device=device, dtype=flat_pred.dtype).view(-1, 1, 1)
        pred = flat_pred * (1.0 - alpha_mask) + tlr_pred * alpha_mask
        target = descs["targets"][test_idx]
        cond = descs["conds"][test_idx]
        selected_names = [row["tlr_name"] for row, use in zip(outer_rows_raw, decisions) if use]

        top_scored = sorted(
            [
                {
                    "candidate": s["candidate"],
                    "objective": s["objective"],
                    "inner_mean_cd": s["inner_mean_cd"],
                    "inner_mean_gain": s["inner_mean_gain"],
                    "inner_min_prefix_gain": s["inner_min_prefix_gain"],
                    "inner_q10_prefix_gain": s["inner_q10_prefix_gain"],
                    "inner_harm_prefixes": s["inner_harm_prefixes"],
                    "accepted": s["accepted"],
                    "tlr_fraction": s["tlr_fraction"],
                    "selected_collapse": s["selected_collapse"],
                }
                for s in scored
            ],
            key=lambda x: (x["objective"], x["inner_mean_cd"]),
        )[:20]

        summary = {
            "fold": fold,
            "test_prefix": split.get("test_prefix"),
            "val_prefix": split.get("val_prefix"),
            "mean_cd": float(np.mean([r["cd"] for r in outer_rows])),
            "flat_l2_cd": float(np.mean([r["flat_cd"] for r in outer_rows])),
            "tlr_all_cd": float(np.mean([r["tlr_cd"] for r in outer_rows])),
            "num_cases": len(outer_rows),
            "num_tlr_samples": int(sum(decisions)),
            "tlr_sample_fraction": float(sum(decisions) / max(len(decisions), 1)),
            "outer_batch": outer_batch,
            "outer_selected_collapse": collapse_metrics(selected_names) if selected_names else collapse_metrics([]),
            "hard_batch_ok": hard_batch_ok(outer_batch, args),
            "best_candidate": best["candidate"],
            "best_inner_objective": best["objective"],
            "best_inner_mean_cd": best["inner_mean_cd"],
            "best_inner_flat_mean_cd": best["inner_flat_mean_cd"],
            "best_inner_mean_gain": best["inner_mean_gain"],
            "best_inner_min_prefix_gain": best["inner_min_prefix_gain"],
            "best_inner_q10_prefix_gain": best["inner_q10_prefix_gain"],
            "best_inner_harm_prefixes": best["inner_harm_prefixes"],
            "best_inner_tlr_fraction": best["tlr_fraction"],
            "best_alpha": alpha,
            "val_repeat": args.val_repeat,
            "final_include_val": bool(args.final_include_val and val_idx),
            "final_train_size": len(final_train_idx),
            "best_group_summaries": best["group_summaries"],
            "top_candidate_scores": top_scored,
            "split_path": str(split_path),
        }
        summary.update({f"gen_{key}": value for key, value in distribution_metrics(pred, target).items()})
        summary.update({f"condition_{key}": value for key, value in distribution_metrics(cond, target).items()})
        summaries.append(summary)
        all_rows.extend([{**r, "fold": fold} for r in outer_rows])
        print(json.dumps(summary), flush=True)

    aggregate = {
        "folds": len(summaries),
        "mean_cd": float(np.mean([s["mean_cd"] for s in summaries])),
        "std_cd": float(np.std([s["mean_cd"] for s in summaries], ddof=1)),
        "mean_flat_l2_cd": float(np.mean([s["flat_l2_cd"] for s in summaries])),
        "mean_tlr_all_cd": float(np.mean([s["tlr_all_cd"] for s in summaries])),
        "mean_coverage_cd": float(np.mean([s["gen_coverage_cd"] for s in summaries])),
        "mean_mmd_cd": float(np.mean([s["gen_mmd_cd"] for s in summaries])),
        "tlr_sample_fraction": float(np.sum([s["num_tlr_samples"] for s in summaries]) / np.sum([s["num_cases"] for s in summaries])),
        "tlr_enabled_prefixes": [s["test_prefix"] for s in summaries if s["num_tlr_samples"] > 0],
        "hard_batch_failed_prefixes": [s["test_prefix"] for s in summaries if not s["hard_batch_ok"]],
        "max_outer_top_frac": float(max(s["outer_batch"]["top_frac"] for s in summaries)),
        "min_outer_unique_ratio": float(min(s["outer_batch"]["unique_ratio"] for s in summaries)),
        "min_outer_eff_proto_ratio": float(min(s["outer_batch"]["eff_proto_ratio"] for s in summaries)),
    }
    result = {"aggregate": aggregate, "splits": summaries}
    with open(out / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
