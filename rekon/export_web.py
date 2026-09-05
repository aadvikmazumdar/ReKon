#!/usr/bin/env python3
"""Export the pipeline's results as a single JSON file for the web demo.

    python -m rekon.export_web                 uses data already on disk
    python -m rekon.export_web --regenerate    rebuilds everything first

Everything the interface needs is computed here, once. The frontend is a
viewer over a static file: nothing is scored at request time, so there is no
server to cold-start, time out, or fail during a recording.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evidence import build_pair_table
from .fellegi_sunter import FellegiSunter, agreement_matrix
from .generate import generate, label_evasion
from .graph import assign_clusters, build_graph, cluster_features, cluster_purity, edge_score
from .risk import (
    RiskModel,
    build_features,
    calibration_table,
    cost_optimal_thresholds,
    precision_at_n,
)
from .split import operator_split

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent.parent / "web" / "public" / "data.json"

COST_FP, COST_FN, COST_REVIEW = 25_000, 300_000, 500
REVIEW_CAPACITY = 0.03

FIELD_LABELS = [
    ("pan_exact", "PAN identical"),
    ("gstin_pan_match", "GSTIN carries the same PAN"),
    ("gstin_pan_cross", "GSTIN carries the other record's PAN"),
    ("phone_exact", "Phone identical"),
    ("account_exact", "Settlement account identical"),
    ("ifsc_exact", "Same bank branch"),
    ("name_jw", "Name similarity"),
    ("business_jw", "Business name similarity"),
    ("address_overlap", "Address overlap"),
    ("email_jw", "Email similarity"),
    ("pincode_exact", "Same pincode"),
    ("mcc_exact", "Same merchant category"),
]


def f(x) -> float:
    """Plain float, with NaN and inf flattened so JSON stays valid."""
    v = float(x)
    return 0.0 if (np.isnan(v) or np.isinf(v)) else round(v, 6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.regenerate or not (DATA / "applications.csv").exists():
        print("generating...")
        df = generate(n_target=args.n, seed=args.seed)
        df["is_evasion"] = label_evasion(df)
        df.to_csv(DATA / "applications.csv", index=False)
    else:
        df = pd.read_csv(DATA / "applications.csv")

    print("building evidence...")
    pairs = build_pair_table(df)
    pairs["hand_score"] = edge_score(pairs)

    train, test = operator_split(df)
    tr_seq, te_seq = set(train.app_seq), set(test.app_seq)

    print("fitting linkage model...")
    tr_pairs = pairs[pairs.seq_a.isin(tr_seq) & pairs.seq_b.isin(tr_seq)]
    fs = FellegiSunter().fit(agreement_matrix(tr_pairs))
    pairs["edge_score"] = fs.score(agreement_matrix(pairs))

    y_pair = pairs.same_operator.values
    order = np.argsort(-pairs.edge_score.values)
    cum = np.cumsum(y_pair[order])
    prec = cum / np.arange(1, len(order) + 1)
    ok = np.where(prec == 1.0)[0]
    k = int(ok.max()) if len(ok) else 0
    edge_threshold = f(pairs.edge_score.values[order][k])
    linkage_recall = f(cum[k] / max(1, y_pair.sum()))

    print("clustering...")
    g = build_graph(df, pairs, edge_threshold)
    df["cluster_id"] = assign_clusters(df, g)
    feats = cluster_features(df, pairs)
    df = df.merge(feats.drop(columns=["cluster_id"]), on="app_seq", how="left")
    purity = cluster_purity(df)

    print("scoring...")
    X = build_features(df, pairs, edge_threshold=edge_threshold)
    d = df.merge(X, on="app_seq", suffixes=("", "_f"))
    tr = d[d.app_seq.isin(tr_seq)]
    te = d[d.app_seq.isin(te_seq)]
    model = RiskModel().fit(tr, tr.is_evasion.values)
    d["risk"] = model.predict_proba(d)

    p_te = model.predict_proba(te)
    y_te = te.is_evasion.values
    econ_review, t_block = cost_optimal_thresholds(COST_FP, COST_FN, COST_REVIEW)
    blocked_frac = float((d.risk >= t_block).mean())
    t_review = min(
        t_block,
        float(np.quantile(d.risk, max(0.0, 1 - blocked_frac - REVIEW_CAPACITY))),
    )

    # ---- metrics ----------------------------------------------------------
    # A ring's level is a property of the operator, not of each row: the origin
    # application carries level 0, so grouping by level first would split every
    # ring in half and report roughly a quarter of the rings that exist.
    by_level: dict[int, list[tuple[float, bool]]] = {}
    for _op, ring in d[d.cohort.str.startswith("ring")].groupby("operator_id"):
        if len(ring) < 2:
            continue
        lvl = int(ring.evasion_level.max())
        if lvl == 0:
            continue
        biggest = ring.cluster_id.value_counts().iloc[0]
        by_level.setdefault(lvl, []).append((biggest / len(ring), biggest == len(ring)))

    evasion_curve = [
        {
            "level": lvl,
            "rings": len(rows),
            "meanRecovered": f(np.mean([r[0] for r in rows]) * 100),
            "fullyRecovered": f(np.mean([r[1] for r in rows]) * 100),
        }
        for lvl, rows in sorted(by_level.items())
    ]

    sweep = []
    for step in [-4, -2, 0, 2, 4]:
        thr = edge_threshold + step
        gg = build_graph(df, pairs, thr)
        tmp = df.copy()
        tmp["cluster_id"] = assign_clusters(tmp, gg)
        pu = cluster_purity(tmp)
        kept = pairs[pairs.edge_score >= thr]
        sweep.append({
            "threshold": f(thr),
            "edges": int(len(kept)),
            "precision": f(kept.same_operator.mean() if len(kept) else 0),
            "recall": f(kept.same_operator.sum() / max(1, y_pair.sum())),
            "purity": f(pu["mean_purity"]),
            "contaminated": int(pu["contaminated_clusters"]),
        })

    cost_rows = []
    for ratio in [2, 5, 12, 30, 60]:
        _r, tb = cost_optimal_thresholds(COST_FP, COST_FP * ratio, COST_REVIEW)
        blocked = p_te >= tb
        cost_rows.append({
            "ratio": ratio,
            "threshold": f(tb),
            "blocked": int(blocked.sum()),
            "recall": f((blocked & (y_te == 1)).sum() / max(1, y_te.sum()) * 100),
            "fpRate": f((blocked & (y_te == 0)).sum() / max(1, (y_te == 0).sum()) * 100),
        })

    queue = []
    for n in [10, 25, 50, 100, 200]:
        pr, rc = precision_at_n(y_te, p_te, n)
        queue.append({"n": n, "precision": f(pr * 100), "recall": f(rc * 100)})

    cal = [
        {"predicted": f(r.mean_predicted), "observed": f(r.observed), "n": int(r.n)}
        for r in calibration_table(y_te, p_te).itertuples()
    ]

    fs_w = [
        {"field": r.field, "m": f(r.m), "u": f(r.u),
         "agree": f(r.agree_weight), "disagree": f(r.disagree_weight)}
        for r in fs.weights().itertuples()
    ]
    coefs = [
        {"feature": r.feature, "coefficient": f(r.coefficient)}
        for r in model.coefficients().itertuples()
    ]

    # ---- per-application payload -------------------------------------------
    print("assembling records...")
    d = d.sort_values("risk", ascending=False).reset_index(drop=True)
    idx_by_seq = {int(s): i for i, s in enumerate(d.app_seq)}

    apps = []
    for r in d.itertuples():
        apps.append({
            "id": r.app_id,
            "name": r.name,
            "business": r.business_name,
            "pan": r.pan,
            "gstin": r.gstin,
            "phone": str(r.phone),
            "account": str(r.settlement_account),
            "ifsc": r.settlement_ifsc,
            "address": r.address,
            "city": r.city,
            "pincode": str(r.pincode),
            "email": r.email,
            "date": str(r.app_date),
            "amount": int(r.requested_amount),
            "status": r.status,
            "mcc": str(r.mcc),
            "cluster": int(r.cluster_id),
            "cohort": r.cohort,
            "risk": f(r.risk),
            "isEvasion": bool(r.is_evasion),
            "level": int(r.evasion_level),
            "priorRejections": int(r.prior_rejections),
            "clusterSize": int(r.cluster_size_so_far),
            "breadth": f(r.alteration_breadth),
            "bestLink": f(r.best_link_score),
            "hasPrior": int(r.has_prior),
            "split": "test" if r.app_seq in te_seq else "train",
        })

    # clusters with more than one member, plus their edges
    seq_to_id = dict(zip(d.app_seq, d.app_id))
    clusters = {}
    sizes = d.groupby("cluster_id").size()
    for cid in sizes[sizes > 1].index:
        members = d[d.cluster_id == cid].sort_values("date" if "date" in d else "app_date")
        clusters[str(int(cid))] = {
            "members": list(members.app_id),
            "edges": [],
        }
    linked_pairs = pairs[pairs.edge_score >= edge_threshold]
    for e in linked_pairs.itertuples():
        a, b = seq_to_id.get(e.seq_a), seq_to_id.get(e.seq_b)
        if a is None or b is None:
            continue
        row = d[d.app_id == a]
        if row.empty:
            continue
        cid = str(int(row.iloc[0].cluster_id))
        if cid not in clusters:
            continue
        top = ("PAN" if e.pan_exact else "GSTIN" if e.gstin_pan_cross
               else "account" if e.account_exact else "phone" if e.phone_exact
               else "name/address")
        clusters[cid]["edges"].append(
            {"a": a, "b": b, "score": f(e.edge_score), "via": top}
        )

    # strongest prior link and model contributions, for linked records only
    seq_date = df.set_index("app_seq")["app_date"].to_dict()
    lp = linked_pairs.assign(this=linked_pairs.seq_b, other=linked_pairs.seq_a)
    lp = lp[lp["other"].map(seq_date) <= lp["this"].map(seq_date)]
    details = {}
    if len(lp):
        best = lp.loc[lp.groupby("this")["edge_score"].idxmax()].set_index("this")
        for seq, row in best.iterrows():
            app_id = seq_to_id.get(seq)
            other_id = seq_to_id.get(row.seq_a if row.seq_b == seq else row.seq_b)
            if app_id is None or other_id is None:
                continue
            fields = [
                {"label": label, "value": f(row[col]), "match": bool(float(row[col]) >= 0.85)}
                for col, label in FIELD_LABELS
            ]
            i = idx_by_seq.get(int(seq))
            contribs = []
            if i is not None:
                for e in model.explain(d, i, top=8).itertuples():
                    contribs.append({
                        "feature": e.feature,
                        "value": f(e.value),
                        "contribution": f(e.contribution),
                    })
            details[app_id] = {
                "linkedTo": other_id,
                "bits": f(row.edge_score),
                "fields": fields,
                "contributions": contribs,
            }

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": int(len(d)),
            "evasionRate": f(d.is_evasion.mean() * 100),
            "positives": int(d.is_evasion.sum()),
            "edgeThreshold": edge_threshold,
            "blockThreshold": f(t_block),
            "reviewThreshold": f(t_review),
            "economicReviewThreshold": f(econ_review),
            "costFp": COST_FP,
            "costFn": COST_FN,
            "costReview": COST_REVIEW,
            "trainSize": int(len(tr)),
            "testSize": int(len(te)),
        },
        "metrics": {
            "testAP": f(average_precision_score(y_te, p_te)),
            "testAUC": f(roc_auc_score(y_te, p_te)),
            "linkageRecall": linkage_recall,
            "clusterPurity": f(purity["mean_purity"]),
            "contaminated": int(purity["contaminated_clusters"]),
            "multiClusters": int(purity["n_multi_clusters"]),
            "candidatePairs": int(len(pairs)),
            "evasionCurve": evasion_curve,
            "thresholdSweep": sweep,
            "costSensitivity": cost_rows,
            "queueBudget": queue,
            "calibration": cal,
            "fsWeights": fs_w,
            "coefficients": coefs,
        },
        "applications": apps,
        "clusters": clusters,
        "details": details,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({mb:.1f} MB, {len(apps)} applications, "
          f"{len(clusters)} clusters, {len(details)} detail records)")


if __name__ == "__main__":
    main()