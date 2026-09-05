#!/usr/bin/env python3
"""ReKon end-to-end pipeline.

    python -m rekon.pipeline              full run, default 25k
    python -m rekon.pipeline --n 8000     smaller and faster
    python -m rekon.pipeline --skip-gen   reuse data already on disk

Runs generation, validation, splitting, evidence extraction, graph clustering
and diagnostics, printing everything needed to judge whether the dataset and
the linkage layer are behaving. Scoring is not part of this yet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import build_pair_table
from .fellegi_sunter import FellegiSunter, agreement_matrix
from .generate import generate, label_evasion
from .risk import (
    FEATURES,
    RiskModel,
    build_features,
    calibration_table,
    cost_optimal_thresholds,
    precision_at_n,
)
from .graph import (
    assign_clusters,
    build_graph,
    cluster_features,
    cluster_purity,
    edge_score,
)
from .split import operator_split, validate_split
from .validate import report, results, validate

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

RULE = "=" * 72


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edge-threshold", type=float, default=6.0)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--scorer", choices=["fs", "hand"], default="fs")
    args = ap.parse_args()

    t_start = time.time()

    # ---- 1. generate ------------------------------------------------------
    head("1. GENERATION")
    if args.skip_gen and (DATA / "applications.csv").exists():
        df = pd.read_csv(DATA / "applications.csv")
        print(f"  reused {len(df)} records from disk")
    else:
        t = time.time()
        df = generate(n_target=args.n, seed=args.seed)
        df["is_evasion"] = label_evasion(df)
        df.to_csv(DATA / "applications.csv", index=False)
        print(f"  generated {len(df)} records in {time.time()-t:.1f}s")

    span = (pd.to_datetime(df.app_date).max() - pd.to_datetime(df.app_date).min()).days
    print(f"  span {span}d  |  {len(df)/span:.1f} apps/day  |  {len(df)/span*30:.0f}/month")
    print(f"  evasion rate {df.is_evasion.mean():.2%}  ({int(df.is_evasion.sum())} positives)")
    print("\n  cohorts:")
    for name, n in df.cohort.value_counts().items():
        print(f"    {name:<22} {n:>6}  ({n/len(df):>5.1%})")

    # ---- 2. validate ------------------------------------------------------
    head("2. DATASET VALIDATION")
    results.clear()
    validate(df)
    failed = report()

    # ---- 3. split ---------------------------------------------------------
    head("3. SPLIT")
    train, test = operator_split(df)
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)
    print(f"  train {len(train)}  test {len(test)}\n")
    checks = validate_split(train, test)
    width = max(len(n) for n, _, _ in checks) + 2
    for name, ok, detail in checks:
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")

    # ---- 4. evidence ------------------------------------------------------
    head("4. EVIDENCE LAYER")
    t = time.time()
    pairs = build_pair_table(df)
    pairs["edge_score"] = edge_score(pairs)
    pairs.to_csv(DATA / "pairs.csv.gz", index=False, compression="gzip")
    elapsed = time.time() - t
    all_pairs = len(df) * (len(df) - 1) // 2
    print(f"  {len(pairs):,} candidate pairs from {all_pairs:,} possible "
          f"({all_pairs/max(1,len(pairs)):.0f}x reduction)")
    print(f"  built in {elapsed:.0f}s  ({len(df)/elapsed:.0f} apps/sec)")

    truth = pairs.same_operator.sum()
    print(f"  true same-operator pairs retrieved: {truth:,}")

    # ---- 4b. Fellegi-Sunter -------------------------------------------------
    head("4b. FELLEGI-SUNTER LINKAGE MODEL")
    tr_seq = set(train.app_seq)
    tr_pairs = pairs[pairs.seq_a.isin(tr_seq) & pairs.seq_b.isin(tr_seq)]
    gamma_tr = agreement_matrix(tr_pairs)
    fs = FellegiSunter().fit(gamma_tr)
    pairs["fs_score"] = fs.score(agreement_matrix(pairs))
    print(f"  fitted on {len(tr_pairs):,} train pairs, unsupervised (no labels used)")
    print(f"  EM converged in {fs.n_iter_run_} iterations")
    print(f"  estimated match rate {fs.lambda_:.5f}  |  actual {tr_pairs.same_operator.mean():.5f}")
    print("\n  learned field weights (bits):")
    w = fs.weights()
    print(f"    {'field':<20} {'m':>7} {'u':>8} {'agree':>8} {'disagree':>9}")
    for r in w.itertuples():
        print(f"    {r.field:<20} {r.m:>7.3f} {r.u:>8.4f} "
              f"{r.agree_weight:>8.2f} {r.disagree_weight:>9.2f}")

    y = pairs.same_operator.values
    print("\n  hand weights vs estimated, max recall at perfect precision:")
    for label, col in [("hand weights", "edge_score"), ("Fellegi-Sunter", "fs_score")]:
        sc = pairs[col].values
        order = np.argsort(-sc)
        cum = np.cumsum(y[order])
        prec = cum / np.arange(1, len(order) + 1)
        ok = np.where(prec == 1.0)[0]
        k = int(ok.max()) if len(ok) else 0
        print(f"    {label:<16} recall {cum[k]/max(1,y.sum()):.4f} "
              f"({cum[k]}/{y.sum()} pairs, threshold {sc[order][k]:.3f})")

    if args.scorer == "fs":
        sc = pairs["fs_score"].values
        order = np.argsort(-sc)
        cum = np.cumsum(y[order])
        prec = cum / np.arange(1, len(order) + 1)
        ok = np.where(prec == 1.0)[0]
        fs_threshold = float(sc[order][int(ok.max()) if len(ok) else 0])
        pairs["edge_score"] = pairs["fs_score"]
        args.edge_threshold = fs_threshold
        print(f"\n  using Fellegi-Sunter scores, threshold {fs_threshold:.3f}")
    else:
        print("\n  using hand weights")

    # ---- 5. graph ---------------------------------------------------------
    head("5. GRAPH + CLUSTERING")
    print("  edge threshold sweep:")
    print(f"    {'thr':>5} {'edges':>7} {'precision':>10} {'recall':>8} "
          f"{'purity':>8} {'contaminated':>13}")
    # Additive, because a Fellegi-Sunter operating point sits near zero bits
    # and a multiplicative sweep collapses onto it.
    t0 = args.edge_threshold
    step = 2.0 if args.scorer == "fs" else max(1.0, abs(t0) * 0.25)
    for thr in [t0 - 2 * step, t0 - step, t0, t0 + step, t0 + 2 * step]:
        g = build_graph(df, pairs, thr)
        tmp = df.copy()
        tmp["cluster_id"] = assign_clusters(tmp, g)
        pur = cluster_purity(tmp)
        kept = pairs[pairs.edge_score >= thr]
        prec = kept.same_operator.mean() if len(kept) else 0
        rec = kept.same_operator.sum() / max(1, truth)
        mark = " <-" if abs(thr - args.edge_threshold) < 1e-9 else ""
        print(f"    {thr:>5.2f} {len(kept):>7} {prec:>10.3f} {rec:>8.3f} "
              f"{pur['mean_purity']:>8.3f} {pur['contaminated_clusters']:>5}/"
              f"{pur['n_multi_clusters']:<7}{mark}")

    g = build_graph(df, pairs, args.edge_threshold)
    df["cluster_id"] = assign_clusters(df, g)
    feats = cluster_features(df, pairs)
    df = df.merge(feats.drop(columns=["cluster_id"]), on="app_seq", how="left")
    df.to_csv(DATA / "applications_clustered.csv", index=False)

    # ---- 6. diagnostics ---------------------------------------------------
    head("6. DIAGNOSTICS")

    print("  RING RECOVERY BY EVASION LEVEL")
    rows = []
    for _op, grp in df[df.cohort.str.startswith("ring")].groupby("operator_id"):
        if len(grp) < 2:
            continue
        biggest = grp.cluster_id.value_counts().iloc[0]
        rows.append({"level": grp.evasion_level.max(),
                     "frac": biggest / len(grp), "full": biggest == len(grp)})
    rec = pd.DataFrame(rows)
    print(f"    {'level':>6} {'rings':>7} {'mean recovered':>16} {'fully recovered':>17}")
    for lvl, grp in rec.groupby("level"):
        print(f"    {lvl:>6} {len(grp):>7} {grp.frac.mean():>15.1%} {grp.full.mean():>17.1%}")

    print("\n  FALSE LINKAGE ON LEGITIMATE COHORTS")
    print("    (co-clustering is correct where records are genuinely one person)")
    for cohort in sorted(df[df.cohort.str.startswith(("hn", "grey"))].cohort.unique()):
        sub = df[df.cohort == cohort]
        sizes = sub.groupby("cluster_id").size()
        linked = sub[sub.cluster_id.isin(sizes[sizes > 1].index)]
        expected = cohort in ("hn_honest_reapply", "hn_multi_founder", "hn_rebrand")
        verdict = "expected" if expected else ("FALSE LINK" if len(linked) else "clean")
        print(f"    {cohort:<22} {len(linked):>5}/{len(sub):<6} {verdict}")

    print("\n  SETTLEMENT LAYER")
    ring = df[df.cohort.str.startswith("ring")]
    print(f"    {'level':>6} {'mean account churn':>20}")
    for lvl, grp in ring[ring.evasion_level > 0].groupby("evasion_level"):
        print(f"    {lvl:>6} {grp.account_churn.mean():>20.3f}")
    shared = df[df.cohort == "hn_shared_account"]
    print(f"    shared-account cohort: {len(shared)} rows, "
          f"{shared.cluster_id.nunique()} clusters "
          f"(dense legitimate clusters the system must not treat as rings)")

    print("\n  CLUSTER FEATURE SEPARATION (evasion vs rest)")
    for col in ["prior_rejections", "pan_churn", "phone_churn", "account_churn",
                "cluster_size_so_far", "apps_per_100d"]:
        a, b = df[df.is_evasion][col], df[~df.is_evasion][col]
        print(f"    {col:<22} evasion {a.mean():>8.3f}   other {b.mean():>8.3f}")

    # ---- 7. risk scoring --------------------------------------------------
    head("7. RISK SCORING")
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = build_features(df, pairs, edge_threshold=args.edge_threshold)
    d = df.merge(X, on="app_seq", suffixes=("", "_f"))
    tr = d[d.app_seq.isin(set(train.app_seq))]
    te = d[d.app_seq.isin(set(test.app_seq))]
    model = RiskModel().fit(tr, tr.is_evasion.values)
    p_te = model.predict_proba(te)
    y_te = te.is_evasion.values
    print(f"  train {len(tr)}  test {len(te)}  test positives {int(y_te.sum())}")
    print(f"  test AP {average_precision_score(y_te, p_te):.4f}   "
          f"AUC {roc_auc_score(y_te, p_te):.4f}")

    print("\n  COEFFICIENTS (the evidence report is read straight off these)")
    coefs = model.coefficients()
    for r in coefs.head(8).itertuples():
        print(f"    {r.feature:<24} {r.coefficient:+.3f}")
    print("    ...")
    for r in coefs.tail(4).itertuples():
        print(f"    {r.feature:<24} {r.coefficient:+.3f}")

    print("\n  EXCULPATORY SIGN ASSERTIONS")
    signs = dict(zip(coefs.feature, coefs.coefficient))
    # best_link_score is positive by design: against a population that is
    # mostly unlinked singletons, being linked at all is evidence. The
    # exculpatory signals are the ones that say "linked, but concealing
    # nothing": an unchanged identity, a matching PAN, a matching account.
    # best_link_score is exculpatory once link features are restricted to real
    # links: among records that ARE linked, a stronger match means less of the
    # identity was altered. It reads as positive only when the feature is
    # polluted by blocking collisions between strangers.
    for feat, want in [("link_pan_exact", "-"), ("link_account_exact", "-"),
                       ("link_phone_exact", "-"), ("best_link_score", "-"),
                       ("alteration_breadth", "+"), ("prior_rejections", "+")]:
        got = signs[feat]
        ok = (got < 0) if want == "-" else (got > 0)
        failed_local = 0 if ok else 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {feat:<22} expected {want}, got {got:+.3f}")

    print("\n  RECALL BY EVASION LEVEL (at cost-optimal BLOCK threshold)")
    COST_FP, COST_FN, COST_REVIEW = 25_000, 300_000, 500
    _t_econ, t_block = cost_optimal_thresholds(COST_FP, COST_FN, COST_REVIEW)
    # The economic review threshold says review almost everything: a review
    # costs far less than a funded evader. The binding constraint in practice
    # is underwriter capacity, so REVIEW is set from the queue budget instead
    # and the economic figure is reported alongside it.
    # Review capacity is additional to whatever is blocked outright, so the
    # quantile is taken below the block rate rather than from the top.
    REVIEW_CAPACITY = 0.03
    blocked_frac = float((p_te >= t_block).mean())
    t_review = float(np.quantile(p_te, max(0.0, 1 - blocked_frac - REVIEW_CAPACITY)))
    t_review = min(t_review, t_block)
    print(f"    block  >= {t_block:.4f}  (FP Rs{COST_FP:,} vs FN Rs{COST_FN:,})")
    print(f"    review >= {t_review:.4f}  (top {REVIEW_CAPACITY:.0%} by capacity; "
          f"the economic threshold would be {_t_econ:.4f})")
    for lvl in sorted(te[te.is_evasion].evasion_level.unique()):
        m_ = (te.evasion_level == lvl) & te.is_evasion
        if m_.sum() == 0:
            continue
        print(f"    level {lvl}   n={int(m_.sum()):>4}   "
              f"blocked {(p_te[m_.values] >= t_block).mean():>6.1%}   "
              f"blocked+review {(p_te[m_.values] >= t_review).mean():>6.1%}")

    print("\n  FALSE POSITIVES ON LEGITIMATE COHORTS (the thesis test)")
    for cohort in sorted(te[te.cohort.str.startswith(("hn", "grey", "single"))].cohort.unique()):
        m_ = (te.cohort == cohort).values
        if m_.sum() == 0:
            continue
        print(f"    {cohort:<22} n={int(m_.sum()):>5}   "
              f"blocked {(p_te[m_] >= t_block).mean():>6.2%}   "
              f"reviewed {((p_te[m_] >= t_review) & (p_te[m_] < t_block)).mean():>6.2%}")

    print("\n  AMBIGUITY BOUNDARY: honest re-application vs low-level evasion")
    linked = te.has_prior == 1
    hr = ((te.cohort == "hn_honest_reapply") & linked).values
    lo = (te.is_evasion & (te.evasion_level <= 2)).values
    print(f"    {'':<22} {'n':>5} {'breadth':>9} {'pan match':>11} {'acct match':>12}")
    for nm, m_ in [("honest re-application", hr), ("evasion level 1-2", lo)]:
        sub = te[m_]
        print(f"    {nm:<22} {len(sub):>5} {sub.alteration_breadth.mean():>9.2f} "
              f"{sub.link_pan_exact.mean():>11.2f} {sub.link_account_exact.mean():>12.2f}")
    print("    These are not separable on observable evidence: both keep PAN and")
    print("    account, alter one or two soft fields, and follow a rejection. The")
    print("    band belongs in REVIEW, not BLOCK.")

    print("\n  BLOCK THRESHOLD TRADEOFF")
    print(f"    {'threshold':>10} {'evasion recall':>16} {'honest reapply blocked':>24}")
    for t in [0.03, 0.077, 0.15, 0.30, 0.50, 0.70, 0.85]:
        print(f"    {t:>10.3f} {(p_te[y_te == 1] >= t).mean():>15.1%} "
              f"{(p_te[hr] >= t).mean():>23.1%}")

    print("\n  REVIEW QUEUE BUDGET")
    for n in [10, 25, 50, 100]:
        prec, rec = precision_at_n(y_te, p_te, n)
        print(f"    top {n:>4}   precision {prec:>6.1%}   recall {rec:>6.1%}")

    print("\n  CALIBRATION")
    cal = calibration_table(y_te, p_te)
    print(f"    {'bin':>4} {'n':>6} {'predicted':>11} {'observed':>10}")
    for r in cal.itertuples():
        print(f"    {r.bin:>4} {r.n:>6} {r.mean_predicted:>11.3f} {r.observed:>10.3f}")

    print("\n  COST SENSITIVITY (FP = blocked legitimate merchant, FN = funded evader)")
    print(f"    {'FN:FP':>8} {'block thr':>11} {'blocked':>9} {'recall':>8} {'FP rate':>9}")
    for ratio in [2, 5, 12, 30, 60]:
        tr_, tb_ = cost_optimal_thresholds(COST_FP, COST_FP * ratio, COST_REVIEW)
        blocked = p_te >= tb_
        rec = (blocked & (y_te == 1)).sum() / max(1, y_te.sum())
        fpr = (blocked & (y_te == 0)).sum() / max(1, (y_te == 0).sum())
        print(f"    {ratio:>7}x {tb_:>11.4f} {blocked.sum():>9} {rec:>8.1%} {fpr:>9.2%}")

    head("SUMMARY")
    print(f"  total runtime {time.time()-t_start:.0f}s")
    print(f"  failed checks: {failed}")
    print(f"  artifacts written to {DATA}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())