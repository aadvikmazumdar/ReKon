#!/usr/bin/env python3
"""ReKon demo: evidence reports and ring visualisations.

    python -m rekon.demo                 highest-risk cases plus contrasts
    python -m rekon.demo --app APP001234 one specific application
    python -m rekon.demo --svg out.svg   also render the ring graph

The output is deliberately not a score. A risk system that says "FRAUD 0.87"
gives an underwriter nothing to act on or argue with. This prints the evidence,
the cluster history, and what each factor contributed, then recommends.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import build_pair_table
from .fellegi_sunter import FellegiSunter, agreement_matrix
from .graph import assign_clusters, build_graph, cluster_features, edge_score
from .risk import RiskModel, build_features, cost_optimal_thresholds
from .split import operator_split

DATA = Path(__file__).parent / "data"
RULE = "-" * 68

MATCH_FIELDS = [
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
]


def load(edge_threshold: float | None = None):
    df = pd.read_csv(DATA / "applications.csv")
    pairs = build_pair_table(df)
    pairs["hand_score"] = edge_score(pairs)

    train, _ = operator_split(df)
    tr_seq = set(train.app_seq)
    tr_pairs = pairs[pairs.seq_a.isin(tr_seq) & pairs.seq_b.isin(tr_seq)]
    fs = FellegiSunter().fit(agreement_matrix(tr_pairs))
    pairs["edge_score"] = fs.score(agreement_matrix(pairs))

    thr = 0.0 if edge_threshold is None else edge_threshold
    g = build_graph(df, pairs, thr)
    df["cluster_id"] = assign_clusters(df, g)
    feats = cluster_features(df, pairs)
    df = df.merge(feats.drop(columns=["cluster_id"]), on="app_seq", how="left")

    X = build_features(df, pairs, edge_threshold=thr)
    d = df.merge(X, on="app_seq", suffixes=("", "_f"))
    tr = d[d.app_seq.isin(tr_seq)]
    model = RiskModel().fit(tr, tr.is_evasion.values)
    d["risk"] = model.predict_proba(d)
    return d, pairs, model


def evidence_report(d: pd.DataFrame, pairs: pd.DataFrame, model: RiskModel,
                    app_id: str, t_review: float, t_block: float,
                    known_false_positive: bool = False) -> None:
    row = d[d.app_id == app_id]
    if row.empty:
        print(f"  no such application: {app_id}")
        return
    i = row.index[0]
    r = row.iloc[0]

    risk = float(r["risk"])
    action = "BLOCK" if risk >= t_block else ("REVIEW" if risk >= t_review else "PASS")

    print(RULE)
    print(f"  APPLICATION {app_id}      RISK {risk:.3f}      -> {action}")
    print(RULE)
    print(f"  {r['name']}  |  {r['business_name']}")
    print(f"  PAN {r['pan']}   GSTIN {r['gstin']}   phone {r['phone']}")
    print(f"  {r['address']}, {r['city']} {r['pincode']}")
    print(f"  applied {r['app_date']}   requesting Rs{int(r['requested_amount']):,}")

    cluster = d[d.cluster_id == r["cluster_id"]].sort_values("app_date")
    print(f"\n  LINKED GROUP: {len(cluster)} applications")
    for c in cluster.itertuples():
        mark = " <- this" if c.app_id == app_id else ""
        print(f"    {c.app_date}  {c.app_id}  {str(c.name)[:22]:<22} "
              f"{str(c.status):<12}{mark}")

    prior = cluster[cluster.app_date < r["app_date"]]
    bad = prior[prior.status.isin(["rejected", "blocklisted"])]
    print(f"\n  HISTORY: {len(prior)} earlier application(s), "
          f"{len(bad)} previously rejected or blocklisted")

    mine = pairs[((pairs.seq_a == r["app_seq"]) | (pairs.seq_b == r["app_seq"]))]
    mine = mine[mine.edge_score >= 0].sort_values("edge_score", ascending=False)
    if len(mine):
        link = mine.iloc[0]
        other_seq = link.seq_a if link.seq_b == r["app_seq"] else link.seq_b
        other = d[d.app_seq == other_seq].iloc[0]
        print(f"\n  STRONGEST LINK: {other['app_id']} "
              f"({link.edge_score:.1f} bits of evidence)")
        for col, label in MATCH_FIELDS:
            v = float(link[col])
            if v >= 0.85:
                print(f"    match    {label:<38} {v:.2f}")
        for col, label in MATCH_FIELDS:
            v = float(link[col])
            if v < 0.85:
                print(f"    differs  {label:<38} {v:.2f}")

    print("\n  WHAT DROVE THE SCORE")
    for e in model.explain(d, i).itertuples():
        direction = "raises" if e.contribution > 0 else "lowers"
        print(f"    {direction:<7} {e.feature:<24} value {e.value:>8.2f}   "
              f"{e.contribution:+.2f}")

    # The wording must follow the evidence actually present. Asserting an
    # "altered identity" over a record that matches on every field is worse
    # than a wrong score: it is a wrong reason, and an underwriter cannot
    # act on it.
    breadth = float(r.get("alteration_breadth", 0.0))
    n_bad = len(bad)
    print(f"\n  RECOMMENDATION: {action}")
    if known_false_positive:
        print(f"    KNOWN FALSE POSITIVE. {int(breadth)} field(s) differ, which is")
        print("    within the range an ordinary applicant produces between two")
        print("    applications months apart. The evidence here cannot distinguish")
        print("    a light disguise from a changed phone number, so ReKon blocks a")
        print("    legitimate merchant. See REPORT section 5.")
    elif n_bad == 0:
        print("    No prior rejection anywhere in this group. Linkage alone is not")
        print("    risk: the same person may legitimately apply more than once.")
    elif breadth == 0:
        print(f"    Linked to {n_bad} prior rejection(s), but every compared field")
        print("    matches: nothing about this identity has been altered. This is")
        print("    consistent with an open re-application rather than evasion, and")
        print("    is exactly the band ReKon cannot resolve on evidence alone.")
        if action == "BLOCK":
            print("    FLAGGED AS A KNOWN FALSE POSITIVE -- see REPORT section 5.")
    else:
        print(f"    Linked to {n_bad} prior rejection(s), and {int(breadth)} identity")
        print("    field(s) differ from the linked record. Concealment is the signal,")
        print("    not the linkage itself.")
    print()


def ring_svg(d: pd.DataFrame, pairs: pd.DataFrame, cluster_id: int, path: Path) -> None:
    members = d[d.cluster_id == cluster_id].sort_values("app_date").reset_index(drop=True)
    n = len(members)
    if n < 2:
        return
    w, h, pad = 760, 420, 90
    cx, cy, rad = w / 2, h / 2, min(w, h) / 2 - pad
    pos = {
        row.app_seq: (
            cx + rad * np.cos(2 * np.pi * i / n - np.pi / 2),
            cy + rad * np.sin(2 * np.pi * i / n - np.pi / 2),
        )
        for i, row in enumerate(members.itertuples())
    }
    seqs = set(pos)
    edges = pairs[
        pairs.seq_a.isin(seqs) & pairs.seq_b.isin(seqs) & (pairs.edge_score >= 0)
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        '<text x="20" y="30" font-family="monospace" font-size="15" '
        f'fill="#111">ReKon ring {cluster_id} - {n} applications, '
        f'{int(members.status.isin(["rejected","blocklisted"]).sum())} previously flagged</text>',
    ]
    for e in edges.itertuples():
        x1, y1 = pos[e.seq_a]
        x2, y2 = pos[e.seq_b]
        parts.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="#b0b6c0" stroke-width="{1 + min(3, e.edge_score / 30):.1f}"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label = ("PAN" if e.pan_exact else "GSTIN" if e.gstin_pan_cross
                 else "account" if e.account_exact else "phone" if e.phone_exact
                 else "name/addr")
        parts.append(
            f'<text x="{mx:.0f}" y="{my:.0f}" font-family="monospace" '
            f'font-size="10" fill="#6b7280" text-anchor="middle">{label}</text>'
        )
    for row in members.itertuples():
        x, y = pos[row.app_seq]
        flagged = row.status in ("rejected", "blocklisted")
        fill = "#c0392b" if flagged else "#2d6a4f"
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="11" fill="{fill}"/>')
        parts.append(
            f'<text x="{x:.0f}" y="{y - 20:.0f}" font-family="monospace" '
            f'font-size="11" fill="#111" text-anchor="middle">'
            f'{str(row.name)[:18]}</text>'
        )
        parts.append(
            f'<text x="{x:.0f}" y="{y + 28:.0f}" font-family="monospace" '
            f'font-size="9" fill="#6b7280" text-anchor="middle">'
            f'{row.app_date} {row.status}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"  ring diagram written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=str, default=None)
    ap.add_argument("--svg", type=str, default=None)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    d, pairs, model = load()
    _econ, t_block = cost_optimal_thresholds(25_000, 300_000, 500)
    blocked_frac = float((d["risk"] >= t_block).mean())
    t_review = min(t_block, float(np.quantile(d["risk"], max(0.0, 1 - blocked_frac - 0.03))))

    if args.app:
        evidence_report(d, pairs, model, args.app, t_review, t_block)
    else:
        print("\n### HIGHEST-RISK APPLICATIONS\n")
        for app in d.nlargest(args.n, "risk").app_id:
            evidence_report(d, pairs, model, app, t_review, t_block)

        print("\n### CONTRAST: STRONGLY LINKED, CORRECTLY NOT BLOCKED\n")
        for cohort in ["hn_multi_founder", "hn_rebrand", "hn_shared_account"]:
            sub = d[(d.cohort == cohort) & (d.has_prior == 1) & (d.risk < t_block)]
            if len(sub):
                pick = sub.nlargest(1, "best_link_score").iloc[0]
                evidence_report(d, pairs, model, pick.app_id, t_review, t_block)

        print("\n### KNOWN FAILURE MODE: the ambiguity band\n")
        print("  Honest re-application after a rejection is not separable from")
        print("  low-level evasion on observable evidence. Both keep PAN and")
        print("  settlement account and alter one or two soft fields. ReKon blocks")
        print("  50% of this cohort at the cost-optimal threshold. The case below")
        print("  is one it gets wrong, shown rather than hidden.\n")
        fp = d[(d.cohort == "hn_honest_reapply") & (d.has_prior == 1)
               & (d.risk >= t_block)]
        if len(fp):
            evidence_report(d, pairs, model, fp.nlargest(1, "risk").iloc[0].app_id,
                            t_review, t_block, known_false_positive=True)

    if args.svg:
        ring = d[d.cohort.str.startswith("ring")]
        sizes = ring.groupby("cluster_id").size()
        big = sizes[sizes >= 3]
        if len(big):
            ring_svg(d, pairs, int(big.index[0]), Path(args.svg))


if __name__ == "__main__":
    main()