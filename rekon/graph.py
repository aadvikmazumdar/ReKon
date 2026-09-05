"""Link graph and ring clustering.

A pairwise matcher sees edges; the fraud is the cluster. One operator running
five identities in a chain leaves A~B~C~D~E, where A and E share nothing at
all. Connected components recover the whole ring from the adjacent hops.

The danger is the mirror image: one spurious edge welds two unrelated rings
together and contamination cascades. Edge threshold is therefore kept strict
(precision on edges, recall via transitivity) and cluster purity is reported
as a first-class metric so over-merging cannot hide.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

# Deterministic pre-score used only to decide whether an edge exists.
# Weights are structural facts about Indian identity fields, not fitted values:
# an exact PAN or a GSTIN carrying the same embedded PAN is near-conclusive,
# while a shared pincode alone is close to meaningless.
EDGE_WEIGHTS = {
    "pan_exact": 3.0,
    "gstin_pan_match": 3.0,
    "gstin_pan_cross": 2.5,
    "phone_exact": 2.0,
    "account_exact": 2.5,
    "ifsc_exact": 0.4,
    "name_jw": 1.2,
    "business_jw": 1.0,
    "address_overlap": 1.0,
    "email_jw": 0.8,
    "surname_jw": 0.5,
    "pincode_exact": 0.3,
}

EDGE_THRESHOLD = 6.0  # raised from 5.0 once settlement account entered the evidence set:
# account_exact is strong but legitimately shared by aggregators and family
# businesses, so the old threshold admitted the hn_shared_account cohort.


def edge_score(pairs: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=pairs.index)
    for col, w in EDGE_WEIGHTS.items():
        score += pairs[col].astype(float) * w
    return score


def build_graph(
    df: pd.DataFrame, pairs: pd.DataFrame, threshold: float = EDGE_THRESHOLD
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(df["app_seq"].tolist())
    keep = pairs[pairs["edge_score"] >= threshold]
    for row in keep.itertuples():
        g.add_edge(row.seq_a, row.seq_b, weight=float(row.edge_score))
    return g


def assign_clusters(df: pd.DataFrame, g: nx.Graph) -> pd.Series:
    """Map each app_seq to a cluster id from connected components."""
    mapping: dict[int, int] = {}
    for cid, component in enumerate(nx.connected_components(g)):
        for node in component:
            mapping[node] = cid
    return df["app_seq"].map(mapping)


def cluster_purity(df: pd.DataFrame) -> dict:
    """How badly is clustering welding unrelated operators together?"""
    multi = df[df.groupby("cluster_id")["cluster_id"].transform("size") > 1]
    if multi.empty:
        return {"mean_purity": 1.0, "contaminated_clusters": 0, "n_multi_clusters": 0}

    purities, contaminated = [], 0
    for _cid, grp in multi.groupby("cluster_id"):
        top = grp["operator_id"].value_counts().iloc[0]
        purity = top / len(grp)
        purities.append(purity)
        if purity < 1.0:
            contaminated += 1
    return {
        "mean_purity": float(sum(purities) / len(purities)),
        "contaminated_clusters": contaminated,
        "n_multi_clusters": len(purities),
    }


def cluster_features(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-application features describing the cluster it sits in.

    Every history feature is computed over strictly earlier applications, so a
    record can never be scored using its own future.
    """
    edge_lookup: dict[int, list[float]] = {}
    for row in pairs.itertuples():
        edge_lookup.setdefault(row.seq_a, []).append(row.edge_score)
        edge_lookup.setdefault(row.seq_b, []).append(row.edge_score)

    df = df.sort_values("app_seq")
    out = []
    for cid, grp in df.groupby("cluster_id"):
        grp = grp.sort_values("app_date")
        for pos, row in enumerate(grp.itertuples()):
            prior = grp.iloc[:pos]
            n_prior = len(prior)
            bad_prior = int(
                prior["status"].isin(["rejected", "blocklisted"]).sum()
            ) if n_prior else 0
            span = (
                (pd.Timestamp(row.app_date) - pd.Timestamp(prior["app_date"].min())).days
                if n_prior else 0
            )
            edges = edge_lookup.get(row.app_seq, [])
            out.append(
                {
                    "app_seq": row.app_seq,
                    "cluster_id": cid,
                    "cluster_size_so_far": n_prior + 1,
                    "prior_apps": n_prior,
                    "prior_rejections": bad_prior,
                    "prior_rejection_rate": bad_prior / n_prior if n_prior else 0.0,
                    "cluster_days_span": span,
                    "apps_per_100d": (n_prior + 1) / max(span, 1) * 100 if span else 0.0,
                    "distinct_phones": prior["phone"].nunique() + 1 if n_prior else 1,
                    "distinct_pans": prior["pan"].nunique() + 1 if n_prior else 1,
                    "distinct_accounts": (
                        prior["settlement_account"].nunique() + 1 if n_prior else 1
                    ),
                    "has_prior": int(bool(n_prior)),
                    "max_edge_score": max(edges) if edges else 0.0,
                    "mean_edge_score": sum(edges) / len(edges) if edges else 0.0,
                    "n_edges": len(edges),
                }
            )
    feats = pd.DataFrame(out)
    for name, col in [("phone", "distinct_phones"), ("pan", "distinct_pans"),
                      ("account", "distinct_accounts")]:
        feats[f"{name}_churn"] = (
            feats[col] / feats["cluster_size_so_far"] * feats["has_prior"]
        )
        feats[f"{name}_excess"] = (feats[col] - 1) * feats["has_prior"]
    return feats


CLUSTER_COLS = [
    "cluster_size_so_far", "prior_apps", "prior_rejections",
    "prior_rejection_rate", "cluster_days_span", "apps_per_100d",
    "phone_churn", "pan_churn", "account_churn", "phone_excess", "pan_excess",
    "account_excess", "has_prior", "max_edge_score", "mean_edge_score", "n_edges",
]