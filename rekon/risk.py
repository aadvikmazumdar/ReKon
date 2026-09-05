"""Risk scoring.

Linkage says two applications belong to the same operator. That is evidence,
not a verdict: most linked applications are legitimate. A founder runs two
businesses, a company rebrands, a household shares a phone, someone rejected
for thin credit history reapplies openly six months later.

What this layer scores is concealment, not linkage:

    is this application connected to a prior risk decision AND
    presenting a materially altered identity?

Logistic regression, deliberately. The features are already engineered and
close to monotonic in risk, positives number in the hundreds, and the model's
coefficients ARE the evidence report -- not a post-hoc approximation of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .graph import CLUSTER_COLS

# Fields whose disagreement constitutes alteration of identity.
ALTERATION_FIELDS = [
    "pan_exact", "phone_exact", "account_exact", "name_jw",
    "business_jw", "address_overlap", "email_jw",
]

LINK_COLS = [
    "best_link_score", "link_pan_exact", "link_phone_exact",
    "link_account_exact", "link_name_jw", "link_business_jw",
    "link_address_overlap", "alteration_breadth", "identity_unchanged",
]

# Collinear features make logistic coefficients uninterpretable: weight is
# split arbitrarily between correlated columns and signs can flip, which
# matters here because the coefficients ARE the explanation. cluster_size and
# prior_apps differ by exactly one; each churn ratio is a rescaling of its
# excess count; the three link-strength columns measure one thing. One of each
# is kept.
FEATURES = [
    "prior_rejections", "prior_rejection_rate", "cluster_size_so_far",
    "cluster_days_span", "apps_per_100d", "has_prior",
    "pan_excess", "phone_excess", "account_excess", "n_edges",
    "best_link_score", "link_pan_exact", "link_phone_exact",
    "link_account_exact", "alteration_breadth",
]
# identity_unchanged is excluded: it is exactly (alteration_breadth == 0), so
# once name comparison was fixed the two became near-perfectly collinear and
# logistic regression split the weight between them, flipping the sign of the
# smaller one. alteration_breadth carries the concealment signal on its own.


def build_features(
    df: pd.DataFrame, pairs: pd.DataFrame, edge_threshold: float = 0.0
) -> pd.DataFrame:
    """Per-application features, using only strictly earlier applications.

    An application is described by the strongest link it has to something that
    already existed, plus the state of the cluster it lands in.

    Only pairs that cleared the edge threshold count as links. Candidate pairs
    below it are blocking collisions between strangers, and letting one stand
    in as an application's "best link" makes alteration_breadth measure whether
    a record happened to collide with someone rather than how much of its own
    identity changed -- which is the opposite of what the feature is for.
    """
    seq_date = df.set_index("app_seq")["app_date"].to_dict()

    linked = pairs[pairs["edge_score"] >= edge_threshold]
    a_first = linked.assign(
        this=linked.seq_b, other=linked.seq_a
    )  # b is always the later record
    prior = a_first[a_first["other"].map(seq_date) <= a_first["this"].map(seq_date)]
    if prior.empty:
        prior = a_first.head(0)

    if len(prior):
        best_idx = prior.groupby("this")["edge_score"].idxmax()
        best = prior.loc[best_idx].set_index("this")
    else:
        best = prior.set_index("this")

    alt = np.zeros(len(best))
    for col in ALTERATION_FIELDS:
        v = best[col].to_numpy(dtype=float)
        alt += (v < 0.9).astype(float)

    link = pd.DataFrame(
        {
            "app_seq": best.index,
            "best_link_score": best["edge_score"].to_numpy(),
            "link_pan_exact": best["pan_exact"].to_numpy(),
            "link_phone_exact": best["phone_exact"].to_numpy(),
            "link_account_exact": best["account_exact"].to_numpy(),
            "link_name_jw": best["name_jw"].to_numpy(),
            "link_business_jw": best["business_jw"].to_numpy(),
            "link_address_overlap": best["address_overlap"].to_numpy(),
            "alteration_breadth": alt,
            # The exculpatory signal: linked, but hiding nothing.
            "identity_unchanged": (alt == 0).astype(float),
        }
    )

    out = df[["app_seq"] + [c for c in CLUSTER_COLS if c in df.columns]].merge(
        link, on="app_seq", how="left"
    )
    for col in LINK_COLS:
        out[col] = out[col].fillna(0.0)
    # No prior link at all means nothing was concealed.
    out.loc[out["best_link_score"] == 0, "identity_unchanged"] = 1.0
    return out


class RiskModel:
    def __init__(self, C: float = 1.0) -> None:
        self.scaler = StandardScaler()
        # No class weighting: it inflates predicted probabilities away from the
        # true base rate, and calibration is a reported metric here rather than
        # an afterthought.
        self.clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RiskModel":
        # Apple's Accelerate BLAS sets FP exception flags during matmul even
        # when every value is finite, and sklearn's solver matmuls trip them.
        # Results are unaffected -- the same fit reproduces exactly on other
        # BLAS backends -- so the flags are suppressed rather than chased.
        with np.errstate(all="ignore"):
            self.clf.fit(self.scaler.fit_transform(X[FEATURES]), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        with np.errstate(all="ignore"):
            return self.clf.predict_proba(self.scaler.transform(X[FEATURES]))[:, 1]

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": FEATURES, "coefficient": self.clf.coef_[0]}
        ).sort_values("coefficient", ascending=False, ignore_index=True)

    def explain(self, X: pd.DataFrame, i: int, top: int = 6) -> pd.DataFrame:
        """Per-application contributions, straight from the model."""
        z = self.scaler.transform(X[FEATURES])[i]
        contrib = z * self.clf.coef_[0]
        out = pd.DataFrame(
            {"feature": FEATURES, "value": X[FEATURES].iloc[i].to_numpy(),
             "contribution": contrib}
        )
        return out.reindex(out.contribution.abs().sort_values(ascending=False).index).head(top)


def cost_optimal_thresholds(
    cost_fp: float, cost_fn: float, review_cost: float = 0.0
) -> tuple[float, float]:
    """Expected-loss thresholds, in closed form.

    Blocking is worth it when the expected cost of approving exceeds the cost
    of wrongly blocking:

        p * cost_fn > (1 - p) * cost_fp   ->   p > cost_fp / (cost_fp + cost_fn)

    Reviewing is worth it when the expected loss avoided exceeds what a review
    costs: p > review_cost / cost_fn. No grid search and no tuning -- the
    operating point is implied by the cost ratio.

    Note this is the economically optimal review threshold, not an operationally
    feasible one: with a review costing far less than a funded evader, it sends
    a large queue to underwriters. precision_at_n reports the budget-constrained
    view alongside it.
    """
    t_block = cost_fp / (cost_fp + cost_fn)
    t_review = min(t_block, review_cost / cost_fn) if cost_fn > 0 else t_block
    return t_review, t_block


def precision_at_n(y: np.ndarray, p: np.ndarray, n: int) -> tuple[float, float]:
    """Precision within a fixed daily review budget, and recall inside it."""
    order = np.argsort(-p)[:n]
    hits = y[order].sum()
    return hits / max(1, n), hits / max(1, y.sum())


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 8) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    return pd.DataFrame(rows)