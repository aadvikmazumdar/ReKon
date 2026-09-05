"""Fellegi-Sunter probabilistic record linkage.

The edge weights in graph.py are hand-set constants chosen from domain
reasoning. They work, but "we picked 3.0 for PAN" is the weakest claim in the
system and the easiest thing to challenge.

Fellegi-Sunter (1969) replaces them with estimated likelihood ratios. For each
comparison field i:

    m_i = P(field i agrees | the pair is a match)
    u_i = P(field i agrees | the pair is not a match)

and the evidence contributed by that field is

    agree     ->  log2(m_i / u_i)
    disagree  ->  log2((1 - m_i) / (1 - u_i))

Summing across fields gives a log-likelihood ratio. This is the standard model
behind recordlinkage and Splink.

Two properties matter for ReKon:

1. u_i is the chance agreement rate, so a field that collides often by accident
   (a common surname, a shared pincode) earns little weight automatically. The
   population-frequency problem solves itself rather than needing a rule.
2. The parameters are fitted by EM without labels, so this consumes none of our
   ground truth. Labels stay reserved for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous comparisons are binarised at the point where a field stops being
# plausibly the same value. These are comparison thresholds, not weights: the
# weight each field earns is estimated, not chosen.
AGREEMENT_RULES: dict[str, tuple[str, float]] = {
    "pan_exact": ("eq", 1.0),
    "gstin_pan_match": ("eq", 1.0),
    "gstin_pan_cross": ("eq", 1.0),
    "phone_exact": ("eq", 1.0),
    "account_exact": ("eq", 1.0),
    "ifsc_exact": ("eq", 1.0),
    "pincode_exact": ("eq", 1.0),
    "mcc_exact": ("eq", 1.0),
    "name_sdx_match": ("eq", 1.0),
    "name_jw": ("gte", 0.90),
    "surname_jw": ("gte", 0.92),
    "business_jw": ("gte", 0.85),
    "email_jw": ("gte", 0.85),
    "address_overlap": ("gte", 0.50),
    "pan_jw": ("gte", 0.90),
    "phone_jw": ("gte", 0.90),
}

FIELDS = list(AGREEMENT_RULES)


def agreement_matrix(pairs: pd.DataFrame) -> np.ndarray:
    """Binary agreement vector per candidate pair."""
    cols = []
    for field, (op, thr) in AGREEMENT_RULES.items():
        v = pairs[field].to_numpy(dtype=float)
        cols.append((v >= thr).astype(float) if op == "gte" else (v == thr).astype(float))
    return np.column_stack(cols)


class FellegiSunter:
    """EM-fitted two-class mixture over agreement patterns."""

    def __init__(
        self,
        n_iter: int = 60,
        tol: float = 1e-7,
        seed: int = 3,
        prior_strength: float = 50.0,
        m_prior: float = 0.9,
        u_prior: float = 0.05,
    ) -> None:
        # Unsmoothed EM drives m to 1.0 and u to 0 for the strongest fields.
        # A disagreement weight of -13 bits then claims a single mismatched
        # field makes a match near-impossible, which no finite sample supports.
        # Beta pseudo-counts keep every estimate strictly interior, which also
        # keeps the total score on a scale where the operating threshold is a
        # positive number of bits rather than a large negative one.
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.prior_strength = prior_strength
        self.m_prior = m_prior
        self.u_prior = u_prior
        self.m_: np.ndarray | None = None
        self.u_: np.ndarray | None = None
        self.lambda_: float = 0.01
        self.n_iter_run_: int = 0

    @staticmethod
    def _loglik(gamma: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = gamma @ np.log(p) + (1 - gamma) @ np.log(1 - p)
        return np.nan_to_num(out, nan=-1e6, posinf=1e6, neginf=-1e6)

    def fit(self, gamma: np.ndarray) -> "FellegiSunter":
        # Apple's Accelerate BLAS raises FP exception flags during matmul even
        # when every result is finite, so numpy reports warnings that do not
        # correspond to anything wrong. Values are clipped explicitly either
        # side of each product, so the flags are suppressed here.
        rng = np.random.default_rng(self.seed)
        n_fields = gamma.shape[1]

        # Matches agree on most fields; non-matches agree at roughly the rate
        # the field collides by chance. Jitter avoids a symmetric start.
        self.m_ = np.clip(0.90 + rng.normal(0, 0.02, n_fields), 0.6, 0.99)
        self.u_ = np.clip(gamma.mean(axis=0), 1e-4, 0.5)
        self.lambda_ = 0.01

        prev = -np.inf
        with np.errstate(all="ignore"):
          for it in range(self.n_iter):
              # E step: posterior probability that each pair is a match
              lm = np.log(self.lambda_) + self._loglik(gamma, self.m_)
              lu = np.log(1 - self.lambda_) + self._loglik(gamma, self.u_)
              mx = np.maximum(lm, lu)
              denom = mx + np.log(np.exp(lm - mx) + np.exp(lu - mx))
              g = np.exp(lm - denom)

              # M step, with Beta(alpha, beta) pseudo-counts
              gs = g.sum()
              a_m = self.prior_strength * self.m_prior
              b_m = self.prior_strength * (1 - self.m_prior)
              a_u = self.prior_strength * self.u_prior
              b_u = self.prior_strength * (1 - self.u_prior)
              self.m_ = np.clip(((g @ gamma) + a_m) / (gs + a_m + b_m), 0.01, 0.99)
              self.u_ = np.clip(
                  (((1 - g) @ gamma) + a_u) / ((len(g) - gs) + a_u + b_u), 1e-4, 0.5
              )
              self.lambda_ = float(np.clip(gs / len(g), 1e-6, 0.5))

              ll = float(denom.sum())
              self.n_iter_run_ = it + 1
              if abs(ll - prev) < self.tol * max(1.0, abs(prev)):
                  break
              prev = ll
        return self

    def weights(self) -> pd.DataFrame:
        """Per-field agreement and disagreement weights, in bits."""
        agree, disagree = self._weight_vectors()
        return pd.DataFrame(
            {
                "field": FIELDS,
                "m": self.m_,
                "u": self.u_,
                "agree_weight": agree,
                "disagree_weight": disagree,
            }
        ).sort_values("agree_weight", ascending=False, ignore_index=True)

    def _weight_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        m = np.clip(self.m_, 0.01, 0.99)
        u = np.clip(self.u_, 1e-4, 0.5)
        return np.log2(m / u), np.log2((1 - m) / (1 - u))

    def score(self, gamma: np.ndarray) -> np.ndarray:
        """Match log-likelihood ratio, in bits."""
        agree, disagree = self._weight_vectors()
        with np.errstate(all="ignore"):
            return np.nan_to_num(gamma @ agree + (1 - gamma) @ disagree)

    def match_probability(self, gamma: np.ndarray) -> np.ndarray:
      with np.errstate(all="ignore"):
        lm = np.log(self.lambda_) + self._loglik(gamma, self.m_)
        lu = np.log(1 - self.lambda_) + self._loglik(gamma, self.u_)
        mx = np.maximum(lm, lu)
        return np.exp(lm - (mx + np.log(np.exp(lm - mx) + np.exp(lu - mx))))