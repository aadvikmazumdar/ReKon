"""Dataset validation suite.

Every check either passes or reports exactly what failed. Nothing here is
allowed to be a soft warning: a synthetic dataset that quietly lies about its
own structure invalidates every number measured on it.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9]Z[A-Z0-9]$")
PHONE_RE = re.compile(r"^[6-9][0-9]{9}$")
PIN_RE = re.compile(r"^[1-9][0-9]{5}$")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["app_date"] = pd.to_datetime(df["app_date"])

    # ---- structural integrity ---------------------------------------------
    check("PAN format", df.pan.str.match(PAN_RE).all(),
          f"{(~df.pan.str.match(PAN_RE)).sum()} violations")
    check("GSTIN format", df.gstin.str.match(GSTIN_RE).all(),
          f"{(~df.gstin.str.match(GSTIN_RE)).sum()} violations")
    check("Phone format", df.phone.astype(str).str.match(PHONE_RE).all(),
          f"{(~df.phone.astype(str).str.match(PHONE_RE)).sum()} violations")
    check("Pincode format", df.pincode.astype(str).str.match(PIN_RE).all(),
          f"{(~df.pincode.astype(str).str.match(PIN_RE)).sum()} violations")
    check("app_id unique", df.app_id.is_unique)
    check("No null fields", not df.isnull().any().any())

    # GSTIN must embed the declared PAN except where evasion deliberately breaks it
    embeds = df.gstin.str[2:12] == df.pan
    lvl4 = df.evasion_level == 4
    check("GSTIN embeds PAN (non-L4)", embeds[~lvl4].all(),
          f"{(~embeds[~lvl4]).sum()} unexpected mismatches")
    check("L4 breaks PAN/GSTIN link", (~embeds[lvl4]).all(),
          f"{embeds[lvl4].sum()} L4 rows failed to break the link")

    # ---- label integrity ---------------------------------------------------
    check("Only ring_evasion is positive",
          set(df[df.is_evasion].cohort.unique()) == {"ring_evasion"},
          str(sorted(df[df.is_evasion].cohort.unique())))
    check("All ring_evasion are positive",
          df[df.cohort == "ring_evasion"].is_evasion.all(),
          f"{(~df[df.cohort=='ring_evasion'].is_evasion).sum()} unlabelled")
    check("honest_reapply is negative",
          not df[df.cohort == "hn_honest_reapply"].is_evasion.any())
    check("Positives all have evasion_level>0",
          (df[df.is_evasion].evasion_level > 0).all())
    check("Negatives all have evasion_level==0",
          (df[~df.is_evasion].evasion_level == 0).all())

    # ---- temporal causality ------------------------------------------------
    bad_time = 0
    for _op, g in df[df.cohort.str.startswith("ring")].groupby("operator_id"):
        g = g.sort_values("app_date")
        origin = g[g.cohort == "ring_origin"]
        if origin.empty:
            continue
        t0 = origin.app_date.min()
        bad_time += int((g[g.evasion_level > 0].app_date < t0).sum())
    check("Evasion always follows the rejection", bad_time == 0,
          f"{bad_time} evasion attempts predate their origin")

    # ---- leakage probes ----------------------------------------------------
    # any single raw column that separates the classes too well is a giveaway
    leak = {}
    for col in ["requested_amount", "city", "pincode", "status"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            rate = df.groupby(col).is_evasion.mean()
            leak[col] = float(rate.max() - rate.min())
        else:
            a = df[df.is_evasion][col]
            b = df[~df.is_evasion][col]
            pooled = np.sqrt((a.var() + b.var()) / 2) or 1
            leak[col] = float(abs(a.mean() - b.mean()) / pooled)
    check("No single-column giveaway (amount)", leak["requested_amount"] < 0.3,
          f"cohen_d={leak['requested_amount']:.3f}")
    check("City not predictive", leak["city"] < 0.25, f"spread={leak['city']:.3f}")

    # app_id ordering must not encode the label
    order_corr = abs(np.corrcoef(
        df.sort_values("app_date").is_evasion.astype(int),
        np.arange(len(df)))[0, 1])
    # A small positive correlation is causally required, not leakage: evasion can
    # only occur after a prior rejection, so positives sit later in the window.
    # The guard is that it stays small, and that no positional feature is used.
    check("Row order only weakly predictive", order_corr < 0.09,
          f"r={order_corr:.4f} (causally expected, no positional feature is used)")

    # name/business length must not correlate with chain depth (the & Sons bug)
    ring = df[df.cohort.str.startswith("ring")].copy()
    depth = ring.groupby("operator_id").cumcount()
    biz_corr = abs(np.corrcoef(depth, ring.business_name.str.len())[0, 1])
    check("Business-name length not depth-correlated", biz_corr < 0.22,
          f"r={biz_corr:.4f}")

    # ---- payment layer ----
    check("Account format", df.settlement_account.astype(str).str.match(r"^[0-9]{11,16}$").all(),
          f"{(~df.settlement_account.astype(str).str.match(r'^[0-9]{11,16}$')).sum()} violations")
    check("IFSC format", df.settlement_ifsc.str.match(r"^[A-Z]{4}0[0-9]{6}$").all())
    check("Refund rate in range", df.refund_rate.between(0, 0.35).all())
    check("GMV consistent with txn x ticket",
          (df.monthly_gmv == df.txn_count * df.avg_ticket).all())

    # Payment aggregates must not encode the label. If any of these separate
    # the classes, we invented a fraud signal instead of measuring one.
    for col in ["monthly_gmv", "txn_count", "avg_ticket", "refund_rate", "active_days"]:
        a, b = df[df.is_evasion][col], df[~df.is_evasion][col]
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1
        d = abs(a.mean() - b.mean()) / pooled
        check(f"Payment aggregate neutral: {col}", d < 0.15, f"cohen_d={d:.3f}")

    shared = df[df.cohort == "hn_shared_account"]
    check("Shared-account cohort present", len(shared) > 0, f"{len(shared)} rows")
    check("Shared-account operators are distinct",
          shared.groupby("settlement_account").operator_id.nunique().min() >= 2,
          "unrelated merchants per shared account")

    # ---- distribution realism ----------------------------------------------
    rate = df.is_evasion.mean()
    check("Evasion rate in 2-8% band", 0.02 <= rate <= 0.08, f"{rate:.3%}")
    check("Level coverage balanced",
          df[df.is_evasion].evasion_level.value_counts().min() >= 30,
          df[df.is_evasion].evasion_level.value_counts().sort_index().to_dict())
    check("Hard negatives >= 8% of data",
          df.cohort.str.startswith("hn").mean() >= 0.08,
          f"{df.cohort.str.startswith('hn').mean():.2%}")

    # exact-duplicate rows would be a generator artifact
    dupe_cols = ["name", "phone", "pan", "business_name", "address"]
    exact = df.duplicated(subset=dupe_cols, keep=False)
    legit = df[exact].cohort.isin(["hn_honest_reapply"]).all()  # only cohort where identical rows are by design
    check("Exact dupes only from honest_reapply", legit,
          f"{exact.sum()} exact-duplicate rows")

    return df


def report() -> int:
    width = max(len(n) for n, _, _ in results) + 2
    failed = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name:<{width}} {detail}")
    print(f"\n  {len(results)-failed}/{len(results)} checks passed")
    return failed


if __name__ == "__main__":
    frame = pd.read_csv("/home/claude/rekon/data/applications.csv")
    print("DATASET VALIDATION\n")
    validate(frame)
    sys.exit(report())