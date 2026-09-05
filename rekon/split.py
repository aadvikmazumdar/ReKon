"""Train / test split.

The split is grouped by operator_id, never by row. A ring split across the
boundary would put some of an operator's applications in train and the rest in
test, letting the model memorise a ring it is later scored on. That is the
single easiest way to manufacture a good number on this dataset, so the split
is grouped and then asserted.

Held-out means held out: thresholds are chosen on train only, and the test set
is scored once with everything frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def operator_split(
    df: pd.DataFrame, test_frac: float = 0.30, seed: int = 11
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # stratify operators by whether they contain any evasion, so both sides
    # carry a comparable positive rate
    # Stratify by evasion level, not just by presence: per-level recall is the
    # headline result, so every level needs enough test mass to be reportable.
    per_op = df.groupby("operator_id").agg(
        has_evasion=("is_evasion", "any"),
        level=("evasion_level", "max"),
        n=("app_id", "size"),
    )
    test_ops: list[str] = []
    for _flag, grp in per_op.groupby(["has_evasion", "level"]):
        ops = grp.index.to_numpy()
        rng.shuffle(ops)
        test_ops.extend(ops[: int(len(ops) * test_frac)])

    test_mask = df["operator_id"].isin(set(test_ops))
    return df[~test_mask].copy(), df[test_mask].copy()


def validate_split(train: pd.DataFrame, test: pd.DataFrame) -> list[tuple[str, bool, str]]:
    out = []
    tr_ops, te_ops = set(train.operator_id), set(test.operator_id)
    out.append(("No operator spans the split", not (tr_ops & te_ops),
                f"{len(tr_ops & te_ops)} shared operators"))
    tr_ent, te_ent = set(train.entity_id), set(test.entity_id)
    out.append(("No entity spans the split", not (tr_ent & te_ent),
                f"{len(tr_ent & te_ent)} shared entities"))

    r_tr, r_te = train.is_evasion.mean(), test.is_evasion.mean()
    out.append(("Positive rate comparable", abs(r_tr - r_te) < 0.015,
                f"train={r_tr:.3%} test={r_te:.3%}"))

    lv = test[test.is_evasion].evasion_level.value_counts()
    out.append(("All 5 evasion levels present in test", len(lv) == 5,
                lv.sort_index().to_dict()))
    out.append(("Every level has >=10 test positives",
                bool(lv.min() >= 10) if len(lv) else False, f"min={lv.min() if len(lv) else 0}"))

    cohorts = set(train.cohort) == set(test.cohort)
    out.append(("All cohorts represented both sides", cohorts,
                str(sorted(set(train.cohort) ^ set(test.cohort)) or "identical")))

    # a PAN appearing on both sides would leak an identity across the boundary
    shared_pan = set(train.pan) & set(test.pan)
    out.append(("No PAN spans the split", not shared_pan,
                f"{len(shared_pan)} shared PANs"))
    return out


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/rekon/data/applications.csv")
    train, test = operator_split(df)
    train.to_csv("/home/claude/rekon/data/train.csv", index=False)
    test.to_csv("/home/claude/rekon/data/test.csv", index=False)

    print(f"SPLIT VALIDATION   train={len(train)}  test={len(test)}\n")
    failed = 0
    checks = validate_split(train, test)
    width = max(len(n) for n, _, _ in checks) + 2
    for name, ok, detail in checks:
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
    print(f"\n  {len(checks)-failed}/{len(checks)} checks passed")