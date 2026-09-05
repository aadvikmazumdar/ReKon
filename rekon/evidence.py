"""Candidate retrieval (blocking) and pairwise evidence extraction.

Two rules shape this module:

1. Blocking is generous, scoring is strict. Recall lost here can never be
   recovered downstream, so several independent keys are used and only
   pathologically large blocks are dropped.
2. Pairs are time-ordered: an application may only be compared against
   applications that already existed. This mirrors production and removes an
   entire class of lookahead leakage from the evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import jellyfish
import pandas as pd

from .identity import pan_from_gstin

MAX_BLOCK = 60  # blocks larger than this are uninformative and quadratic


def _surname(name: str) -> str:
    parts = str(name).split()
    return parts[-1] if parts else ""


def _first(name: str) -> str:
    parts = str(name).split()
    return parts[0] if parts else ""


def _email_local(email: str) -> str:
    return str(email).split("@")[0]


def _addr_tokens(address: str) -> set[str]:
    return {t.strip().lower() for t in str(address).replace(",", " ").split() if len(t) > 2}


def blocking_keys(row) -> list[str]:
    """Independent keys; a pair is a candidate if it collides on any one."""
    keys = [
        f"pan:{row.pan}",
        f"panhead:{row.pan[:5]}",
        f"pandigits:{row.pan[5:9]}",
        f"gstpan:{pan_from_gstin(row.gstin)}",
        f"phone:{row.phone}",
        f"phone6:{str(row.phone)[-6:]}",
        f"pin:{row.pincode}",
        f"sdx:{jellyfish.soundex(_surname(row.name))}|{row.city}",
        f"sdxf:{jellyfish.soundex(_first(row.name))}|{row.pincode}",
        f"biz:{jellyfish.soundex(str(row.business_name).split()[-1])}|{row.city}",
        f"email:{_email_local(row.email)[:8]}",
        f"acct:{row.settlement_account}",
        f"acct8:{str(row.settlement_account)[-8:]}",
    ]
    return [k for k in keys if not k.endswith(":None")]


def candidate_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    """Return (earlier_seq, later_seq) index pairs sharing at least one key."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in df.itertuples():
        for key in blocking_keys(row):
            buckets[key].append(row.app_seq)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > MAX_BLOCK:
            continue
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return pairs


def evidence_vector(a, b) -> dict:
    """Field-level comparison of two applications. Raw and inspectable.

    Note the bracket access on "name". These are pandas Series, and Series.name
    is the index label, not the column -- attribute access silently returns the
    app_seq integer instead of the applicant's name.
    """
    name_a, name_b = str(a["name"]), str(b["name"])
    pan_a, pan_b = str(a.pan), str(b.pan)
    gst_pan_a, gst_pan_b = pan_from_gstin(a.gstin), pan_from_gstin(b.gstin)
    ph_a, ph_b = str(a.phone), str(b.phone)

    tok_a, tok_b = _addr_tokens(a.address), _addr_tokens(b.address)
    overlap = len(tok_a & tok_b) / max(1, min(len(tok_a), len(tok_b)))

    return {
        "pan_exact": float(pan_a == pan_b),
        "pan_jw": jellyfish.jaro_winkler_similarity(pan_a, pan_b),
        "gstin_pan_match": float(
            gst_pan_a is not None and gst_pan_a == gst_pan_b
        ),
        "gstin_pan_cross": float(
            (gst_pan_a == pan_b) or (gst_pan_b == pan_a)
        ),
        "phone_exact": float(ph_a == ph_b),
        "phone_jw": jellyfish.jaro_winkler_similarity(ph_a, ph_b),
        "name_jw": jellyfish.jaro_winkler_similarity(name_a, name_b),
        "surname_jw": jellyfish.jaro_winkler_similarity(
            _surname(name_a), _surname(name_b)
        ),
        "name_sdx_match": float(
            jellyfish.soundex(_surname(name_a)) == jellyfish.soundex(_surname(name_b))
        ),
        "business_jw": jellyfish.jaro_winkler_similarity(
            str(a.business_name), str(b.business_name)
        ),
        "address_overlap": overlap,
        "pincode_exact": float(str(a.pincode) == str(b.pincode)),
        "city_exact": float(a.city == b.city),
        "email_jw": jellyfish.jaro_winkler_similarity(
            _email_local(a.email), _email_local(b.email)
        ),
        "days_apart": abs((pd.Timestamp(a.app_date) - pd.Timestamp(b.app_date)).days),
        # Settlement account is costly to churn (KYC on a current account), so
        # reuse is strong linkage evidence -- but it is also shared innocently
        # by aggregators and family businesses, so it cannot stand alone.
        "account_exact": float(str(a.settlement_account) == str(b.settlement_account)),
        "account_jw": jellyfish.jaro_winkler_similarity(
            str(a.settlement_account), str(b.settlement_account)
        ),
        "ifsc_exact": float(str(a.settlement_ifsc) == str(b.settlement_ifsc)),
        "bank_exact": float(str(a.settlement_ifsc)[:4] == str(b.settlement_ifsc)[:4]),
        "mcc_exact": float(str(a.mcc) == str(b.mcc)),
        "gmv_ratio": min(a.monthly_gmv, b.monthly_gmv) / max(1, max(a.monthly_gmv, b.monthly_gmv)),
        "ticket_ratio": min(a.avg_ticket, b.avg_ticket) / max(1, max(a.avg_ticket, b.avg_ticket)),
    }


EVIDENCE_COLS = [
    "pan_exact", "pan_jw", "gstin_pan_match", "gstin_pan_cross",
    "phone_exact", "phone_jw", "name_jw", "surname_jw", "name_sdx_match",
    "business_jw", "address_overlap", "pincode_exact", "city_exact",
    "email_jw", "days_apart", "account_exact", "account_jw", "ifsc_exact",
    "bank_exact", "mcc_exact", "gmv_ratio", "ticket_ratio",
]


def build_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Evidence table for every candidate pair, with ground-truth link flags."""
    rows = df.set_index("app_seq")
    records = []
    for i, j in sorted(candidate_pairs(df)):
        a, b = rows.loc[i], rows.loc[j]
        rec = evidence_vector(a, b)
        rec["seq_a"], rec["seq_b"] = i, j
        rec["app_a"], rec["app_b"] = a.app_id, b.app_id
        rec["same_operator"] = int(a.operator_id == b.operator_id)
        rec["same_entity"] = int(a.entity_id == b.entity_id)
        records.append(rec)
    return pd.DataFrame(records)