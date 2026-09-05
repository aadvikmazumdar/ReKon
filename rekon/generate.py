"""Population generator for ReKon.

Produces an application table with hidden ground truth:
  entity_id    the same real-world identity
  operator_id  the same controlling actor (a ring is one operator, many entities)

The label used downstream is derived, never stored as a feature:
  an application is an evasion attempt when its operator already has a
  rejected or blocklisted application AND its identity was deliberately altered.

Honest re-application after rejection is therefore a negative, even though it
is linked to a rejection. That distinction is the whole point of the project.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import pandas as pd

from .corrupt import apply_evasion, corrupt_business, corrupt_name, natural_drift
from .identity import (
    CITIES,
    Identity,
    fresh_address,
    fresh_phone,
    gstin_from_pan,
    new_identity,
    sibling_identity,
)

START = date(2024, 1, 1)
WINDOW_DAYS = 540

RING_SIZES = [2, 3, 4, 5, 6]
RING_WEIGHTS = [0.55, 0.22, 0.12, 0.07, 0.04]


@dataclass
class Application:
    app_id: str
    name: str
    settlement_account: str
    settlement_ifsc: str
    mcc: str
    monthly_gmv: int
    txn_count: int
    avg_ticket: int
    refund_rate: float
    active_days: int
    phone: str
    pan: str
    gstin: str
    business_name: str
    address: str
    city: str
    pincode: str
    email: str
    app_date: date
    requested_amount: int
    status: str
    entity_id: str
    operator_id: str
    evasion_level: int
    cohort: str


def _amount(rng: random.Random) -> int:
    return int(rng.choice([50, 75, 100, 150, 200, 300, 500, 750, 1000]) * 1000)


def _payment_profile(rng: random.Random) -> dict:
    """Merchant payment aggregates.

    Drawn from one distribution for every cohort, deliberately. Correlating
    GMV or refund rate with fraud would mean inventing a relationship we have
    no evidence for, and the model would then recover our assumption rather
    than a real pattern. These enter as neutral features and earn their weight
    from the data or not at all.
    """
    txn_count = int(rng.lognormvariate(6.0, 1.1))
    avg_ticket = int(rng.lognormvariate(6.6, 0.8))
    return {
        "monthly_gmv": txn_count * avg_ticket,
        "txn_count": txn_count,
        "avg_ticket": avg_ticket,
        "refund_rate": round(min(0.35, abs(rng.gauss(0.04, 0.035))), 4),
        "active_days": min(730, int(abs(rng.gauss(280, 160))) + 20),
    }


def _date(rng: random.Random) -> date:
    return START + timedelta(days=rng.randrange(WINDOW_DAYS))


def _mk(
    rng: random.Random,
    ident: Identity,
    app_date: date,
    status: str,
    entity_id: str,
    operator_id: str,
    evasion_level: int,
    cohort: str,
    counter: list[int],
) -> Application:
    counter[0] += 1
    return Application(
        app_id=f"APP{counter[0]:06d}",
        name=ident.name,
        settlement_account=ident.settlement_account,
        settlement_ifsc=ident.settlement_ifsc,
        mcc=ident.mcc,
        **_payment_profile(rng),
        phone=ident.phone,
        pan=ident.pan,
        gstin=ident.gstin,
        business_name=ident.business_name,
        address=ident.address,
        city=ident.city,
        pincode=ident.pincode,
        email=ident.email,
        app_date=app_date,
        requested_amount=_amount(rng),
        status=status,
        entity_id=entity_id,
        operator_id=operator_id,
        evasion_level=evasion_level,
        cohort=cohort,
    )


def generate(
    n_target: int = 8000,
    seed: int = 7,
    p_ring: float = 0.07,
    p_hard_negative: float = 0.10,
    p_grey: float = 0.04,
) -> pd.DataFrame:
    rng = random.Random(seed)
    counter = [0]
    apps: list[Application] = []
    op = [0]
    ent = [0]

    def next_op() -> str:
        op[0] += 1
        return f"OP{op[0]:05d}"

    def next_ent() -> str:
        ent[0] += 1
        return f"ENT{ent[0]:05d}"

    n_ring_apps = int(n_target * p_ring)
    n_hard_apps = int(n_target * p_hard_negative)
    n_grey_apps = int(n_target * p_grey)
    n_singleton = n_target - n_ring_apps - n_hard_apps - n_grey_apps

    # ---- rings: one operator, a chain of derived identities -----------------
    produced = 0
    while produced < n_ring_apps:
        size = rng.choices(RING_SIZES, weights=RING_WEIGHTS)[0]
        level = rng.randint(1, 5)
        operator_id = next_op()
        base = new_identity(rng)
        # Origins sit in the earlier part of the window so that long chains fit
        # inside it. Otherwise later hops are silently truncated, which both
        # depresses the positive rate and pushes positives toward the tail.
        first_date = START + timedelta(days=rng.randrange(int(WINDOW_DAYS * 0.45)))

        # the original application gets caught
        status = rng.choices(["rejected", "blocklisted"], weights=[0.6, 0.4])[0]
        apps.append(
            _mk(rng, base, first_date, status, next_ent(), operator_id, 0,
                "ring_origin", counter)
        )
        produced += 1

        current = base
        when = first_date
        # A chain hop can revert an earlier change and land back on an identity
        # already used in this ring, producing a "duplicate" that is really an
        # honest re-application wearing an evasion label.
        seen = {base}
        for _ in range(size - 1):
            for _attempt in range(8):
                candidate, _changed = apply_evasion(rng, current, level)
                if candidate not in seen:
                    break
            current = candidate
            seen.add(current)
            when = when + timedelta(days=rng.randint(5, 400))
            if when > START + timedelta(days=WINDOW_DAYS):
                break
            apps.append(
                _mk(rng, current, when, "rejected" if rng.random() < 0.25 else "approved",
                    next_ent(), operator_id, level, "ring_evasion", counter)
            )
            produced += 1

    # ---- hard negatives -----------------------------------------------------
    produced = 0
    kinds = ["common_name", "multi_founder", "family_phone", "rebrand",
             "honest_reapply", "shared_account"]
    while produced < n_hard_apps:
        kind = rng.choice(kinds)

        if kind == "common_name":
            # two unrelated operators, same name and city, nothing else shared
            city = rng.choice(list(CITIES))
            a = new_identity(rng, city)
            b = sibling_identity(
                rng, new_identity(rng, city), name=a.name
            )
            for ident in (a, b):
                apps.append(
                    _mk(rng, ident, _date(rng), "approved", next_ent(), next_op(),
                        0, "hn_common_name", counter)
                )
            produced += 2

        elif kind == "multi_founder":
            # one genuine person, two legitimate businesses, no rejection history
            operator_id, entity_id = next_op(), next_ent()
            a = new_identity(rng)
            biz = corrupt_business(rng, a.business_name, True)
            while biz == a.business_name:
                biz = corrupt_business(rng, a.business_name, True)
            b, _ = natural_drift(rng, sibling_identity(
                rng, a, business_name=biz,
                address=fresh_address(rng),
                gstin=gstin_from_pan(rng, a.pan, CITIES[a.city][0]),
            ))
            d0 = _date(rng)
            apps.append(_mk(rng, a, d0, "approved", entity_id, operator_id, 0,
                            "hn_multi_founder", counter))
            apps.append(_mk(rng, b, d0 + timedelta(days=rng.randint(60, 400)),
                            "approved", entity_id, operator_id, 0,
                            "hn_multi_founder", counter))
            produced += 2

        elif kind == "family_phone":
            # shared household phone, genuinely different people
            a = new_identity(rng)
            b = sibling_identity(rng, new_identity(rng, a.city), phone=a.phone)
            for ident in (a, b):
                apps.append(
                    _mk(rng, ident, _date(rng), "approved", next_ent(), next_op(),
                        0, "hn_family_phone", counter)
                )
            produced += 2

        elif kind == "rebrand":
            # same entity, renamed business, previously approved
            operator_id, entity_id = next_op(), next_ent()
            a = new_identity(rng)
            biz = corrupt_business(rng, a.business_name, True)
            while biz == a.business_name:
                biz = corrupt_business(rng, a.business_name, True)
            b, _ = natural_drift(rng, sibling_identity(rng, a, business_name=biz))
            d0 = _date(rng)
            apps.append(_mk(rng, a, d0, "approved", entity_id, operator_id, 0,
                            "hn_rebrand", counter))
            apps.append(_mk(rng, b, d0 + timedelta(days=rng.randint(90, 400)),
                            "approved", entity_id, operator_id, 0,
                            "hn_rebrand", counter))
            produced += 2

        elif kind == "shared_account":
            # A payment aggregator, a family business, or a CA filing on behalf
            # of several merchants: unrelated operators, one settlement account.
            # This is the dense legitimate cluster that a linkage system must
            # not treat as a ring.
            anchor = new_identity(rng)
            for _ in range(rng.randint(2, 5)):
                other = new_identity(rng)
                other = sibling_identity(
                    rng, other,
                    settlement_account=anchor.settlement_account,
                    settlement_ifsc=anchor.settlement_ifsc,
                )
                apps.append(
                    _mk(rng, other, _date(rng), "approved", next_ent(), next_op(),
                        0, "hn_shared_account", counter)
                )
                produced += 1

        else:  # honest_reapply
            # rejected once, reapplies openly with the same details: NOT evasion
            operator_id, entity_id = next_op(), next_ent()
            a = new_identity(rng)
            d0 = _date(rng)
            a2, _ = natural_drift(rng, a)
            apps.append(_mk(rng, a, d0, "rejected", entity_id, operator_id, 0,
                            "hn_honest_reapply", counter))
            apps.append(_mk(rng, a2, d0 + timedelta(days=rng.randint(60, 500)),
                            "approved", entity_id, operator_id, 0,
                            "hn_honest_reapply", counter))
            produced += 2

    # ---- grey: a single weak overlap, unrelated operators --------------------
    produced = 0
    while produced < n_grey_apps:
        a = new_identity(rng)
        overlap = rng.choice(["pincode", "surname", "phone_prefix"])
        b = new_identity(rng, a.city)
        if overlap == "pincode":
            b = sibling_identity(rng, b, pincode=a.pincode)
        elif overlap == "surname":
            b = sibling_identity(
                rng, b, name=f"{b.name.split()[0]} {a.name.split()[-1]}"
            )
        else:
            b = sibling_identity(rng, b, phone=a.phone[:5] + fresh_phone(rng)[5:])
        for ident in (a, b):
            apps.append(
                _mk(rng, ident, _date(rng), "approved", next_ent(), next_op(),
                    0, "grey", counter)
            )
        produced += 2

    # ---- singleton legitimate applicants ------------------------------------
    for _ in range(max(0, n_singleton)):
        ident = new_identity(rng)
        status = rng.choices(["approved", "rejected"], weights=[0.88, 0.12])[0]
        apps.append(
            _mk(rng, ident, _date(rng), status, next_ent(), next_op(), 0,
                "singleton", counter)
        )

    df = pd.DataFrame([asdict(a) for a in apps])
    df = df.sort_values("app_date").reset_index(drop=True)
    df["app_seq"] = range(len(df))
    return df


def label_evasion(df: pd.DataFrame) -> pd.Series:
    """Time-ordered label: operator already had a rejection AND identity altered."""
    flagged = set()
    seen_bad: set[str] = set()
    for row in df.sort_values("app_date").itertuples():
        if row.operator_id in seen_bad and row.evasion_level > 0:
            flagged.add(row.app_id)
        if row.status in ("rejected", "blocklisted"):
            seen_bad.add(row.operator_id)
    return df["app_id"].isin(flagged)


if __name__ == "__main__":
    frame = generate()
    frame["is_evasion"] = label_evasion(frame)
    frame.to_csv("/home/claude/rekon/data/applications.csv", index=False)
    print(frame["cohort"].value_counts())
    print("\nevasion rate:", frame["is_evasion"].mean().round(4))