"""Corruption engine.

Every corruption randomises *which* field, *which* transform, and *how hard*.
Nothing is applied by a fixed rule, otherwise a matcher learns the generator
rather than the phenomenon it is supposed to detect.

Evasion levels describe how much effort an operator spends looking like a new
entity. They are the x-axis of the evasion curve in the evaluation harness.
"""

from __future__ import annotations

import random
import string

from .identity import (
    BUSINESS_HEADS,
    fresh_account,
    fresh_ifsc,
    BUSINESS_TAILS,
    CITIES,
    Identity,
    fresh_address,
    fresh_email,
    fresh_pan_for,
    fresh_phone,
    gstin_from_pan,
    sibling_identity,
)

TRANSLIT_RULES = [
    ("aa", "a"), ("a", "aa"), ("ee", "i"), ("i", "ee"), ("oo", "u"),
    ("u", "oo"), ("ksh", "x"), ("th", "t"), ("t", "th"), ("v", "w"),
    ("sh", "s"), ("ck", "k"), ("ph", "f"), ("y", "i"),
]

NAME_ALIASES = {
    "Mohammed": ["Mohammad", "Muhammad", "Md.", "Mohd"],
    "Lakshmi": ["Laxmi", "Lakshmy"],
    "Sathish": ["Satish", "Sateesh"],
    "Praveen": ["Pravin", "Parveen"],
    "Naveen": ["Navin"],
    "Sanjay": ["Sanjai", "Sanjaya"],
    "Suresh": ["Sooresh"],
    "Sneha": ["Snehaa"],
    "Kavya": ["Kaavya"],
    "Karthik": ["Kartik", "Karthick"],
    "Sowmya": ["Soumya", "Saumya"],
    "Rekha": ["Reka"],
    "Vinod": ["Vinodh"],
}

STREET_ABBREV = [
    ("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Cross", "Crs"),
    ("Main Road", "Main Rd"), ("Lane", "Ln"),
]

BUSINESS_REWRITES = [
    ("Traders", "Trading Co."), ("Enterprises", "Enterprise"),
    ("Industries", "Industrial"), ("Agencies", "Agency"),
    ("Distributors", "Distribution"), ("Solutions", "Solution"),
    ("Services", "Service Co."), ("Exports", "Export House"),
]


def _typo(rng: random.Random, s: str, preserve_length: bool = False) -> str:
    """One character-level slip: swap, drop, duplicate or substitute.

    Fixed-format fields (PAN, phone, pincode) must keep their length, otherwise
    the corruption itself becomes a giveaway feature.
    """
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    modes = ["swap", "sub"] if preserve_length else ["swap", "drop", "dup", "sub"]
    mode = rng.choice(modes)
    if mode == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if mode == "drop":
        return s[:i] + s[i + 1:]
    if mode == "dup":
        return s[:i] + s[i] + s[i:]
    repl = rng.choice("abcdefghijklmnopqrstuvwxyz")
    if s[i].isupper():
        repl = repl.upper()
    if s[i].isdigit():
        repl = str(rng.randint(0, 9))
    return s[:i] + repl + s[i + 1:]


def corrupt_pan(rng: random.Random, pan: str) -> str:
    """Alter one character while keeping the AAAAA9999A structure valid."""
    chars = list(pan)
    i = rng.choice([0, 1, 2, 5, 6, 7, 8, 9])  # never the entity-type or surname slot
    if chars[i].isdigit():
        chars[i] = str((int(chars[i]) + rng.randint(1, 9)) % 10)
    else:
        alphabet = [c for c in string.ascii_uppercase if c != chars[i]]
        chars[i] = rng.choice(alphabet)
    return "".join(chars)


def corrupt_name(rng: random.Random, name: str, hard: bool) -> str:
    first, *rest = name.split()
    surname = rest[-1] if rest else ""
    choices = ["alias", "translit", "typo", "initial", "reorder"]
    mode = rng.choice(choices if hard else choices[:3])

    if mode == "alias" and first in NAME_ALIASES:
        first = rng.choice(NAME_ALIASES[first])
    elif mode == "translit":
        target = first if rng.random() < 0.6 else surname
        low = target.lower()
        applicable = [(a, b) for a, b in TRANSLIT_RULES if a in low]
        if applicable:
            a, b = rng.choice(applicable)
            new = low.replace(a, b, 1).capitalize()
            if target == first:
                first = new
            else:
                surname = new
    elif mode == "typo":
        if rng.random() < 0.5:
            first = _typo(rng, first)
        else:
            surname = _typo(rng, surname)
    elif mode == "initial":
        first = first[0] + "."
    elif mode == "reorder":
        first, surname = surname, first

    return f"{first} {surname}".strip()


def corrupt_phone(rng: random.Random, phone: str, hard: bool) -> str:
    if hard:
        return fresh_phone(rng)
    return _typo(rng, phone, preserve_length=True)


def corrupt_business(rng: random.Random, business: str, hard: bool) -> str:
    applicable = [(a, b) for a, b in BUSINESS_REWRITES if a in business]
    mode = rng.random()
    if applicable and mode < 0.45:
        a, b = rng.choice(applicable)
        return business.replace(a, b)
    if mode < 0.65 and "& Sons" not in business:
        return business + " & Sons"
    if mode < 0.8 and hard:
        tokens = business.split()
        return f"{rng.choice(BUSINESS_HEADS)} {tokens[-1]}"
    if hard:
        return f"{rng.choice(BUSINESS_HEADS)} {rng.choice(BUSINESS_TAILS)}"
    return _typo(rng, business)


def corrupt_address(rng: random.Random, address: str, hard: bool) -> str:
    parts = [p.strip() for p in address.split(",")]
    mode = rng.random()
    if mode < 0.4:
        for a, b in STREET_ABBREV:
            if a in address:
                return address.replace(a, b)
    if mode < 0.6 and len(parts) > 2:
        return ", ".join(parts[1:])
    if mode < 0.8 and len(parts) > 1:
        parts[0], parts[-1] = parts[-1], parts[0]
        return ", ".join(parts)
    if hard:
        return fresh_address(rng)
    return _typo(rng, address)


def corrupt_pincode(rng: random.Random, pincode: str) -> str:
    delta = rng.choice([-9, -3, -1, 1, 3, 9])
    return f"{max(100000, int(pincode) + delta)}"


def apply_evasion(
    rng: random.Random, base: Identity, level: int
) -> tuple[Identity, list[str]]:
    """Wrapper enforcing that evasion actually alters the identity.

    A corruption that silently no-ops produces a record labelled as evasion
    while being behaviourally an honest re-application, which poisons the
    exact distinction this project exists to make.
    """
    for _ in range(8):
        new_id, changed = _apply_evasion(rng, base, level)
        if new_id != base:
            return new_id, changed
    forced = _typo(rng, base.name.split()[0]) + " " + base.name.split()[-1]
    return sibling_identity(rng, base, name=forced), ["name"]


def _apply_evasion(
    rng: random.Random, base: Identity, level: int
) -> tuple[Identity, list[str]]:
    """Derive a new identity from `base` at a given evasion level.

    Returns the new identity and the list of fields that were altered.
    Level semantics:
      1  one soft field, lightly corrupted
      2  two or three soft fields; phone replaced
      3  every soft field reworked; PAN and GSTIN still intact
      4  new PAN too, but the stale GSTIN still carries the original PAN
      5  everything independent except one weak residual field
    """
    changed: list[str] = []
    state_code = CITIES[base.city][0]

    soft_fields = ["name", "business_name", "address", "email", "pincode"]

    if level == 1:
        field = rng.choice(soft_fields[:3])
        if field == "name":
            new = sibling_identity(rng, base, name=corrupt_name(rng, base.name, False))
        elif field == "business_name":
            new = sibling_identity(
                rng, base, business_name=corrupt_business(rng, base.business_name, False)
            )
        else:
            new = sibling_identity(
                rng, base, address=corrupt_address(rng, base.address, False)
            )
        changed.append(field)
        return new, changed

    if level == 2:
        picks = rng.sample(soft_fields[:4], k=rng.randint(2, 3))
        upd: dict = {"phone": corrupt_phone(rng, base.phone, hard=True)}
        changed.append("phone")
        for f in picks:
            if f == "name":
                upd["name"] = corrupt_name(rng, base.name, False)
            elif f == "business_name":
                upd["business_name"] = corrupt_business(rng, base.business_name, False)
            elif f == "address":
                upd["address"] = corrupt_address(rng, base.address, False)
            elif f == "email":
                upd["email"] = fresh_email(rng, base.name)
            changed.append(f)
        return sibling_identity(rng, base, **upd), changed

    if level == 3:
        upd = {
            **({"settlement_account": fresh_account(rng)} if rng.random() < 0.5 else {}),
            "name": corrupt_name(rng, base.name, True),
            "business_name": corrupt_business(rng, base.business_name, True),
            "address": corrupt_address(rng, base.address, True),
            "phone": fresh_phone(rng),
            "email": fresh_email(rng, base.name),
        }
        upd["pincode"] = corrupt_pincode(rng, base.pincode)
        changed += list(upd)
        return sibling_identity(rng, base, **upd), changed

    if level == 4:
        name = corrupt_name(rng, base.name, True)
        upd = {
            "settlement_account": fresh_account(rng),
            "name": name,
            "business_name": f"{rng.choice(BUSINESS_HEADS)} {rng.choice(BUSINESS_TAILS)}",
            "address": fresh_address(rng),
            "phone": fresh_phone(rng),
            "email": fresh_email(rng, name),
            "pincode": corrupt_pincode(rng, base.pincode),
            "pan": fresh_pan_for(rng, base),
            "gstin": gstin_from_pan(rng, base.pan, state_code),
        }
        changed += list(upd)
        return sibling_identity(rng, base, **upd), changed

    # level 5: only one weak residual link survives
    name = _strong_name_change(rng, base.name)
    new_pan = fresh_pan_for(rng, base)
    residual = rng.choice(["pincode", "address_locality", "none"])
    address = fresh_address(rng)
    if residual == "address_locality":
        address = address.rsplit(",", 1)[0] + ", " + base.address.rsplit(",", 1)[-1].strip()
    upd = {
        "name": name,
        "settlement_account": fresh_account(rng),
        "settlement_ifsc": fresh_ifsc(rng),
        "business_name": f"{rng.choice(BUSINESS_HEADS)} {rng.choice(BUSINESS_TAILS)}",
        "address": address,
        "phone": fresh_phone(rng),
        "email": fresh_email(rng, name),
        "pan": new_pan,
        "gstin": gstin_from_pan(rng, new_pan, state_code),
        "pincode": base.pincode if residual == "pincode" else corrupt_pincode(rng, base.pincode),
    }
    changed += list(upd)
    return sibling_identity(rng, base, **upd), changed


def _strong_name_change(rng: random.Random, name: str) -> str:
    """Level-5 names must not remain recognisable through a single typo."""
    for _ in range(6):
        candidate = corrupt_name(rng, name, hard=True)
        if candidate.split()[0] != name.split()[0]:
            return candidate
    first, *rest = name.split()
    return f"{first[0]}. {rest[-1] if rest else ''}".strip()


def natural_drift(rng: random.Random, base: Identity) -> tuple[Identity, list[str]]:
    """Legitimate change over time, which is not concealment.

    A real applicant moves house, changes number, updates an email. What they
    do NOT do is acquire a new PAN: a PAN is issued once and never changes.
    That asymmetry is the honest discriminator, rather than "did anything
    change at all", which only holds when legitimate records are frozen.

    Drift is narrow (one or two fields) where concealment is broad.
    """
    changed: list[str] = []
    upd: dict = {}
    n = rng.choices([0, 1, 2], weights=[0.25, 0.5, 0.25])[0]
    for field in rng.sample(["phone", "address", "email", "name"], k=n):
        if field == "phone":
            upd["phone"] = fresh_phone(rng)
        elif field == "address":
            upd["address"] = (
                corrupt_address(rng, base.address, False)
                if rng.random() < 0.5 else fresh_address(rng)
            )
        elif field == "email":
            upd["email"] = fresh_email(rng, base.name)
        else:
            upd["name"] = corrupt_name(rng, base.name, False)
        changed.append(field)
    return sibling_identity(rng, base, **upd), changed