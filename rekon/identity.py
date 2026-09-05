"""Indian identity field primitives.

PAN and GSTIN are structurally linked in reality: a GSTIN embeds the holder's
PAN at positions 2..12. That structural fact is a genuine high-value evidence
signal for ReKon, so it is modelled faithfully here.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, replace

LETTERS = string.ascii_uppercase

FIRST_NAMES_M = [
    "Aarav", "Aditya", "Akash", "Amit", "Anand", "Arjun", "Ashok", "Deepak",
    "Gaurav", "Harish", "Imran", "Jatin", "Karthik", "Kiran", "Lokesh",
    "Mahesh", "Manoj", "Mohammed", "Naveen", "Nikhil", "Pankaj", "Praveen",
    "Rahul", "Rajesh", "Rakesh", "Ramesh", "Ravi", "Rohit", "Sandeep",
    "Sanjay", "Santosh", "Sathish", "Suresh", "Sunil", "Vikram", "Vinod",
    "Vijay", "Yogesh", "Prakash", "Girish",
]

FIRST_NAMES_F = [
    "Aishwarya", "Ananya", "Anjali", "Deepika", "Divya", "Geetha", "Kavya",
    "Lakshmi", "Meena", "Nandini", "Neha", "Pooja", "Priya", "Radha",
    "Rashmi", "Rekha", "Sangeetha", "Shalini", "Shruti", "Sneha", "Sowmya",
    "Sudha", "Swathi", "Uma", "Vidya", "Yamini", "Kalpana", "Bhavana",
]

SURNAMES = [
    "Agarwal", "Bhatt", "Chauhan", "Desai", "Gupta", "Iyer", "Jain", "Joshi",
    "Kapoor", "Khan", "Kumar", "Malhotra", "Menon", "Mehta", "Nair", "Patel",
    "Pillai", "Rao", "Reddy", "Sharma", "Shah", "Singh", "Sinha", "Subramanian",
    "Thakur", "Trivedi", "Varma", "Verma", "Yadav", "Bose", "Chatterjee",
    "Das", "Ghosh", "Mukherjee", "Naidu", "Prasad", "Saxena", "Shetty",
]

# city -> (state_code for GSTIN, pincode prefix)
CITIES = {
    "Mumbai": ("27", "400"),
    "Pune": ("27", "411"),
    "Nagpur": ("27", "440"),
    "Bengaluru": ("29", "560"),
    "Mysuru": ("29", "570"),
    "Chennai": ("33", "600"),
    "Coimbatore": ("33", "641"),
    "Madurai": ("33", "625"),
    "Hyderabad": ("36", "500"),
    "Delhi": ("07", "110"),
    "Gurugram": ("06", "122"),
    "Noida": ("09", "201"),
    "Kolkata": ("19", "700"),
    "Ahmedabad": ("24", "380"),
    "Surat": ("24", "395"),
    "Jaipur": ("08", "302"),
    "Lucknow": ("09", "226"),
    "Kochi": ("32", "682"),
    "Indore": ("23", "452"),
    "Chandigarh": ("04", "160"),
}

LOCALITIES = [
    "Anna Nagar", "Adyar", "Velachery", "T Nagar", "Indiranagar", "Koramangala",
    "Whitefield", "Jayanagar", "Andheri East", "Bandra West", "Borivali",
    "Powai", "Kothrud", "Baner", "Hinjewadi", "Banjara Hills", "Gachibowli",
    "Madhapur", "Saket", "Rohini", "Dwarka", "Salt Lake", "Ballygunge",
    "Navrangpura", "Satellite", "Vaishali Nagar", "Malviya Nagar",
    "Gomti Nagar", "Kakkanad", "Vijay Nagar",
]

STREET_TYPES = ["Road", "Street", "Main Road", "Cross Street", "Avenue", "Lane"]

BUSINESS_HEADS = [
    "Sri", "Shree", "New", "Royal", "Global", "Prime", "Star", "Unity",
    "Sunrise", "Metro", "National", "Supreme", "Elite", "Fortune", "Crown",
]

BUSINESS_TAILS = [
    "Traders", "Enterprises", "Industries", "Textiles", "Electronics",
    "Agencies", "Distributors", "Solutions", "Retail", "Foods", "Exports",
    "Motors", "Hardware", "Pharma", "Logistics", "Services",
]

ENTITY_TYPE_CODES = "PCFHT"  # PAN 4th char: P=individual, C=company, F=firm...

EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com"]

BANKS = ["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "IDFB", "YESB", "PUNB", "BARB"]

# Merchant category codes, roughly matching the small-business mix Capital lends to
MCCS = ["5411", "5651", "5732", "5812", "5912", "7372", "4214", "5399",
        "8299", "5045", "5122", "7299"]


@dataclass(frozen=True)
class Identity:
    """One coherent applicant identity. Applications are drawn from these."""

    name: str
    settlement_account: str
    settlement_ifsc: str
    mcc: str
    phone: str
    pan: str
    gstin: str
    business_name: str
    address: str
    city: str
    pincode: str
    email: str


def _pan(rng: random.Random, surname: str) -> str:
    """PAN format AAAAA9999A. 4th char = entity type, 5th = surname initial."""
    head = "".join(rng.choice(LETTERS) for _ in range(3))
    entity_type = rng.choices(ENTITY_TYPE_CODES, weights=[60, 20, 12, 4, 4])[0]
    surname_initial = surname[0].upper()
    digits = f"{rng.randint(0, 9999):04d}"
    check = rng.choice(LETTERS)
    return f"{head}{entity_type}{surname_initial}{digits}{check}"


def gstin_from_pan(rng: random.Random, pan: str, state_code: str) -> str:
    """GSTIN = 2-digit state + 10-char PAN + entity number + 'Z' + checksum."""
    entity_no = str(rng.randint(1, 9))
    check = rng.choice(LETTERS + string.digits)
    return f"{state_code}{pan}{entity_no}Z{check}"


def pan_from_gstin(gstin: str) -> str | None:
    """Extract the embedded PAN. This is the structural evidence signal."""
    if not gstin or len(gstin) != 15:
        return None
    return gstin[2:12]


def _account(rng: random.Random) -> str:
    """Settlement account. Harder to churn than a phone: opening a current
    account needs KYC, so reuse across identities is a strong linkage signal."""
    # Leading digit is non-zero so the value survives a CSV round-trip as an
    # integer without silently losing a character.
    n = rng.choice([11, 12, 14, 16])
    return str(rng.randint(1, 9)) + "".join(
        str(rng.randint(0, 9)) for _ in range(n - 1)
    )


def _ifsc(rng: random.Random) -> str:
    return f"{rng.choice(BANKS)}0{rng.randint(100000, 999999)}"


def fresh_account(rng: random.Random) -> str:
    return _account(rng)


def fresh_ifsc(rng: random.Random) -> str:
    return _ifsc(rng)


def _phone(rng: random.Random) -> str:
    return str(rng.choice("6789")) + "".join(
        str(rng.randint(0, 9)) for _ in range(9)
    )


def _address(rng: random.Random) -> str:
    door = f"{rng.randint(1, 199)}"
    if rng.random() < 0.45:
        door = f"{rng.randint(1, 40)}/{rng.randint(1, 99)}"
    parts = []
    if rng.random() < 0.5:
        parts.append(f"Flat {rng.choice('ABCD')}-{rng.randint(1, 30)}")
    parts.append(f"{door} {rng.randint(1, 20)}th {rng.choice(STREET_TYPES)}")
    parts.append(rng.choice(LOCALITIES))
    return ", ".join(parts)


def _email(rng: random.Random, name: str) -> str:
    tokens = [t.lower() for t in name.replace(".", "").split() if t]
    style = rng.random()
    if style < 0.4:
        local = "".join(tokens)
    elif style < 0.7:
        local = ".".join(tokens)
    else:
        local = tokens[0] + str(rng.randint(1, 999))
    if rng.random() < 0.35:
        local += str(rng.randint(1, 99))
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def _business_name(rng: random.Random, surname: str) -> str:
    style = rng.random()
    tail = rng.choice(BUSINESS_TAILS)
    if style < 0.45:
        return f"{surname} {tail}"
    if style < 0.75:
        return f"{rng.choice(BUSINESS_HEADS)} {tail}"
    return f"{rng.choice(BUSINESS_HEADS)} {surname} {tail}"


def new_identity(rng: random.Random, city: str | None = None) -> Identity:
    """A fresh, unrelated applicant identity."""
    city = city or rng.choice(list(CITIES))
    state_code, pin_prefix = CITIES[city]

    first = rng.choice(FIRST_NAMES_M if rng.random() < 0.72 else FIRST_NAMES_F)
    surname = rng.choice(SURNAMES)
    name = f"{first} {surname}"

    pan = _pan(rng, surname)
    return Identity(
        name=name,
        settlement_account=_account(rng),
        settlement_ifsc=_ifsc(rng),
        mcc=rng.choice(MCCS),
        phone=_phone(rng),
        pan=pan,
        gstin=gstin_from_pan(rng, pan, state_code),
        business_name=_business_name(rng, surname),
        address=_address(rng),
        city=city,
        pincode=pin_prefix + f"{rng.randint(0, 999):03d}",
        email=_email(rng, name),
    )


def sibling_identity(rng: random.Random, base: Identity, **overrides) -> Identity:
    """A related identity: copy of `base` with explicit field overrides."""
    return replace(base, **overrides)


def fresh_pan_for(rng: random.Random, identity: Identity) -> str:
    surname = identity.name.split()[-1]
    return _pan(rng, surname)


def fresh_phone(rng: random.Random) -> str:
    return _phone(rng)


def fresh_address(rng: random.Random) -> str:
    return _address(rng)


def fresh_email(rng: random.Random, name: str) -> str:
    return _email(rng, name)