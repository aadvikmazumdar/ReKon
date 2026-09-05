# ReKon

**A blocklisted merchant reapplies as someone new. ReKon follows the chain,
weighs the evidence in bits, and says where it stops working.**

Built for Razorpay's Buildathon, Track 02 — AI Risk Manager.

https://re-kon.vercel.app

---

## The problem

Razorpay Capital lends to merchants. When one turns out to be a fraud, they get
blocklisted — and then they apply again as somebody else. New phone, name
spelled differently, business renamed, sometimes a fresh PAN.

Razorpay's own engineering team published this: their dedupe engine flags ~5% of
applications and has stopped roughly **₹30 crore of bad loans**.

## Why the obvious solution fails

Find applications matching a blocklisted one, block them. Now consider who else
that catches:

- a founder who genuinely owns two businesses
- a company that rebranded
- a couple sharing a phone number
- someone rejected for thin credit history, reapplying openly six months later

All look like matches. None is fraud. And a false positive here is a real
merchant denied credit.

So ReKon asks a different question:

> **Is this applicant concealing a connection to a decision we already made?**

Linkage is the evidence. Concealment is the signal.

---

## Results

Held-out test set of 2,391 applications, split by operator so no fraud ring
straddles the boundary.

| Metric | |
|---|---|
| **Precision / recall** | **0.769 / 0.824** |
| Linkage recall at 100% precision | **90.8%** |
| Cluster purity | **1.000** — 0 contaminated of 376 |
| False blocks on legitimate merchants | **0.9%** |
| Loss avoided on test data | **₹2.05 crore** |

**Where it breaks, measured deliberately:**

| Evasion effort | What survives the disguise | Rings recovered |
|---|---|---|
| Level 1 | one field lightly corrupted | 100% |
| Level 2 | phone replaced, 2–3 fields changed | 100% |
| Level 3 | every soft field reworked, PAN intact | 100% |
| Level 4 | new PAN — but a stale GSTIN carries the old one | 98.9% |
| Level 5 | nothing carries over | 45.6% |

The data is synthetic, so "you built the test and passed it" is fair. The answer
is that controlling the generator is what lets me dial evasion effort up and
measure the failure point. That isn't an accuracy claim, it's a breaking point,
and it's only measurable *because* the data is synthetic.

---

## What makes it work

### The fraud is the group, not the pair

Fraudsters run several identities, each derived from the last, so the changes
accumulate:

```
2024-02-13  Neha Mehta   blocklisted
2024-03-17  Neha Mahta   rejected
2024-05-18  Neah Mahta   approved
2024-11-18  Nzah Mahta   approved
```

By the end: different PAN, phone, settlement account, address, pincode. The
first and last records share **nothing**. Compare them pairwise and you see two
strangers.

ReKon builds a graph of confident links and takes connected components, so the
whole chain surfaces from the adjacent hops. The last application links back at
15.4 bits of evidence.

### The GSTIN slip

A GSTIN structurally contains the holder's PAN at characters 3–12:

```
27  ABCDE1234F  1  Z  5
│   └ the PAN
└ state code
```

An operator can buy a new PAN. If they reuse an old GSTIN, it still carries the
original PAN inside it. Generic record-linkage tools don't know this
relationship exists — it's why level 4 evasion is still caught 98.9% of the time.

The other half: **a person's PAN never changes.** Legitimate drift moves phones
and addresses, not PANs. So a matching PAN carries a *negative* weight in the
risk model — it argues nothing was hidden.

### Evidence weights that aren't made up

I started with hand-picked weights and it bothered me that I'd invented them.
**Fellegi-Sunter** (1969) is the standard record-linkage model, what Splink and
`recordlinkage` implement. For each field it estimates `m` (agreement when
records truly match) and `u` (agreement by chance); evidence is `log2(m/u)` in
bits.

It fits by expectation-maximisation with **no labels**, so it consumes none of
the ground truth. And `u` handles name frequency automatically — a field that
collides by accident earns less, with no rule saying so.

90.8% linkage recall against 88.7% for my hand-tuned weights, with the operating
threshold landing at −4.1 bits, essentially the theoretically correct zero.

### An honest failure boundary

ReKon blocks **50% of honest re-applications**. That's a real failure, and here's
why it can't be fixed with better features:

| | fields altered | PAN match | account match |
|---|---|---|---|
| honest re-application | 0.57 | 1.00 | 1.00 |
| evasion, level 1–2 | 0.81 | 1.00 | 1.00 |

Statistically indistinguishable. Both keep their hard identifiers, change one or
two soft fields, follow a rejection. The difference is *intent*, and no feature
observes intent.

That band routes to a human. Showing the boundary beats hiding it inside an
average.

---

## How it runs

```
applications (time-ordered)
   ├─ blocking             160k candidate pairs from 32M possible, 200× reduction
   ├─ pairwise evidence    22 comparison features
   ├─ Fellegi-Sunter       linkage weights fitted without labels
   ├─ graph clustering     connected components → operator-level rings
   ├─ logistic regression  cluster + link features
   └─ PASS / REVIEW / BLOCK with a readable evidence report
```

**Logistic regression, deliberately.** The features are already engineered, there
are ~300 positives, and the coefficients *are* the explanation. XGBoost would win
a point or two of AUC, then need SHAP to reconstruct why a merchant got blocked —
an approximation of the reasoning rather than the reasoning. Coefficient signs are
asserted on every build; if an exculpatory feature turns positive, the build fails.

**Thresholds from decision theory, not tuning.** Blocking is worth it when
`p × cost(missed evader) > (1−p) × cost(blocked merchant)` → `p > 0.077`, closed
form. Review is different: the economic threshold says review almost everything,
which is mathematically right and operationally useless, so review comes from an
underwriter queue budget instead.

**40 automated checks** on every run — format validity, label integrity, temporal
causality, leakage probes, and a test that payment aggregates carry no label
signal (Cohen's d 0.006–0.063).

---

## Stack

**Python** does the thinking. `pandas`, `numpy`, `jellyfish` (Jaro-Winkler,
Soundex), `networkx` (connected components), `scikit-learn`. Fellegi-Sunter with
EM is written from scratch — 130 lines, no library.

**React 18 + TypeScript + Vite** for the interface, `recharts` for plots,
hand-written SVG for the ring graph.

**No backend, by design.** The pipeline exports one `data.json` and the app reads
it. Nothing is scored at request time, so there's no server to cold-start or fall
over mid-demo, and it deploys as a static site.

```bash
pip install -r requirements.txt
python3 -m rekon.pipeline --n 8000     # full run, ~25s
python3 -m rekon.demo                  # evidence reports in the terminal
python3 -m rekon.export_web            # refresh web data

cd web && npm install && npm run dev
```

---

## Limitations

- The label needs a prior rejection, so a ring where nothing was ever flagged
  clusters correctly but scores low
- Operators using genuinely different people's KYC are invisible by construction
- No feedback loop from underwriter decisions back into scoring
- It routes, it doesn't remediate — no auto-blocklisting, no case management
- Base rate, ring sizes and corruption patterns are reasoned assumptions, not
  measured from real fraud

The corruption engine generates synthetic test data only. It creates fake
identities inside my own dataset, never touches a live system, and encodes only
obvious alterations. It exists to measure where detection fails.
