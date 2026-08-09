# Settlement Feasibility & Fee Engine — Take-home

Welcome, and thanks for taking the time. The full problem is in
[`ASSIGNMENT.md`](./ASSIGNMENT.md). This README is just orientation.

## The task in one line 

Given a client's escrow account, a settlement offer, and a creditor's rules,
decide whether the offer is affordable (and schedule it, collecting our fee as
early as allowed) or — if not — compute the minimum extra funding needed.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
hiring_takehome/
├── ASSIGNMENT.md            # full specification — read this
├── feasibility/
│   ├── models.py            # data models, JSON loaders, date/EOM helpers (provided)
│   └── engine.py            # >>> implement evaluate_offer here <<< (+ Result shape)
├── cases/                   # four example cases (client.json / offer.json / creditor_rules.json)
│   ├── case1_feasible_even
│   ├── case2_infeasible_minima
│   ├── case3_balloon
│   └── case4_tiers
├── tests/
│   ├── test_smoke.py        # scaffolding sanity tests (pass out of the box)
│   └── test_cases.py        # example expectations — make these pass, then add your own
├── run.py                   # python run.py cases/<case>
└── requirements.txt
```

## Run

```bash
# evaluate a single case (prints the Result as JSON)
python run.py cases/case1_feasible_even

# tests
pytest -q
```

Out of the box, `tests/test_smoke.py` passes and `tests/test_cases.py` fails —
the latter is your target. Go beyond those four cases with your own tests.

## What to submit

Your implementation, your tests, and a short README section describing:
- your approach and the alternatives you considered,
- **your interpretation of the payment shapes** (even / staircase / balloon — we
  left these loosely defined on purpose),
- assumptions you made, and known edge cases / limitations.

Budget ~5–6 hours. Prefer a correct, well-tested core over breadth. When in
doubt, write down your assumption and keep going.

---

# Submission — implementation notes

All logic lives in `feasibility/engine.py` (the only file the task asks you to
implement); `feasibility/models.py` gained a `round_half_up` helper and the
money helpers now use it. Run `pytest -q` (10 provided + 16 of my own = 26
tests) or `python run.py cases/<case>`.

## Approach

I split the problem into three layers:

1. **A date-by-date ledger simulator.** Everything is integer cents. I build a
   list of dated events (future ledger entries, creditor payments, bank fees,
   program fees), bucket them by date, and on each date apply **all credits
   before all debits**. End-of-day balance is the relevant minimum for a day, so
   non-negativity only needs checking there. The horizon (`last_draft_date`) is
   inclusive; cadence dates are generated with the provided EOM/clamp helpers
   and truncated at the horizon.

2. **A candidate generator per shape** that, for a given payment count `k`,
   emits valid creditor-payment vectors (floors, non-decreasing, exact sum,
   shape rules).

3. **A fee placer + objective.** For a *fixed* creditor schedule, the program
   fee is the only free variable. I observe that the program fee and creditor
   payments both drain the same account, so the cleanest way to express
   "front-load the fee" is:

   > Let `B(t)` be the running balance assuming **no** fee. The largest
   > cumulative fee we can have collected by cadence date `dᵢ` without ever
   > going negative is `C(dᵢ) = min(F, min_{t ≥ dᵢ} B(t))`.

   This is exactly the greedy "collect as much fee as you can now without
   starving a future mandatory creditor payment" rule, and it is provably the
   front-loaded optimum **for that schedule** (`C` is non-decreasing because the
   suffix-minimum is). An offer is feasible iff some schedule keeps `B(t) ≥ 0`
   everywhere *and* `C` reaches the full fee `F` by the last cadence date.

   I then search over `k` (and, for staircases, over segment layouts) and keep
   the schedule whose cumulative-fee vector `(C(d₁), C(d₂), …)` is
   **lexicographically largest** — i.e. the one that collects the most fee
   earliest. Because fee front-loading rewards a high early balance, the optimum
   naturally keeps early creditor payments low and defers the big ones, so the
   staircase/balloon shapes *fall out of the objective* rather than being
   hard-coded.

### Alternatives considered

- **MILP / OR-Tools.** A mixed-integer program would model this cleanly, but
  it is overkill for `k ≤ 12`, adds a heavy dependency, and obscures the
  reasoning the task wants to see. The structured search is fast and auditable.
- **One greedy pass over `k`.** Tempting, but the segment cap and the
  exact-sum + integrality constraints interact, so I enumerate segment layouts
  (only `≤ 2^{k-1}` of them) and let the objective choose. Tiny in practice.

## My interpretation of the payment shapes

The `Result.pay_shape_used` is driven by the flags: `even_pays → "even"`, else
`is_ballooning_allowed → "balloon"`, else `"staircase"`. Within each:

- **Even** — all payments equal; when `offer_total` isn't divisible by `k`, the
  remainder cents go onto the **latest** payments (stays non-decreasing,
  "as equal as possible"). I pick the `k` that best front-loads the fee.
- **Balloon** — payments `1..k-1` sit at their **floors** (minimum allowed, so
  the early account drain is as small as possible) and the final payment absorbs
  the entire remainder. `max_segments` is ignored, per the spec. Token pays /
  tiers still bind the small early payments; the balloon itself just has to be
  `≥` the previous payment and `≥` its own floor. The objective prefers more
  small early payments (a "truer" balloon) when the fee is indifferent.
- **Staircase** — at most `max_segments` distinct levels. I model a staircase as
  consecutive equal-valued runs (segments). For a chosen layout, the first
  `s-1` segments take their minimal feasible flat level and the **last** segment
  absorbs the remainder — this is the front-loaded choice and keeps the step-up
  late. A layout is discarded if the last segment can't hit the exact sum with
  an integer level `≥` the previous level and its floor.

### Floors (the `max`-of-three rule)

Per position `i` (1-based), `floor(i) = max(base_min, tier_floor(i),
token_floor(i))` where `token_floor(i) = base_min + 1` for `i > max_token_pays`
(so beyond the token budget a payment must *strictly* exceed the base) and `0`
otherwise. Because each component is non-decreasing in `i`, the floor sequence
is non-decreasing, which dovetails with the non-decreasing payment requirement.

## Part 2 — minimum additional funds

Feasibility is monotone in money (more cash never hurts), so each minimum is a
**binary search** over the same feasibility oracle:

- **Lump sum `L`** — a single extra credit. Since an earlier lump is weakly more
  useful, I place it on the **earliest future draft date** (clamped to the
  horizon) and search for the smallest `L`.
- **Monthly increment `X`** — added to **every** future draft (each draft dated
  after `as_of_date`); `num_drafts` counts all of them, even ones that land too
  late to help. This is why the lump and increment can imply different totals
  (e.g. case 2: lump 10000 vs increment 2500 × 5 = 12500 — the last draft
  arrives after the final cadence date and is wasted).

Guardrails: reject the increment if `X > max(10000, round(0.40 × draft))`;
reject the lump if `L > round(0.65 × offer_total)`. Both are reported with a
boolean and a reason string.

## Assumptions

- **Field name.** `ASSIGNMENT.md` mentions `creditor_balance_cents`, but the
  scaffolding (`models.py`, the case JSON, and `offer_total_cents`) uses
  `current_balance_cents` on the offer. I followed the code.
- **Rounding.** `round(...)` means round-half-up (away from zero), implemented
  with `Decimal`; the provided money helpers were switched over to it.
- **"Strictly exceed" for token pays** means `≥ base + 1` cent.
- **Increment applies to credits** (drafts) only; existing debits are fixed.
- **Fee is collected only on cadence dates**, never before the first creditor
  payment date (same date allowed), and fully by the horizon.
- Balance is checked at end-of-day (after credits-then-debits), which is the
  daily minimum.

## Known edge cases / limitations

- If `offer_total` is smaller than a single minimum payment (so even `k=1` fails
  its floor), the offer is infeasible and **no** amount of funding fixes it
  (`offer_total` doesn't change); Part 2 then reports `within_guardrail=False`
  with an explanatory reason.
- The staircase search is exponential in `k` per segment count
  (`≤ 2^{k-1}` layouts), which is fine for `k ≤ 12` but would need a smarter
  enumeration for much larger caps.
- When the program fee is `0`, all candidates tie on the objective; the
  shape-aware tie-break (more/smaller payments for balloon, fewer otherwise)
  picks a canonical schedule.
