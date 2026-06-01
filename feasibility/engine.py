"""Candidate implementation goes here.

Implement ``evaluate_offer`` so that it satisfies the rules in ASSIGNMENT.md and
the example expectations in tests/test_cases.py. The dataclasses below define the
required OUTPUT shape (see ASSIGNMENT.md "Output"). You may add helpers, modules,
or rewrite internals freely, but keep ``evaluate_offer``'s signature and the
serialized shape of ``Result`` (so the runner and tests work).
"""

from __future__ import annotations

import itertools
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from feasibility.models import (
    Client,
    CreditorRules,
    Offer,
    add_months,
    default_first_payment_date,
    end_of_month,
    is_end_of_month,
    offer_total_cents,
    program_fee_cents,
)


@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    # lump-sum only:
    date: date | None = None
    # monthly-increment only:
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    # One of "even", "staircase", or "balloon" — the shape your solution produced
    # (driven by the creditor flags). None when infeasible.
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        out: dict = {"feasible": self.feasible, "pay_shape_used": self.pay_shape_used}
        out["schedule"] = (
            [
                {
                    "date": r.date.isoformat(),
                    "creditor_payment_cents": r.creditor_payment_cents,
                    "program_fee_cents": r.program_fee_cents,
                    "bank_fee_cents": r.bank_fee_cents,
                    "balance_cents": r.balance_cents,
                }
                for r in self.schedule
            ]
            if self.schedule is not None
            else None
        )
        if self.additional_funds is None:
            out["additional_funds"] = None
        else:
            def opt(o: FundsOption) -> dict:
                d = {
                    "amount_cents": o.amount_cents,
                    "within_guardrail": o.within_guardrail,
                    "reason": o.reason,
                }
                if o.date is not None:
                    d["date"] = o.date.isoformat()
                if o.num_drafts is not None:
                    d["num_drafts"] = o.num_drafts
                return d

            out["additional_funds"] = {
                "lump_sum": opt(self.additional_funds.lump_sum),
                "monthly_increment": opt(self.additional_funds.monthly_increment),
            }
        return out


# ---------------------------------------------------------------------------
# Internal solver
#
# High-level model (see README "Approach"):
#
#   * The creditor payments live on consecutive cadence dates d_1..d_k. Their
#     dates are fixed; we only choose the count k and the per-payment amounts
#     (constrained by floors, non-decreasing, exact sum, and the shape flag).
#   * The program fee is flexible: it may be poured onto any cadence date in
#     d_1..d_M (M = number of cadence dates on/before the horizon), as long as
#     the running balance never goes negative and the full fee is collected by
#     the horizon.
#   * Objective: collect the fee as early as possible. Given a fixed creditor
#     schedule, the earliest-possible cumulative fee at cadence date d_i is
#         C(d_i) = min(F, min_{t >= d_i} B(t))
#     where B(t) is the running balance WITHOUT any fee. This is the greedy
#     "take as much fee as you can without starving a future creditor payment"
#     rule, and it is provably the front-loaded optimum for a given schedule.
#   * We then search over k (and, for staircases, over segment layouts) and keep
#     the schedule whose cumulative-fee vector is lexicographically largest.
# ---------------------------------------------------------------------------

_Event = tuple[date, int, bool]  # (date, amount_cents, is_credit)


def _round_half_up(value) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _shape(rules: CreditorRules) -> str:
    if rules.even_pays:
        return "even"
    if rules.is_ballooning_allowed:
        return "balloon"
    return "staircase"


def _cadence_dates(first_payment: date, horizon: date) -> list[date]:
    """All cadence dates on or before the horizon (creditor payments + fees)."""
    eom = is_end_of_month(first_payment)
    out: list[date] = []
    for i in range(1200):  # generous upper bound; we break at the horizon
        d = add_months(first_payment, i)
        if eom:
            d = end_of_month(d)
        if d > horizon:
            break
        out.append(d)
    return out


def _floors(k: int, rules: CreditorRules) -> list[int]:
    """Per-position minimum payment (1-based), already non-decreasing.

    Combines: the base minimum; the token-pay rule (positions beyond
    ``max_token_pays`` may not sit *at* the base, so they are bumped to
    base + 1); and any ``min_payment_tiers`` step-ups.
    """
    floors: list[int] = []
    for i in range(1, k + 1):
        f = rules.min_payment_cents
        for frm, min_cents in rules.min_payment_tiers:
            if i >= frm:
                f = max(f, min_cents)
        if i > rules.max_token_pays:
            f = max(f, rules.min_payment_cents + 1)
        floors.append(f)
    return floors


def _valid_vector(vec: list[int] | None, floors: list[int], rules: CreditorRules) -> bool:
    if vec is None:
        return False
    if any(vec[i] < floors[i] for i in range(len(vec))):
        return False
    if any(vec[i] < vec[i - 1] for i in range(1, len(vec))):
        return False
    # token pays: at most max_token_pays payments may equal the base minimum.
    at_base = sum(1 for x in vec if x == rules.min_payment_cents)
    if at_base > rules.max_token_pays:
        return False
    return True


def _even_vector(k: int, total: int) -> list[int]:
    """Equal payments; spread the remainder cents onto the latest payments."""
    base, rem = divmod(total, k)
    vec = [base] * k
    for i in range(k - rem, k):
        vec[i] += 1
    return vec


def _balloon_vector(k: int, total: int, floors: list[int]) -> list[int] | None:
    """Minimum payments early, one final payment absorbing the remainder."""
    if k == 1:
        return [total]
    vec = list(floors[: k - 1])
    last = total - sum(vec)
    return vec + [last]


def _compositions(k: int, s: int):
    """All ways to split ``k`` consecutive positions into ``s`` non-empty runs."""
    if s == 1:
        yield (k,)
        return
    for splits in itertools.combinations(range(1, k), s - 1):
        prev = 0
        parts = []
        for sp in splits:
            parts.append(sp - prev)
            prev = sp
        parts.append(k - prev)
        yield tuple(parts)


def _segment_levels(parts: tuple[int, ...], floors: list[int], total: int) -> list[int] | None:
    """Front-loaded level assignment for a fixed segment layout.

    Each of the first s-1 segments is set to its minimal feasible (flat) level;
    the final segment absorbs whatever is left, which front-loads the schedule.
    Returns None when the layout cannot hit the exact sum with integer levels.
    """
    s = len(parts)
    group_floor: list[int] = []
    idx = 0
    for p in parts:
        group_floor.append(max(floors[idx : idx + p]))
        idx += p

    levels: list[int] = []
    fixed = 0
    for g in range(s - 1):
        v = group_floor[g] if not levels else max(group_floor[g], levels[-1])
        levels.append(v)
        fixed += v * parts[g]

    last_len = parts[-1]
    rem = total - fixed
    if rem <= 0 or rem % last_len != 0:
        return None
    v_last = rem // last_len
    if levels and v_last < levels[-1]:
        return None
    if v_last < group_floor[-1]:
        return None
    levels.append(v_last)

    vec: list[int] = []
    for p, lv in zip(parts, levels):
        vec.extend([lv] * p)
    return vec


def _staircase_vectors(k: int, total: int, floors: list[int], rules: CreditorRules) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_seg = max(1, min(rules.max_segments, k))
    for s in range(1, max_seg + 1):
        for parts in _compositions(k, s):
            vec = _segment_levels(parts, floors, total)
            if vec is not None and _valid_vector(vec, floors, rules):
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    out.append(vec)
    return out


def _candidate_vectors(k: int, total: int, rules: CreditorRules) -> list[list[int]]:
    floors = _floors(k, rules)
    if sum(floors) > total:
        return []
    if rules.even_pays:
        vec = _even_vector(k, total)
        return [vec] if _valid_vector(vec, floors, rules) else []
    if rules.is_ballooning_allowed:
        vec = _balloon_vector(k, total, floors)
        return [vec] if _valid_vector(vec, floors, rules) else []
    return _staircase_vectors(k, total, floors, rules)


def _running_balances(start: int, events: list[_Event]) -> list[tuple[date, int]]:
    """End-of-day balances, applying all credits before all debits each date."""
    credits: dict[date, int] = defaultdict(int)
    debits: dict[date, int] = defaultdict(int)
    dates: set[date] = set()
    for d, amount, is_credit in events:
        dates.add(d)
        if is_credit:
            credits[d] += amount
        else:
            debits[d] += amount
    bal = start
    out: list[tuple[date, int]] = []
    for d in sorted(dates):
        bal += credits[d]
        bal -= debits[d]
        out.append((d, bal))
    return out


@dataclass
class _Scored:
    cumulative_fee: tuple[int, ...]
    rows: list[ScheduleRow]


def _score_vector(
    vec: list[int],
    start: int,
    base_events: list[_Event],
    fee_total: int,
    rules: CreditorRules,
    cadence: list[date],
) -> _Scored | None:
    """Validate a creditor-payment vector and compute its front-loaded schedule."""
    k = len(vec)

    # B(t): balance trajectory WITHOUT the program fee. Cadence dates are added
    # as zero-credit markers so every cadence date appears in the timeline.
    b_events = list(base_events)
    for i, pay in enumerate(vec):
        b_events.append((cadence[i], pay, False))
        if rules.bank_fee_cents:
            b_events.append((cadence[i], rules.bank_fee_cents, False))
    for d in cadence:
        b_events.append((d, 0, True))

    balances = _running_balances(start, b_events)
    if any(bal < 0 for _, bal in balances):
        return None

    dates = [d for d, _ in balances]
    suffix_min = [0] * len(balances)
    running = None
    for j in range(len(balances) - 1, -1, -1):
        running = balances[j][1] if running is None else min(running, balances[j][1])
        suffix_min[j] = running

    # Earliest-possible cumulative fee per cadence date.
    cumulative: list[int] = []
    for d in cadence:
        j = bisect_left(dates, d)
        cumulative.append(min(fee_total, suffix_min[j]))
    if cumulative[-1] != fee_total:
        return None  # cannot collect the whole fee by the horizon

    per_date_fee = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]

    # Final simulation including the fee, used for the reported running balance.
    f_events = list(base_events)
    for i, pay in enumerate(vec):
        f_events.append((cadence[i], pay, False))
        if rules.bank_fee_cents:
            f_events.append((cadence[i], rules.bank_fee_cents, False))
    for i, d in enumerate(cadence):
        if per_date_fee[i] > 0:
            f_events.append((d, per_date_fee[i], False))
        f_events.append((d, 0, True))

    final_balances = dict(_running_balances(start, f_events))
    if any(bal < 0 for bal in final_balances.values()):
        return None

    rows: list[ScheduleRow] = []
    for i, d in enumerate(cadence):
        creditor = vec[i] if i < k else 0
        fee = per_date_fee[i]
        bank = rules.bank_fee_cents if i < k else 0
        if creditor > 0 or fee > 0:
            rows.append(
                ScheduleRow(
                    date=d,
                    creditor_payment_cents=creditor,
                    program_fee_cents=fee,
                    bank_fee_cents=bank,
                    balance_cents=final_balances[d],
                )
            )
    return _Scored(cumulative_fee=tuple(cumulative), rows=rows)


def _solve(
    start: int,
    base_events: list[_Event],
    total: int,
    fee_total: int,
    rules: CreditorRules,
    cadence: list[date],
) -> tuple[list[ScheduleRow], str] | None:
    """Return the best (front-loaded) schedule for the active shape, or None."""
    max_k = min(rules.max_payments, rules.max_terms, len(cadence))
    if max_k < 1:
        return None
    shape = _shape(rules)

    best_key = None
    best_rows: list[ScheduleRow] | None = None
    for k in range(1, max_k + 1):
        for vec in _candidate_vectors(k, total, rules):
            scored = _score_vector(vec, start, base_events, fee_total, rules, cadence)
            if scored is None:
                continue
            # Maximize earliest cumulative fee; tie-break favours more, smaller
            # creditor payments for a balloon (a "truer" balloon) and fewer
            # payments otherwise.
            tie = k if shape == "balloon" else -k
            key = (scored.cumulative_fee, tie)
            if best_key is None or key > best_key:
                best_key = key
                best_rows = scored.rows
    if best_rows is None:
        return None
    return best_rows, shape


def _future_events(client: Client, lump: int, lump_date: date | None, increment: int) -> list[_Event]:
    """Ledger entries dated after as_of, with optional extra funding applied.

    ``increment`` is added to every future draft (credit); ``lump`` is a single
    extra credit placed on ``lump_date``.
    """
    events: list[_Event] = []
    for entry in client.ledger:
        if entry.date > client.as_of_date:
            is_credit = entry.type == "credit"
            amount = entry.amount_cents + (increment if is_credit else 0)
            events.append((entry.date, amount, is_credit))
    if lump > 0 and lump_date is not None:
        events.append((lump_date, lump, True))
    return events


def _min_extra(feasible_fn, total: int, fee_total: int, rules: CreditorRules) -> int | None:
    """Smallest non-negative extra amount for which ``feasible_fn`` is True."""
    hi = max(1, total + fee_total + rules.bank_fee_cents * max(rules.max_payments, rules.max_terms))
    while not feasible_fn(hi):
        hi *= 2
        if hi > 10**12:
            return None
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible_fn(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _additional_funds(
    client: Client,
    rules: CreditorRules,
    total: int,
    fee_total: int,
    cadence: list[date],
) -> AdditionalFunds:
    start = client.current_balance_cents
    future_credit_dates = [
        e.date for e in client.ledger if e.date > client.as_of_date and e.type == "credit"
    ]
    num_drafts = len(future_credit_dates)
    # An earlier lump is weakly more useful, so place it on the earliest future
    # draft date (clamped to the horizon).
    lump_date = min(future_credit_dates) if future_credit_dates else client.first_draft_date
    if lump_date > client.last_draft_date:
        lump_date = client.last_draft_date

    def feasible_lump(amount: int) -> bool:
        events = _future_events(client, amount, lump_date, 0)
        return _solve(start, events, total, fee_total, rules, cadence) is not None

    def feasible_increment(amount: int) -> bool:
        events = _future_events(client, 0, None, amount)
        return _solve(start, events, total, fee_total, rules, cadence) is not None

    lump = _min_extra(feasible_lump, total, fee_total, rules)
    increment = _min_extra(feasible_increment, total, fee_total, rules)

    lump_cap = _round_half_up(Decimal("0.65") * Decimal(total))
    increment_cap = max(10000, _round_half_up(Decimal("0.40") * Decimal(client.draft_amount_cents)))

    if lump is None:
        lump_opt = FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason="No lump sum makes this offer feasible.",
            date=lump_date,
        )
    else:
        ok = lump <= lump_cap
        lump_opt = FundsOption(
            amount_cents=lump,
            within_guardrail=ok,
            reason="" if ok else f"lump_sum {lump} exceeds guardrail {lump_cap} (round(0.65 x offer_total))",
            date=lump_date,
        )

    if increment is None:
        increment_opt = FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason="No monthly increment makes this offer feasible.",
            num_drafts=num_drafts,
        )
    else:
        ok = increment <= increment_cap
        increment_opt = FundsOption(
            amount_cents=increment,
            within_guardrail=ok,
            reason="" if ok else f"monthly_increment {increment} exceeds guardrail {increment_cap} (max(10000, round(0.40 x draft)))",
            num_drafts=num_drafts,
        )

    return AdditionalFunds(lump_sum=lump_opt, monthly_increment=increment_opt)


def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    """Evaluate a single offer. See ASSIGNMENT.md for the full specification.

    Return a Result with feasible=True and a schedule when the offer fits, or
    feasible=False with additional_funds (minimum lump sum AND minimum monthly
    increment) when it does not.
    """
    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)
    first_payment = offer.first_payment_date or default_first_payment_date(client)
    horizon = client.last_draft_date
    cadence = _cadence_dates(first_payment, horizon)

    start = client.current_balance_cents
    base_events = _future_events(client, 0, None, 0)

    solved = _solve(start, base_events, total, fee_total, rules, cadence)
    if solved is not None:
        rows, shape = solved
        return Result(feasible=True, pay_shape_used=shape, schedule=rows, additional_funds=None)

    funds = _additional_funds(client, rules, total, fee_total, cadence)
    return Result(feasible=False, pay_shape_used=None, schedule=None, additional_funds=funds)
