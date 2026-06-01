"""Extended test suite for evaluate_offer.

These go well beyond the four provided cases: they validate every hard
constraint in ASSIGNMENT.md §5 against the produced schedules, exercise each
payment shape, and check both Part 2 minima (including a guardrail rejection).

Most tests build inputs in-memory rather than from disk so the constraint under
test is obvious and isolated.
"""

from __future__ import annotations

from datetime import date

import pytest

from feasibility.engine import evaluate_offer
from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    add_months,
    default_first_payment_date,
    end_of_month,
    is_end_of_month,
    load_case,
    offer_total_cents,
    program_fee_cents,
    round_half_up,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _drafts(amount, day, start, count):
    """A simple monthly draft ledger (all credits)."""
    entries = []
    d = date(start[0], start[1], day)
    for _ in range(count):
        entries.append(LedgerEntry(d, amount, "credit"))
        d = add_months(d, 1)
    return entries


def _client(*, amount=20000, day=1, start=(2026, 1), count=6, as_of="2025-12-31",
            balance=0, ledger=None):
    ledger = ledger if ledger is not None else _drafts(amount, day, start, count)
    last = max(e.date for e in ledger if e.type == "credit")
    return Client(
        draft_amount_cents=amount,
        draft_day=day,
        first_draft_date=date(start[0], start[1], day),
        last_draft_date=last,
        as_of_date=date.fromisoformat(as_of),
        current_balance_cents=balance,
        ledger=ledger,
    )


def _offer(*, balance=100000, original=100000, pct=0.5, first="2026-01-31"):
    return Offer(
        creditor="Test",
        current_balance_cents=balance,
        original_balance_cents=original,
        settlement_pct=pct,
        first_payment_date=date.fromisoformat(first) if first else None,
    )


def _rules(**kw):
    base = dict(
        max_terms=12, max_payments=12, min_payment_cents=2500, max_token_pays=12,
        min_payment_tiers=[], even_pays=False, is_ballooning_allowed=False,
        max_segments=4, bank_fee_cents=0, program_fee_pct=0.0,
    )
    base.update(kw)
    return CreditorRules(**base)


def _cadence(first, horizon):
    eom = is_end_of_month(first)
    out = []
    for i in range(1200):
        d = add_months(first, i)
        if eom:
            d = end_of_month(d)
        if d > horizon:
            break
        out.append(d)
    return out


def assert_valid_schedule(client, offer, rules, result):
    """Assert a feasible Result obeys every hard constraint in §5."""
    assert result.feasible is True
    assert result.schedule is not None
    rows = result.schedule

    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)
    first = offer.first_payment_date or default_first_payment_date(client)
    horizon = client.last_draft_date
    cadence = _cadence(first, horizon)

    payments = [r.creditor_payment_cents for r in rows if r.creditor_payment_cents > 0]

    # 2. exact sum
    assert sum(payments) == total

    # 1. count & placement: payments on consecutive cadence dates from the first
    pay_dates = [r.date for r in rows if r.creditor_payment_cents > 0]
    assert pay_dates == cadence[: len(pay_dates)]
    assert len(payments) <= min(rules.max_payments, rules.max_terms)

    # horizon: nothing scheduled past it
    assert all(r.date <= horizon for r in rows)

    # 3. non-decreasing
    assert all(payments[i] >= payments[i - 1] for i in range(1, len(payments)))

    # 4. floors (base + token + tiers)
    for i, p in enumerate(payments, start=1):
        floor = rules.min_payment_cents
        for frm, mc in rules.min_payment_tiers:
            if i >= frm:
                floor = max(floor, mc)
        assert p >= floor
    at_base = sum(1 for p in payments if p == rules.min_payment_cents)
    assert at_base <= rules.max_token_pays

    # 5. bank fee on exactly the creditor-payment dates
    for r in rows:
        if r.creditor_payment_cents > 0:
            assert r.bank_fee_cents == rules.bank_fee_cents
        else:
            assert r.bank_fee_cents == 0

    # 6. program fee timing: none before first payment; fully collected; correct total
    assert sum(r.program_fee_cents for r in rows) == fee_total
    assert all(r.date >= first for r in rows if r.program_fee_cents > 0)

    # 9. segments (only when staircase)
    if not rules.even_pays and not rules.is_ballooning_allowed:
        assert len(set(payments)) <= rules.max_segments

    # 10. balance never negative
    assert all(r.balance_cents >= 0 for r in rows)


# --------------------------------------------------------------------------- #
# Round-half-up
# --------------------------------------------------------------------------- #

def test_round_half_up_rounds_away_from_zero():
    assert round_half_up("0.5") == 1
    assert round_half_up("1.5") == 2
    assert round_half_up("2.5") == 3  # not banker's 2
    assert round_half_up("2.4") == 2


def test_offer_total_uses_round_half_up():
    # 0.5 * 12345 = 6172.5 -> 6173
    offer = _offer(balance=12345, pct=0.5)
    assert offer_total_cents(offer) == 6173


# --------------------------------------------------------------------------- #
# Provided cases re-validated against the full constraint checker
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("case", ["case1_feasible_even", "case3_balloon", "case4_tiers"])
def test_provided_feasible_cases_obey_all_constraints(case):
    client, offer, rules = load_case(f"cases/{case}")
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(client, offer, rules, result)


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #

def test_even_distributes_remainder_onto_latest():
    # total 50000 over k payments; verify equal-as-possible with remainder late.
    client = _client(amount=20000, count=6)
    offer = _offer(balance=100000, original=100000, pct=0.5)  # total 50000
    rules = _rules(even_pays=True, max_segments=1, bank_fee_cents=1000,
                   program_fee_pct=0.25)
    result = evaluate_offer(client, offer, rules)
    assert result.pay_shape_used == "even"
    assert_valid_schedule(client, offer, rules, result)
    payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
    # equal as possible: max - min <= 1, and the larger ones are last
    assert max(payments) - min(payments) <= 1
    assert payments == sorted(payments)


def test_balloon_defers_into_a_large_final_payment():
    client = _client(amount=10000, count=13)  # long horizon
    offer = _offer(balance=60000, original=60000, pct=0.5)  # total 30000
    rules = _rules(is_ballooning_allowed=True, max_payments=6, max_terms=6)
    result = evaluate_offer(client, offer, rules)
    assert result.pay_shape_used == "balloon"
    assert_valid_schedule(client, offer, rules, result)
    payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
    assert payments[-1] > payments[0]  # final payment is the balloon
    assert payments[-1] == max(payments)


def test_staircase_respects_max_segments_cap():
    client = _client(amount=10000, count=13)
    offer = _offer(balance=150000, original=150000, pct=0.4, first="2026-01-31")  # total 60000
    rules = _rules(min_payment_tiers=[[7, 5000]], max_token_pays=6, max_segments=2,
                   bank_fee_cents=500, program_fee_pct=0.2)
    result = evaluate_offer(client, offer, rules)
    assert result.pay_shape_used == "staircase"
    assert_valid_schedule(client, offer, rules, result)
    payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
    assert len(set(payments)) <= 2


def test_tier_floor_applies_from_payment_number():
    client = _client(amount=10000, count=13)
    offer = _offer(balance=150000, original=150000, pct=0.4)
    rules = _rules(min_payment_tiers=[[7, 5000]], max_token_pays=6, max_segments=2,
                   bank_fee_cents=500, program_fee_pct=0.2)
    result = evaluate_offer(client, offer, rules)
    payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
    if len(payments) >= 7:
        assert all(p >= 5000 for p in payments[6:])


def test_token_pays_cap_forces_payments_above_base():
    # Only 2 token pays allowed; with >2 payments the rest must exceed the base.
    client = _client(amount=10000, count=13)
    offer = _offer(balance=80000, original=80000, pct=0.5)  # total 40000
    rules = _rules(min_payment_cents=2500, max_token_pays=2, max_segments=4,
                   max_payments=6, max_terms=6)
    result = evaluate_offer(client, offer, rules)
    if result.feasible:
        payments = [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents > 0]
        at_base = sum(1 for p in payments if p == 2500)
        assert at_base <= 2


# --------------------------------------------------------------------------- #
# Simulation invariants
# --------------------------------------------------------------------------- #

def test_credits_before_debits_same_day():
    # Draft and the first creditor payment both land on the same day. With
    # credits-first, a payment equal to the day's credit must keep balance >= 0.
    ledger = [
        LedgerEntry(date(2026, 1, 31), 5000, "credit"),
        LedgerEntry(date(2026, 2, 28), 5000, "credit"),
    ]
    client = Client(
        draft_amount_cents=5000, draft_day=31,
        first_draft_date=date(2026, 1, 31), last_draft_date=date(2026, 2, 28),
        as_of_date=date(2025, 12, 31), current_balance_cents=0, ledger=ledger,
    )
    offer = _offer(balance=20000, original=20000, pct=0.5, first="2026-01-31")  # total 10000
    rules = _rules(min_payment_cents=5000, max_payments=2, max_terms=2, max_segments=2)
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is True
    assert all(r.balance_cents >= 0 for r in result.schedule)


def test_balance_can_hit_exactly_zero():
    client, offer, rules = load_case("cases/case1_feasible_even")
    result = evaluate_offer(client, offer, rules)
    assert any(r.balance_cents == 0 for r in result.schedule)


def test_nothing_scheduled_past_horizon():
    client, offer, rules = load_case("cases/case4_tiers")
    result = evaluate_offer(client, offer, rules)
    assert all(r.date <= client.last_draft_date for r in result.schedule)


# --------------------------------------------------------------------------- #
# Part 2 — minimum additional funds
# --------------------------------------------------------------------------- #

def test_part2_lump_and_increment_match_expected():
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    result = evaluate_offer(client, offer, rules)
    af = result.additional_funds
    assert result.feasible is False
    assert result.schedule is None
    assert af.lump_sum.amount_cents == 10000
    assert af.lump_sum.within_guardrail is True
    assert af.monthly_increment.amount_cents == 2500
    assert af.monthly_increment.num_drafts == 5
    assert af.monthly_increment.within_guardrail is True


def test_part2_minima_are_actually_minimal():
    # One cent less than each reported minimum must remain infeasible.
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    result = evaluate_offer(client, offer, rules)
    af = result.additional_funds

    # Lump minus 1 cent: rebuild a client with that lump and confirm infeasible.
    lump = af.lump_sum.amount_cents
    near = list(client.ledger) + [LedgerEntry(af.lump_sum.date, lump - 1, "credit")]
    client_lump = Client(**{**client.__dict__, "ledger": near})
    assert evaluate_offer(client_lump, offer, rules).feasible is False

    # Increment minus 1 cent on every future draft.
    inc = af.monthly_increment.amount_cents
    bumped = [
        LedgerEntry(e.date, e.amount_cents + (inc - 1), e.type)
        if e.type == "credit" and e.date > client.as_of_date else e
        for e in client.ledger
    ]
    client_inc = Client(**{**client.__dict__, "ledger": bumped})
    assert evaluate_offer(client_inc, offer, rules).feasible is False


def test_part2_lump_guardrail_can_reject():
    # Make the offer require more than 0.65 * offer_total to become feasible.
    # Tiny drafts vs a large settlement => big deficit => lump exceeds guardrail.
    client = _client(amount=1000, count=4, start=(2026, 1))
    offer = _offer(balance=80000, original=80000, pct=0.5, first="2026-01-31")  # total 40000
    rules = _rules(min_payment_cents=2500, max_payments=4, max_terms=4, max_segments=4)
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    af = result.additional_funds
    # 0.65 * 40000 = 26000 guardrail; deficit here is far larger.
    assert af.lump_sum.amount_cents > 26000
    assert af.lump_sum.within_guardrail is False
    assert af.lump_sum.reason != ""
