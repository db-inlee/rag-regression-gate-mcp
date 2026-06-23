"""F3 — the noise-band floor uses the RUNTIME population size (domain-agnostic).

Pins the denom fix (audit finding): the ±1-case floor must be measured in the
candidate domain's actual cases, not DART's hardcoded 85/15/100. These tests fail
if anyone reintroduces a fixed denominator.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.config import RagConfig
from app.regression.detect import (_bootstrap, _effective_band_cases,
                                   _population_size)


def _ans_case(grounded: bool) -> dict:
    """A full answerable feats record (all keys the 8 metrics read)."""
    return {"answerable": True, "grounded_correct": grounded, "strict_ok": True,
            "value_ok": grounded, "na_correct": False, "over_answer": False,
            "mode": "correct" if grounded else "hallucination"}


def _feats(n_answerable: int, n_no_answer: int, grounded_first: int | None = None) -> dict:
    """Build synthetic feats. grounded_first: how many answerable cases are grounded
    (default = all). no_answer cases are all correct (na_correct)."""
    g = n_answerable if grounded_first is None else grounded_first
    feats = {f"a{i}": _ans_case(i < g) for i in range(n_answerable)}
    for j in range(n_no_answer):
        feats[f"n{j}"] = {"answerable": False, "grounded_correct": False, "strict_ok": False,
                          "value_ok": False, "na_correct": True, "over_answer": False, "mode": "correct"}
    return feats


# --- _population_size: actual counts, not 85/15/100 ------------------------- #

def test_population_size_counts_runtime():
    f = _feats(40, 0)
    ids = list(f)
    assert _population_size("answerable", ids, f) == 40   # NOT 85
    assert _population_size("no_answer", ids, f) == 0      # NOT 15
    assert _population_size("all", ids, f) == 40           # NOT 100

    f2 = _feats(85, 15)
    ids2 = list(f2)
    assert _population_size("answerable", ids2, f2) == 85  # DART happens to be 85
    assert _population_size("no_answer", ids2, f2) == 15
    assert _population_size("all", ids2, f2) == 100


# --- _effective_band_cases: scales by the PASSED denom, not a constant ------ #

def test_effective_band_uses_passed_denom():
    band = {"metric_band": {"answerable_accuracy": {"std": 0.05}}}
    eff_40 = _effective_band_cases("answerable_accuracy", band, 40)   # max(0.05*40,1)=2.0
    eff_100 = _effective_band_cases("answerable_accuracy", band, 100)  # max(0.05*100,1)=5.0
    assert eff_40 == pytest.approx(2.0)
    assert eff_100 == pytest.approx(5.0)
    assert eff_40 != eff_100  # proves the denom argument is honored (not hardcoded 85)
    # std=0 → floor of 1.0 case regardless of domain
    assert _effective_band_cases("answerable_accuracy", {"metric_band": {}}, 40) == 1.0


# --- end-to-end: a 1-case change in a 40-case domain sits AT the floor ------ #

def _ans_row(case_ids, B, C):
    band = {"metric_band": {}}  # all std=0 → eff_band floored at 1.0 case
    rows = _bootstrap(case_ids, B, C, band, RagConfig())
    return next(r for r in rows if r["metric"] == "answerable_accuracy")


def test_one_case_change_is_floor_boundary():
    """40-case domain, exactly 1 answerable case flipped → delta_cases == 1.0 == floor.
    The '1 case' is 1/40 (runtime), NOT 1/85; if denom were still 85, delta_cases
    would be 0.025*85 = 2.125."""
    B = _feats(40, 0)                      # all grounded
    C = _feats(40, 0, grounded_first=39)   # 1 case flipped
    ids = list(B)
    row = _ans_row(ids, B, C)
    assert row["delta"] == pytest.approx(-0.025)
    assert row["delta_cases"] == pytest.approx(1.0)            # = 0.025*40, NOT 2.125 (0.025*85)
    assert row["effective_band_cases"] == pytest.approx(1.0)   # floor
    assert row["direction"] != "regression"                   # at boundary, not beyond → not significant


def test_multi_case_change_crosses_floor():
    """Same 40-case domain, 8 cases flipped → clearly beyond the 1-case floor → regression."""
    B = _feats(40, 0)
    C = _feats(40, 0, grounded_first=32)   # 8 flipped
    ids = list(B)
    row = _ans_row(ids, B, C)
    assert row["delta_cases"] == pytest.approx(8.0)            # 0.2*40, runtime denom
    assert row["direction"] == "regression"
