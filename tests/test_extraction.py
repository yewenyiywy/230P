import math

import pandas as pd

from newsalpha.extraction.dictionary import DictionaryExtractor
from newsalpha.extraction.rules import RulesExtractor
from newsalpha.extraction.schema import EventType, finalize

R = RulesExtractor()


def test_earnings_headline():
    x = R.extract_one(
        "Apple Q4 EPS $1.64 Beats $1.60 Estimate, Sales $94.93B Beat $94.36B Estimate"
    )
    assert x.event_type == EventType.EARNINGS
    assert (x.eps_actual, x.eps_expected) == (1.64, 1.60)
    assert (x.revenue_actual, x.revenue_expected) == (94.93, 94.36)


def test_unit_letter_is_not_read_from_next_word():
    x = R.extract_one("XYZ Q3 EPS $0.79 Misses $0.85 Estimate, Sales $1589M vs $1683M Est")
    assert (x.eps_actual, x.eps_expected) == (0.79, 0.85)


def test_analyst_headline():
    x = R.extract_one(
        "UBS Downgrades XYZ to Neutral from Overweight, Lowers Price Target to $258 from $298"
    )
    assert x.event_type == EventType.ANALYST
    assert (x.rating_new, x.rating_old, x.target_new, x.target_old) == (
        "Neutral",
        "Overweight",
        258,
        298,
    )


def test_guidance_headline():
    x = R.extract_one("XYZ Raises FY26 Revenue Guidance to $10.00B-$11.00B from $9.00B-$10.00B")
    assert x.event_type == EventType.GUIDANCE
    assert math.isclose(x.guidance_new_mid, 10.5) and math.isclose(x.guidance_old_mid, 9.5)


def test_finalize_computes_surprises_in_code():
    rows = [
        {"article_id": "e", **R.extract_one("X Q1 EPS $1.10 Beats $1.00 Estimate").model_dump(mode="json")},
        {"article_id": "a", **R.extract_one("Citi Upgrades X to Buy from Neutral, Raises Price Target to $120 from $100").model_dump(mode="json")},
        {"article_id": "o", **R.extract_one("X Shares Are Trading Lower: What's Going On?").model_dump(mode="json")},
    ]  # fmt: skip
    f = finalize(pd.DataFrame(rows), "rules:v1").set_index("article_id")
    assert math.isclose(f.loc["e", "eps_surprise"], 0.10, rel_tol=1e-9)
    assert f.loc["e", "direction"] == 1
    assert f.loc["a", "rating_change"] == 2 and math.isclose(f.loc["a", "target_change"], 0.2)
    assert math.isnan(f.loc["o", "eps_surprise"]) and f.loc["o", "event_type"] == "other"


def test_dictionary_tone():
    d = DictionaryExtractor()
    assert d.extract_one("Company beats estimates, raises outlook").direction == 1
    assert d.extract_one("Company misses, cuts guidance amid probe").direction == -1


def test_extraction_cache_has_one_row_per_article(extractions, articles):
    assert extractions["article_id"].is_unique
    assert set(extractions["article_id"]) == set(articles["article_id"])
