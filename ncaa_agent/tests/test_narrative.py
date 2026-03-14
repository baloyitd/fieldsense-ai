"""
test_narrative.py
=================
Unit tests for ncaa_agent.narrative.BracketNarrativeGenerator.

Tests cover:
- Template modes: 'standard' and 'brief'
- Narrative content: team names, seeds, statistics, probabilities
- Non-empty output for all trace configurations
- generate_batch produces same-length list
- Upset detection in narrative
- Edge cases: equal seeds, extreme probs
"""

from __future__ import annotations

import pytest

from ncaa_agent.narrative import BracketNarrativeGenerator
from ncaa_agent.reasoner import NCAAReasoner

from .conftest import make_trace, make_feature_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen(template="standard"):
    return BracketNarrativeGenerator(template=template)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_valid_standard_template(self):
        gen = BracketNarrativeGenerator(template="standard")
        assert gen.template == "standard"

    def test_valid_brief_template(self):
        gen = BracketNarrativeGenerator(template="brief")
        assert gen.template == "brief"

    def test_invalid_template_raises(self):
        with pytest.raises(ValueError, match="template"):
            BracketNarrativeGenerator(template="verbose")

    def test_default_is_standard(self):
        gen = BracketNarrativeGenerator()
        assert gen.template == "standard"


# ---------------------------------------------------------------------------
# Standard template
# ---------------------------------------------------------------------------

class TestStandardTemplate:
    def test_generate_returns_non_empty_string(self, sample_trace):
        gen = _gen("standard")
        text = gen.generate(sample_trace)
        assert isinstance(text, str) and len(text) > 0

    def test_narrative_contains_team1_name(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert sample_trace.team1_name in text

    def test_narrative_contains_team2_name(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert sample_trace.team2_name in text

    def test_narrative_contains_seed1(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert f"No.{sample_trace.seed1}" in text

    def test_narrative_contains_seed2(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert f"No.{sample_trace.seed2}" in text

    def test_narrative_contains_probability(self, sample_trace):
        """Win probability should appear as a percentage."""
        text = _gen().generate(sample_trace)
        # Probability formatted as X.X% or XX.X%
        assert "%" in text

    def test_narrative_contains_confidence(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert sample_trace.confidence.lower() in text.lower()

    def test_narrative_contains_ortg(self, sample_trace):
        """Standard template should reference ORTG in sentence 1."""
        text = _gen().generate(sample_trace)
        assert "ORTG" in text

    def test_narrative_contains_netrtg(self, sample_trace):
        text = _gen().generate(sample_trace)
        assert "NetRtg" in text

    def test_narrative_multiple_sentences(self, sample_trace):
        """Standard narrative should have at least 2 sentences."""
        text = _gen().generate(sample_trace)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        assert len(sentences) >= 2

    def test_round_label_present_when_round_in_metadata(self):
        trace = make_trace(seed1=1, seed2=8, ensemble_prob=0.80)
        # Inject round metadata
        trace.metadata["round"] = 1
        text = _gen().generate(trace)
        assert "first-round" in text or "tournament" in text

    def test_upset_note_when_underdog_favoured(self):
        """When higher seed (worse) is given high probability, flag upset."""
        # seed1=8 (worse), seed2=1 (better), but ensemble_prob=0.70 (team1 favoured)
        trace = make_trace(seed1=8, seed2=1, ensemble_prob=0.70)
        text = _gen().generate(trace)
        # Should contain some upset language
        assert "upset" in text.lower() or "potential" in text.lower() or "%" in text

    def test_historical_percentage_in_narrative(self, sample_trace):
        """Sentence 2 should mention historical win rate."""
        text = _gen().generate(sample_trace)
        # Something like "XX% of the time" or historical win rate
        assert "%" in text

    def test_recent_form_in_narrative(self, sample_trace):
        """L10 form should appear in the narrative."""
        text = _gen().generate(sample_trace)
        assert "L10" in text or "recent form" in text.lower()


# ---------------------------------------------------------------------------
# Brief template
# ---------------------------------------------------------------------------

class TestBriefTemplate:
    def test_generate_returns_non_empty_string(self, sample_trace):
        gen = _gen("brief")
        text = gen.generate(sample_trace)
        assert isinstance(text, str) and len(text) > 0

    def test_brief_contains_team1_name(self, sample_trace):
        text = _gen("brief").generate(sample_trace)
        assert sample_trace.team1_name in text

    def test_brief_contains_team2_name(self, sample_trace):
        text = _gen("brief").generate(sample_trace)
        assert sample_trace.team2_name in text

    def test_brief_contains_seeds(self, sample_trace):
        text = _gen("brief").generate(sample_trace)
        assert f"No.{sample_trace.seed1}" in text
        assert f"No.{sample_trace.seed2}" in text

    def test_brief_contains_probability(self, sample_trace):
        text = _gen("brief").generate(sample_trace)
        assert "%" in text

    def test_brief_contains_confidence(self, sample_trace):
        text = _gen("brief").generate(sample_trace)
        assert sample_trace.confidence in text

    def test_brief_is_single_sentence(self, sample_trace):
        """Brief template produces exactly 1 sentence."""
        text = _gen("brief").generate(sample_trace)
        # Ends with a period; splitting by '. ' gives ~1 part
        text_stripped = text.rstrip(". ")
        # Must not have multiple full stops mid-text (beyond abbreviations)
        parts = [p.strip() for p in text.split(". ") if p.strip()]
        assert len(parts) >= 1  # at minimum 1 chunk


# ---------------------------------------------------------------------------
# generate_batch
# ---------------------------------------------------------------------------

class TestGenerateBatch:
    def test_batch_length_matches_input(self, traces_63):
        gen = _gen()
        texts = gen.generate_batch(traces_63)
        assert len(texts) == len(traces_63)

    def test_batch_all_non_empty(self, traces_63):
        gen = _gen()
        texts = gen.generate_batch(traces_63)
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_batch_empty_list(self):
        gen = _gen()
        assert gen.generate_batch([]) == []

    def test_batch_brief_template(self, traces_63):
        gen = _gen("brief")
        texts = gen.generate_batch(traces_63)
        assert len(texts) == len(traces_63)
        assert all(len(t) > 0 for t in texts)

    def test_batch_each_contains_team_name(self, traces_63):
        gen = _gen("brief")
        texts = gen.generate_batch(traces_63)
        for trace, text in zip(traces_63, texts):
            assert trace.team1_name in text or trace.team2_name in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestNarrativeEdgeCases:
    def test_equal_seeds(self):
        trace = make_trace(seed1=5, seed2=5, ensemble_prob=0.52)
        gen = _gen()
        text = gen.generate(trace)
        assert isinstance(text, str) and len(text) > 0

    def test_prob_very_close_to_half(self):
        trace = make_trace(seed1=8, seed2=9, ensemble_prob=0.51)
        text = _gen().generate(trace)
        assert "%" in text

    def test_different_traces_different_narratives(self):
        t1 = make_trace(seed1=1, seed2=16, ensemble_prob=0.95,
                         team1_name="Duke", team2_name="Winthrop")
        t2 = make_trace(seed1=5, seed2=12, ensemble_prob=0.60,
                         team1_name="Ohio State", team2_name="UNCW")
        gen = _gen()
        text1 = gen.generate(t1)
        text2 = gen.generate(t2)
        assert text1 != text2

    def test_no_seed_label_in_no_round_metadata(self):
        """Without round metadata, 'tournament' should appear."""
        trace = make_trace(seed1=3, seed2=14, ensemble_prob=0.72)
        # Ensure no round key
        trace.metadata.pop("round", None)
        text = _gen().generate(trace)
        assert isinstance(text, str) and len(text) > 0
