"""
ncaa_agent.narrative
====================
Stage 07 — Human-readable bracket narrative generator.

BracketNarrativeGenerator converts a :class:`~ncaa_agent.reasoner.ReasoningTrace`
into 2–3 natural-language sentences suitable for blog posts or bracket
analysis content.

Each narrative:
- Names both teams by name (not just ID).
- States both seeds.
- References 2+ key statistics (ORTG/DRTG, net rating, recent form).
- States the predicted win probability and confidence level.
- Flags upset potential when applicable.

Usage
-----
::

    gen = BracketNarrativeGenerator()
    text = gen.generate(trace)
    # "No. 1 Kansas (ORTG 118.0, NetRtg +20.0) faces No. 8 Howard in a …"
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Ordinal suffixes for seed numbers
_ORDINAL = {1: "st", 2: "nd", 3: "rd"}


def _seed_str(seed: int) -> str:
    """Return e.g. '1st', '2nd', '3rd', '4th'."""
    suffix = _ORDINAL.get(seed if seed <= 3 else 0, "th")
    return f"No.{seed} ({seed}{suffix}-seeded)"


def _pct(p: float) -> str:
    return f"{p:.1%}"


def _round_label(rnd: int | None) -> str:
    labels = {
        1: "first-round", 2: "second-round", 3: "Sweet 16",
        4: "Elite Eight", 5: "Final Four", 6: "national championship",
    }
    return labels.get(rnd, "tournament") if rnd else "tournament"


class BracketNarrativeGenerator:
    """
    Generate 2–3 sentence narratives for tournament matchups.

    The narratives are produced deterministically from the reasoning trace
    data — no external calls are made.

    Parameters
    ----------
    template : {'standard', 'brief'}, optional
        'standard' (default) emits 2–3 varied sentences;
        'brief' emits a single sentence summary.
    """

    def __init__(self, template: str = "standard") -> None:
        if template not in ("standard", "brief"):
            raise ValueError(f"Unknown template '{template}'. Choose 'standard' or 'brief'.")
        self.template = template

    def generate(self, trace) -> str:
        """
        Generate a narrative for a single matchup.

        Parameters
        ----------
        trace : ReasoningTrace

        Returns
        -------
        str — non-empty narrative text
        """
        if self.template == "brief":
            return self._brief(trace)
        return self._standard(trace)

    def generate_batch(self, traces) -> List[str]:
        """
        Generate narratives for all traces in the list.

        Returns
        -------
        list of str, same length as ``traces``
        """
        return [self.generate(t) for t in traces]

    # ------------------------------------------------------------------
    # Template implementations
    # ------------------------------------------------------------------

    def _standard(self, trace) -> str:
        """2–3 sentence narrative with team names, seeds, and key stats."""
        t1 = trace.team1_name
        t2 = trace.team2_name
        s1 = trace.seed1
        s2 = trace.seed2
        p  = trace.adjusted_prob
        conf = trace.confidence.lower()
        rnd = trace.metadata.get("round")

        # Pull metrics from step key_metrics
        eff  = _get_step(trace, 1)
        hist = _get_step(trace, 2)
        ctx  = _get_step(trace, 3)

        net1 = eff.get("net_rtg1", 0.0) if eff else 0.0
        net2 = eff.get("net_rtg2", 0.0) if eff else 0.0
        ortg1 = eff.get("ortg1", 100.0) if eff else 100.0
        ortg2 = eff.get("ortg2", 100.0) if eff else 100.0
        l10_1 = ctx.get("win_pct_last10_1", 0.5) if ctx else 0.5
        l10_2 = ctx.get("win_pct_last10_2", 0.5) if ctx else 0.5
        hist_rate = hist.get("historical_win_rate", 0.5) if hist else 0.5

        # --- Sentence 1: Introduction with seeds and efficiency ---
        net_leader = t1 if net1 > net2 else t2
        net_gap = abs(net1 - net2)

        sentence1 = (
            f"In this {_round_label(rnd)} matchup, "
            f"No.{s1} {t1} (ORTG {ortg1:.0f}, NetRtg {net1:+.0f}) "
            f"faces No.{s2} {t2} (ORTG {ortg2:.0f}, NetRtg {net2:+.0f}); "
            f"{net_leader} holds a net-rating edge of {net_gap:.0f} points."
        )

        # --- Sentence 2: Historical context and recent form ---
        fav_name = t1 if s1 < s2 else t2
        fav_hist = hist_rate if s1 < s2 else 1.0 - hist_rate

        form_t1 = "strong recent form" if l10_1 >= 0.70 else "shaky recent form" if l10_1 <= 0.40 else "steady recent form"
        form_t2 = "strong recent form" if l10_2 >= 0.70 else "shaky recent form" if l10_2 <= 0.40 else "steady recent form"

        sentence2 = (
            f"Historically, the No.{min(s1,s2)} seed wins this matchup "
            f"{fav_hist:.0%} of the time — "
            f"{t1} brings {form_t1} (L10: {_pct(l10_1)}), "
            f"while {t2} shows {form_t2} (L10: {_pct(l10_2)})."
        )

        # --- Sentence 3: Prediction ---
        winner = t1 if p > 0.5 else t2
        win_prob = p if p > 0.5 else 1.0 - p

        upset_note = ""
        if min(s1, s2) == s2 and p > 0.5:
            # team1 is higher seed but still favourite
            upset_note = " (upset alert)"
        elif min(s1, s2) == s1 and p < 0.5:
            upset_note = " (potential upset)"

        sentence3 = (
            f"The model gives {t1} a {_pct(p)} win probability "
            f"with {conf} confidence, "
            f"pointing to {winner} as the {_round_label(rnd)} winner{upset_note}."
        )

        return f"{sentence1} {sentence2} {sentence3}"

    def _brief(self, trace) -> str:
        """Single-sentence summary."""
        t1 = trace.team1_name
        t2 = trace.team2_name
        s1 = trace.seed1
        s2 = trace.seed2
        p  = trace.adjusted_prob
        return (
            f"No.{s1} {t1} vs No.{s2} {t2}: "
            f"model gives {t1} a {_pct(p)} win probability "
            f"({trace.confidence} confidence)."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_step(trace, step_num: int):
    """Return the key_metrics dict for the given step number, or {}."""
    for s in trace.steps:
        if s.step == step_num:
            return s.key_metrics
    return {}
