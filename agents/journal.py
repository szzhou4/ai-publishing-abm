# agents/journal.py — Journal agent (plain Python, no Mesa)
# Reviewer pool modeling is not included in this version;
# future versions will add reviewer pool degradation as submission
# volume increases post-AI adoption (Gartenberg et al., 2026).

from functions.quality import compute_acceptance_probability
import numpy as np


class Journal:
    """
    Represents an academic journal.

    Acceptance probability is a function of paper quality and tier only.
    Gaussian review-process noise (REVIEW_NOISE) is added before the
    Bernoulli acceptance draw, capturing reviewer-level randomness.

    Attributes
    ----------
    journal_id : int
    tier : int  {1, 2, 3}
        1 = Top Tier (2% base),  2 = Mid Tier (10%),  3 = Lower Tier (30%).
    """

    def __init__(self, journal_id: int, tier: int):
        self.journal_id = journal_id
        self.tier       = tier

    def evaluate(self, paper, rng: np.random.Generator) -> bool:
        """
        Evaluate a paper and return True if accepted.

        Sets paper.acceptance_prob and paper.published in place.
        """
        prob, accepted      = compute_acceptance_probability(paper.quality, self.tier, rng)
        paper.acceptance_prob = prob
        paper.published       = accepted
        return accepted

    def __repr__(self):
        return f"Journal(id={self.journal_id}, tier={self.tier})"
