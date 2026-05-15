# functions/quality.py — Core quality and acceptance functions

import numpy as np


def compute_quality(research_capacity: float, ai_use_level: float) -> float:
    """
    Compute paper quality from research capacity and AI use level.

    Piecewise penalty function calibrated to Gartenberg et al. (2026,
    Organization Science): ~1.28 SD quality decline at high AI use.

    Parameters
    ----------
    research_capacity : float  [0, 1]
        Scholar's effective research capacity (talent + institutional resources).
    ai_use_level : float  [0, 1]
        Proportion of work performed via AI tools.

    Returns
    -------
    float
        Paper quality score. May be negative at very high AI use.
    """
    from config import (AI_LOW_THRESHOLD, AI_LOW_PENALTY,
                        AI_HIGH_PENALTY, AI_HIGH_EXPONENT)

    if ai_use_level < AI_LOW_THRESHOLD:
        penalty = AI_LOW_PENALTY * ai_use_level
    else:
        penalty = (AI_LOW_PENALTY * AI_LOW_THRESHOLD
                   + AI_HIGH_PENALTY * (ai_use_level - AI_LOW_THRESHOLD) ** AI_HIGH_EXPONENT)

    return float(research_capacity - penalty)


def assign_journal_tier(quality: float, institution_type: str,
                        tier1_pubs: int = 0) -> int:
    """
    Assign a journal tier based on paper quality, institution type, and
    whether the scholar has met their Tier 1 publication target.

    Tier-targeting logic
    --------------------
    R1 scholars who have NOT yet met their TIER1_PUB_TARGET always submit to
    Tier 1 (tenure pressure overrides quality considerations — any chance of
    a top-tier acceptance is worth taking). Once the Tier 1 target is met,
    they switch to quality-based targeting. All other institutions use
    quality-based targeting throughout.

    Parameters
    ----------
    quality : float
        Paper quality from compute_quality(). May be negative.
    institution_type : str
        One of 'R1', 'R2', 'Balanced', 'Teaching'.
    tier1_pubs : int
        Number of Tier 1 publications the scholar has accumulated so far.
        Used to determine whether the scholar is 'behind' or 'ahead' on
        their Tier 1 target. Defaults to 0.

    Returns
    -------
    int : 1, 2, or 3
    """
    from config import TIER_THRESHOLDS, TIER1_PUB_TARGETS

    target  = TIER1_PUB_TARGETS[institution_type]
    behind  = tier1_pubs < target
    key     = 'behind' if behind else 'ahead'
    t1_min, t2_min = TIER_THRESHOLDS[institution_type][key]

    if quality >= t1_min:
        return 1
    elif quality >= t2_min:
        return 2
    else:
        return 3


def compute_acceptance_probability(quality: float,
                                   journal_tier: int,
                                   rng: np.random.Generator) -> tuple:
    """
    Determine acceptance via a percentile-threshold model.

    Model
    -----
    A journal with X% acceptance rate accepts the top X% of papers by quality.
    Thresholds are derived from the population quality distribution
    Normal(QUALITY_DIST_MEAN=0.50, QUALITY_DIST_SD=0.20):

      Tier 1 (5%):  threshold ≈ 0.829
      Tier 2 (15%): threshold ≈ 0.707
      Tier 3 (30%): threshold ≈ 0.605

    Gaussian noise (QUALITY_NOISE_SD=0.10) is added to the paper's quality
    before comparing to the threshold, capturing reviewer randomness.
    The noisy quality is clipped to [0.01, 0.99].
    Accepted iff noisy_quality >= threshold.

    The analytical acceptance probability (stored on the Paper object) is:
      P(accept | quality) = Φ((quality − threshold) / QUALITY_NOISE_SD)

    Note: future versions should add paper resubmission logic so that rejected
    papers can be re-directed to lower-tier journals after a time lag.

    Parameters
    ----------
    quality : float
        Paper quality (already clipped to [0.01, 0.99]).
    journal_tier : int
        1, 2, or 3.
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    (acceptance_prob, accepted) : (float, bool)
    """
    from config import ACCEPTANCE_THRESHOLDS, QUALITY_NOISE_SD
    from scipy.special import ndtr  # standard normal CDF, numerically stable

    threshold   = ACCEPTANCE_THRESHOLDS[journal_tier]
    noisy_q     = float(np.clip(quality + rng.normal(0.0, QUALITY_NOISE_SD), 0.01, 0.99))
    accepted    = noisy_q >= threshold
    # Analytical ex-ante probability given the paper's true quality
    accept_prob = float(np.clip(ndtr((quality - threshold) / QUALITY_NOISE_SD), 0.0, 1.0))
    return accept_prob, accepted
