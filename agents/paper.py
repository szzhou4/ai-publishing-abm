# agents/paper.py — Paper data class

from dataclasses import dataclass, field


@dataclass
class Paper:
    """
    Represents a submitted manuscript.

    Resubmission pipeline (v3): rejected papers enter the submitting scholar's
    resubmission_queue and are re-evaluated each subsequent period at the same
    tier (same quality, new reviewer-noise draw). After MAX_TIER_ATTEMPTS
    successive rejections at one tier, the paper downgrades to the next tier
    (journal_tier += 1, tier_attempts reset to 0). If already at Tier 3 after
    MAX_TIER_ATTEMPTS rejections, the paper is abandoned (removed from queue).

    Attributes
    ----------
    scholar_id : int
    period_produced : int
        Period in which the paper was written and first submitted.
    ai_use_level : float  [0, 1]
        Scholar's AI use level at time of writing. Snapshot; does not change.
    quality : float
        Paper quality from compute_quality(). Fixed at production; same quality
        used for all resubmission evaluations (only reviewer noise varies).
    journal_tier : int {1, 2, 3}
        Current tier being submitted to. Updated when paper downgrades.
    published : bool
        Whether the most recent evaluation resulted in acceptance.
    acceptance_prob : float  [0, 1]
        Analytical acceptance probability from the most recent evaluation.
    original_tier : int {1, 2, 3}
        Tier at first submission. 0 = not yet set (default before assignment).
    tier_attempts : int
        Number of successive rejections at the current journal_tier.
        Incremented on each rejection; reset to 0 when tier downgrades.
        Set to 1 immediately after a new-paper rejection (before queuing).
    """
    scholar_id:      int
    period_produced: int
    ai_use_level:    float
    quality:         float
    journal_tier:    int
    published:       bool  = False
    acceptance_prob: float = 0.0
    original_tier:   int   = 0   # set at production time; 0 = not yet assigned
    tier_attempts:   int   = 0   # rejections at current tier; 0 for new papers
