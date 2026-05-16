# config.py — Master parameters for AI Publishing ABM
# Edit this file to change model behavior. All parameters are imported into simulation.ipynb.

import numpy as np

# ── Simulation scale ───────────────────────────────────────────────────────────
N_SCHOLARS     = 200
N_PERIODS      = 12       # 12 periods = 6-year tenure clock
N_REPLICATIONS = 50
RANDOM_SEED    = 42

# ── Publication lag ────────────────────────────────────────────────────────────
# PUBLICATION_LAG removed (v2): papers are produced and evaluated within the same
# period. Future versions should re-introduce a lag and model paper resubmission
# (rejected papers re-queued to lower tiers or same tier after a delay).

# ── Institution type distribution (static, 25% each) ──────────────────────────
INSTITUTION_TYPES        = ['R1', 'R2', 'Balanced', 'Teaching']
INSTITUTION_DISTRIBUTION = [0.25, 0.25, 0.25, 0.25]   # must sum to 1.0

# research_capacity drawn from Normal(mean, sd), clipped to [0.05, 0.99].
# Calibrated so that at zero AI use (quality = research_capacity), scholars
# produce papers in realistic quality ranges:
#   R1:       average quality ~0.80  (range roughly 0.70–0.90)
#   R2:       average quality ~0.70  (range roughly 0.50–0.90)
#   Balanced: average quality ~0.60  (range roughly 0.45–0.75)
#   Teaching: average quality ~0.50  (range roughly 0.35–0.65)
RESEARCH_CAPACITY_PARAMS = {
    'R1':       {'mean': 0.80, 'sd': 0.05},   # v8: mean reduced from 0.85; SD back to 0.05
    'R2':       {'mean': 0.70, 'sd': 0.10},
    'Balanced': {'mean': 0.60, 'sd': 0.15},
    'Teaching': {'mean': 0.50, 'sd': 0.15},
}

# ── Base paper production rates (papers per period, no AI) ─────────────────────
# Each period = 1 semester (6 months). Rates reflect new-paper submissions per
# semester. A "decent" mid-career R-scholar targets ~3 submissions/year (1.5/period).
# Note: real scholars also resubmit rejected papers, which inflates submission
# counts beyond new production. Future versions should implement a resubmission
# queue so that rejected papers can be re-directed to lower-tier journals.
BASE_PRODUCTION_RATE = {
    'R1':       2.000,   # ~4 new submissions/year
    'R2':       1.500,   # ~3 new submissions/year
    'Balanced': 1.000,   # ~2 new submissions/year
    'Teaching': 0.500,   # ~1 new submission/year
}

# ── Tenure targets ─────────────────────────────────────────────────────────────
# Fixed publication counts required for tenure (v8: changed from ranges to fixed targets).
# Used in RL pressure formula and tenure-attainment figures.
TENURE_TARGET_MIDPOINTS = {
    'R1':       10,
    'R2':        8,
    'Balanced':  5,
    'Teaching':  2,
}

# ── Initial AI use distribution (Nag, Leung, Zhou & Belwalkar, 2025) ───────────
# Survey of 62 tenure-track I-O faculty; values on [0, 1] scale
NAG_AI_VALUES      = [0.00,  0.05,  0.15,  0.30,  0.50,  0.70]
NAG_AI_PROPORTIONS = [0.145, 0.177, 0.048, 0.177, 0.258, 0.194]   # must sum to 1.0

# ── AI productivity multiplier ─────────────────────────────────────────────────
# papers_per_period = base_rate × AI_PRODUCTIVITY_MULTIPLIER ^ ai_use_level
# Calibrated to peer-reviewed estimates:
#   Noy & Zhang (2023, Science): 1.7× speedup for writing tasks (ChatGPT)
#   Dell'Acqua et al. (2023, BCG): 1.4× for analytic/writing tasks
#   Brynjolfsson et al. (2023): 1.3–2.0× for knowledge work
# Consensus range: 1.5–2.0×. Set to 2.0 (upper bound, conservative upper estimate).
# At ai_use=0: rate = base_rate × 1.0 (no change)
# At ai_use=0.5: rate = base_rate × 1.41 (41% boost)
# At ai_use=1.0: rate = base_rate × 2.0 (2× boost)
AI_PRODUCTIVITY_MULTIPLIER = 2.0   # range: 1.5–2.0 (literature-anchored)

# ── Reinforcement learning (pressure-driven, asymmetric steps) ────────────────
# Update rule per period:
#   accepted: ai_use += BASE_STEP_POS × (1 + PRESSURE_WEIGHT × pressure)
#   rejected: ai_use -= BASE_STEP_NEG
#
# Asymmetric steps reflect that scholars don't abandon AI the moment a paper is
# rejected — they have many other reasons (efficiency, student expectations,
# department norms) to maintain some AI use. The smaller negative step means
# rejection produces only a mild downward correction.
#
# Break-even condition (for AI use to grow on average at max pressure=1.0):
#   p_accept × BASE_STEP_POS × (1 + PRESSURE_WEIGHT) > (1-p_accept) × BASE_STEP_NEG
#   → p_accept > BASE_STEP_NEG / (BASE_STEP_POS*(1+PRESSURE_WEIGHT) + BASE_STEP_NEG)
#   With current values: p_accept > 0.01 / (0.10 + 0.01) ≈ 9%
#   R1 scholars reach ~9% per-period acceptance as quality improves → growth possible.
BASE_STEP_POS   = 0.05   # positive step for acceptance
BASE_STEP_NEG   = 0.01   # negative step for rejection (smaller = slower decline)
PRESSURE_WEIGHT = 1.0    # amplification of positive step by publication pressure

# ── Tier 3 submission / cascade policy ────────────────────────────────────────
# R1 scholars have strong institutional norms against Tier 3 publications.
# They will only submit to (or cascade to) Tier 3 when they are "desperate" —
# i.e., total_publications < T3_FLOOR_FRACTION × TENURE_TARGET_MIDPOINTS[inst].
# Applies to BOTH new paper tier assignment and resubmission tier downgrade.
#   R1 threshold: 0.40 × 10 = 4 pubs  (< 4 total pubs = desperate)
# Set to None for institutions with no Tier 3 avoidance norm.
T3_FLOOR_FRACTION = {
    'R1':       0.40,   # only allow T3 if total_pubs < 4 (40% of target midpoint 10)
    'R2':       None,   # no T3 avoidance norm
    'Balanced': None,   # no T3 avoidance norm
    'Teaching': None,   # always submits to T3 anyway
}

# ── Resubmission pipeline ──────────────────────────────────────────────────────
# Rejected papers enter a resubmission queue and are re-evaluated the next period
# at the same tier (same quality, new reviewer-noise draw). After MAX_TIER_ATTEMPTS
# successive rejections at one tier, the paper downgrades to the next tier; if
# already at Tier 3 it is abandoned. Resubmissions do NOT count toward
# papers_per_period but DO count toward publications and the RL signal when accepted.
MAX_TIER_ATTEMPTS = 3    # rejections at one tier before downgrading

# ── Quality function parameters (piecewise; Gartenberg et al., 2026 calibration) ─
AI_LOW_THRESHOLD  = 0.30   # below: linear penalty; above: nonlinear
AI_LOW_PENALTY    = 0.05   # slope for low AI use
AI_HIGH_PENALTY   = 2.50   # coefficient for nonlinear high-AI penalty
AI_HIGH_EXPONENT  = 1.50   # exponent for nonlinear penalty (>1 = accelerating)

# ── Journal acceptance rates ───────────────────────────────────────────────────
# base_accept_prob is the UNCONDITIONAL acceptance rate across ALL papers
# submitted to that tier — i.e., the population-level rate you would observe
# if you sampled a random submission. The logistic acceptance function is anchored
# at NORM_QUALITY (0.65) so that a paper of that quality receives exactly
# base_accept_prob. Papers above NORM_QUALITY receive higher probabilities;
# papers below receive lower ones.
#
# With QUALITY_SLOPE_T1 = 20, the function rises sharply:
#   quality 0.65 → 5%  (base, by construction)
#   quality 0.80 → ~51%
#   quality 0.90 → ~89%
#   quality 0.95 → ~96%  (near 100%, as intended)
JOURNAL_TIERS = {
    1: {'base_accept_prob': 0.03, 'name': 'Top Tier'},    # v8: tightened from 5%
    2: {'base_accept_prob': 0.15, 'name': 'Mid Tier'},
    3: {'base_accept_prob': 0.30, 'name': 'Lower Tier'},
}

# ── Institution-specific & pressure-based tier-targeting thresholds ────────────
# Scholars' journal tier choice depends on institution type AND whether they
# have met their Tier 1 publication target (see TIER1_PUB_TARGETS below).
#
# R1 scholars who are behind their Tier 1 target always submit to Tier 1
# (regardless of quality) — tenure pressure overrides quality considerations.
# Once the Tier 1 target is met, they switch to quality-based targeting.
# All other institutions use quality-based targeting throughout.
#
# Format: {'behind': (tier1_min_quality, tier2_min_quality),
#          'ahead':  (tier1_min_quality, tier2_min_quality)}
# 'behind' = scholar has not yet met their TIER1_PUB_TARGET
# 'ahead'  = scholar has met or exceeded their TIER1_PUB_TARGET
#
# v8: All institutions now use the same quality thresholds (0.82, 0.55) for
# tier targeting. Institution-specific behaviour emerges from differences in
# research capacity rather than hard-coded tier exclusions. Teaching scholars
# can now reach Tier 2 (quality ≥ 0.55) and Balanced scholars can reach Tier 1
# (quality ≥ 0.82), though both are rare given their capacity distributions.
# Only R1's 'behind' case retains a special rule (always attempt Tier 1 under
# tenure pressure).
_T1_THRESH = 0.82   # common Tier 1 quality threshold across all institutions
_T2_THRESH = 0.55   # common Tier 2 quality threshold across all institutions

TIER_THRESHOLDS = {
    'R1': {
        'behind': (0.00, 0.00),          # tenure pressure: always attempt Tier 1
        'ahead':  (_T1_THRESH, _T2_THRESH),
    },
    'R2': {
        'behind': (_T1_THRESH, _T2_THRESH),
        'ahead':  (_T1_THRESH, _T2_THRESH),
    },
    'Balanced': {
        'behind': (_T1_THRESH, _T2_THRESH),
        'ahead':  (_T1_THRESH, _T2_THRESH),
    },
    'Teaching': {
        'behind': (_T1_THRESH, _T2_THRESH),
        'ahead':  (_T1_THRESH, _T2_THRESH),
    },
}

# Number of Tier 1 publications each institution type needs for tenure.
# R1 scholars target Tier 1 until this count is reached; then switch to
# quality-based targeting. Set to 0 for institutions with no Tier 1 requirement.
TIER1_PUB_TARGETS = {
    'R1':       3,   # v8: fixed T1 requirement (was 4); switch to quality-based targeting once met
    'R2':       1,   # v8: fixed T1 requirement (was 2)
    'Balanced': 0,
    'Teaching': 0,
}

# ── Tier-specific publication pressure targets (for RL update) ────────────────
# These define the EXPECTED tier-level publication counts by institution type.
# Used in the RL pressure formula to measure shortfall against tier goals.
# Format: {'tier': int, 'mid': float}  or None if no tier expectation.
#   R1:       3–6 Tier 1 pubs required; midpoint = 4.5 → use 4 (integer target)
#   R2:       1–3 Tier 1 pubs required; midpoint = 2
#   Balanced: 2–4 Tier 2 pubs required; midpoint = 3
#   Teaching: no tier-level expectation
# Note: R1 also expects NO Tier 3 publications (enforced primarily via tier
# targeting; R1 scholars with mean quality 0.85 rarely reach T3 thresholds).
TIER_PRESSURE_TARGETS = {
    'R1':       {'tier': 1, 'mid': 3},   # v8: 3 T1 pubs required (was 4)
    'R2':       {'tier': 1, 'mid': 1},   # v8: 1 T1 pub required (was 2)
    'Balanced': {'tier': 2, 'mid': 2},   # v8: 2 T2 pubs required (was 3)
    'Teaching': None,
}

# ── Acceptance model: percentile-threshold (replaces logistic) ────────────────
# A journal with X% acceptance rate accepts the top X% of papers by quality.
# Quality is assumed ~ Normal(QUALITY_DIST_MEAN, QUALITY_DIST_SD) across the
# full submission pool. The acceptance threshold for each tier is the
# (1 − base_accept_prob) quantile of this distribution.
#
# With QUALITY_DIST_MEAN=0.50, QUALITY_DIST_SD=0.20:
#   Tier 1 (top  3%): threshold = 0.50 + z_{0.97} × 0.20 ≈ 0.876  (v8: was 0.829 at 5%)
#   Tier 2 (top 15%): threshold = 0.50 + z_{0.85} × 0.20 ≈ 0.707
#   Tier 3 (top 30%): threshold = 0.50 + z_{0.70} × 0.20 ≈ 0.605
#
# At acceptance, Gaussian noise (QUALITY_NOISE_SD = 0.10) is added to the paper's
# quality before comparing to the threshold, capturing reviewer randomness and
# variation in paper presentation. The noisy quality is clipped to [0.01, 0.99].
# Accepted iff noisy_quality >= threshold.
#
# The analytical acceptance probability (stored on the Paper object) is:
#   P(accept | quality) = Φ((quality − threshold) / QUALITY_NOISE_SD)
# where Φ is the standard normal CDF.
from scipy.stats import norm as _norm
QUALITY_DIST_MEAN = 0.50    # assumed mean of quality distribution across all submissions
QUALITY_DIST_SD   = 0.20    # assumed SD of quality distribution
QUALITY_NOISE_SD  = 0.10    # Gaussian noise SD added to quality at review (reviewer randomness)

ACCEPTANCE_THRESHOLDS = {
    tier: float(_norm.ppf(1.0 - JOURNAL_TIERS[tier]['base_accept_prob'],
                          QUALITY_DIST_MEAN, QUALITY_DIST_SD))
    for tier in JOURNAL_TIERS
}
# Tier 1: ~0.829 | Tier 2: ~0.707 | Tier 3: ~0.605

# ── Visualization color palette ────────────────────────────────────────────────
COLORS = {
    'primary':   '#981A31',
    'secondary': '#00546B',
    'tertiary':  '#5E2154',
    'gold':      '#C4A35A',
    'green':     '#4A7C59',
    'salmon':    '#D4856A',
}
INST_COLORS = {
    'R1':       '#981A31',
    'R2':       '#00546B',
    'Balanced': '#4A7C59',
    'Teaching': '#C4A35A',
}
