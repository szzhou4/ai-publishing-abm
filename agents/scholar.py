# agents/scholar.py — Scholar agent (plain Python, no Mesa)

import numpy as np
import config as _cfg   # runtime lookup (not module-level name binding) so that
                        # run_scenarios.py can patch config values and have them
                        # take effect without reloading the module.


class Scholar:
    """
    Represents an academic scholar navigating the tenure track.

    Key design decisions
    --------------------
    * research_capacity captures both individual talent and institutional
      resources (time, funding, lab). Drawn from an institution-specific
      Normal distribution.
    * ai_use_level evolves through pressure-driven RL with asymmetric steps:
      acceptance → positive step scaled by publication pressure;
      rejection  → smaller negative step (scholars don't fully abandon AI
      after a single rejection — social/efficiency pressures maintain some use).
    * Pressure = max(overall_pressure, tier_pressure), where tier_pressure
      reflects institution-specific tier publication expectations (e.g., R1
      needs 3–6 Tier 1 pubs; Balanced needs 2–4 Tier 2 pubs).
    * Tier 1 and Tier 2 publications tracked separately for pressure-based
      journal targeting and tier-specific RL pressure.
    * No loss-aversion weighting; no internal belief state.

    Attributes
    ----------
    scholar_id : int
    research_capacity : float  [0, 1]  -- Fixed
    institution_type : str             -- Fixed
    ai_use_level : float  [0, 1]       -- Dynamic
    papers_per_period : float          -- Dynamic (recomputed each period)
    publication_record : list of dict  -- Dynamic: {period, quality, tier, accepted}
    tier1_publications : int           -- Derived: accepted Tier 1 count
    tier2_publications : int           -- Derived: accepted Tier 2 count
    """

    def __init__(self, scholar_id: int, research_capacity: float,
                 institution_type: str, initial_ai_use: float):
        self.scholar_id          = scholar_id
        self.research_capacity   = research_capacity
        self.institution_type    = institution_type
        self.ai_use_level        = initial_ai_use
        self.publication_record  = []
        self.resubmission_queue  = []   # list of Paper objects awaiting re-evaluation
        self._update_papers_per_period()

    # ── Derived quantity ───────────────────────────────────────────────────────
    def _update_papers_per_period(self):
        """Recompute expected papers per period based on current AI use."""
        base = _cfg.BASE_PRODUCTION_RATE[self.institution_type]
        self.papers_per_period = base * (_cfg.AI_PRODUCTIVITY_MULTIPLIER ** self.ai_use_level)

    # ── RL update ──────────────────────────────────────────────────────────────
    def update_ai_use(self, outcome: str):
        """
        Pressure-driven RL update of ai_use_level (asymmetric steps).

        Called once per paper evaluated (new submission or resubmission), so
        scholars receive multiple RL signals per period proportional to their
        submission volume. All papers in a period are produced and evaluated
        first; RL updates are applied afterwards so within-period quality
        calculations use a consistent ai_use snapshot.

        Acceptance: ai_use += BASE_STEP_POS × (1 + PRESSURE_WEIGHT × pressure)
        Rejection:  ai_use -= BASE_STEP_NEG   (smaller step; scholars don't
                    fully abandon AI after a single rejection — social/efficiency
                    pressures maintain some use).

        Pressure = max(overall_pressure, tier_pressure), where:
          overall_pressure = max(0, 1 − total_pubs / tenure_target_midpoint)
          tier_pressure    = max(0, 1 − tier_pubs  / TIER_PRESSURE_TARGETS mid)

        Tier pressure reflects institution-specific publication expectations:
          R1:       behind on Tier 1 pubs (target 3–6, mid=4)
          R2:       behind on Tier 1 pubs (target 1–3, mid=2)
          Balanced: behind on Tier 2 pubs (target 2–4, mid=3)
          Teaching: no tier expectation (tier_pressure = 0)

        Per-paper break-even (for AI use to grow at max pressure=1.0):
          p_accept > BASE_STEP_NEG / (BASE_STEP_POS*(1+PRESSURE_WEIGHT) + BASE_STEP_NEG)
                   = 0.01 / (0.05×2 + 0.01) ≈ 9%
        With per-paper RL, scholars producing many papers below 9% acceptance
        accumulate net-negative updates → AI use self-corrects before quality
        catastrophically degrades (unlike per-period RL where "any acceptance"
        rule could keep AI growing even at low per-paper acceptance rates).

        All parameters read from config at call time via _cfg so that
        run_scenarios.py can patch them without reloading modules.

        Parameters
        ----------
        outcome : str  --  'accepted' or 'rejected'
        """
        pubs_so_far      = sum(1 for r in self.publication_record if r['accepted'])
        total_target_mid = _cfg.TENURE_TARGET_MIDPOINTS[self.institution_type]
        overall_pressure = max(0.0, 1.0 - pubs_so_far / total_target_mid)

        tier_info = _cfg.TIER_PRESSURE_TARGETS[self.institution_type]
        if tier_info is not None:
            tier_pubs = (self.tier1_publications if tier_info['tier'] == 1
                         else self.tier2_publications)
            tier_pressure = max(0.0, 1.0 - tier_pubs / tier_info['mid'])
        else:
            tier_pressure = 0.0

        pressure = max(overall_pressure, tier_pressure)

        if outcome == 'accepted':
            delta = _cfg.BASE_STEP_POS * (1.0 + _cfg.PRESSURE_WEIGHT * pressure)
            self.ai_use_level = float(np.clip(self.ai_use_level + delta, 0.0, 1.0))
        else:
            self.ai_use_level = float(np.clip(self.ai_use_level - _cfg.BASE_STEP_NEG, 0.0, 1.0))

        self._update_papers_per_period()

    # ── Record outcome ─────────────────────────────────────────────────────────
    def record_outcome(self, period: int, quality: float,
                       tier: int, accepted: bool):
        """Record the outcome of a paper evaluation."""
        self.publication_record.append({
            'period':   period,
            'quality':  quality,
            'tier':     tier,
            'accepted': accepted,
        })

    # ── Derived properties ─────────────────────────────────────────────────────
    @property
    def total_publications(self) -> int:
        return sum(1 for r in self.publication_record if r['accepted'])

    @property
    def tier1_publications(self) -> int:
        """Number of accepted Tier 1 publications. Used for tier targeting and RL pressure."""
        return sum(1 for r in self.publication_record
                   if r['tier'] == 1 and r['accepted'])

    @property
    def tier2_publications(self) -> int:
        """Number of accepted Tier 2 publications. Used for RL pressure (Balanced institutions)."""
        return sum(1 for r in self.publication_record
                   if r['tier'] == 2 and r['accepted'])

    def __repr__(self):
        return (f"Scholar(id={self.scholar_id}, inst={self.institution_type}, "
                f"cap={self.research_capacity:.2f}, ai={self.ai_use_level:.2f}, "
                f"pubs={self.total_publications}, "
                f"t1={self.tier1_publications}, t2={self.tier2_publications})")
