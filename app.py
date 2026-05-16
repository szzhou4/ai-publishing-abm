"""
app.py — Streamlit webapp for the AI Publishing ABM.

Deployment: Streamlit Community Cloud (share.streamlit.io)
  1. Push the full abm_project/ directory to a public GitHub repo.
  2. On share.streamlit.io: New app → select repo → Main file path: app.py
  3. Python version: 3.11+.  requirements.txt must be in the same directory.
  4. No secrets needed; all computation is in-memory.

Usage (local):
    streamlit run app.py
"""

import sys
import os
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import streamlit as st

# ── Path setup so all relative imports resolve correctly ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)

# ── Lazy module import (after path setup) ────────────────────────────────────
import config

def apply_overrides(overrides: dict) -> dict:
    """Patch config module with override values; return originals for restoration."""
    saved = {}
    for key, new_val in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config parameter: '{key}'")
        saved[key] = getattr(config, key)
        setattr(config, key, new_val)
    return saved

def restore_overrides(saved: dict):
    """Restore config module to its original values."""
    for key, old_val in saved.items():
        setattr(config, key, old_val)

from run_simulation import (
    initialize_scholars, initialize_journals, period_zero_row,
    INSTITUTION_TYPES, INST_COLORS,
)
from agents.paper import Paper
from agents.journal import Journal
from functions.quality import compute_quality, assign_journal_tier
from config import (
    RANDOM_SEED, N_PERIODS, TENURE_TARGET_MIDPOINTS,
    COLORS,
)

# ── Thread lock — guards global config patching for concurrent sessions ───────
_SIM_LOCK = threading.Lock()

# ── Back-end defaults (used to detect changes and build overrides dict) ───────
DEFAULTS = {
    'AI_HIGH_PENALTY':            2.50,
    'AI_HIGH_EXPONENT':           1.50,   # fixed; never shown in UI
    'AI_PRODUCTIVITY_MULTIPLIER': 2.0,
    'BASE_STEP_POS':              0.05,
    'BASE_STEP_NEG':              0.01,
}

# ── Quality penalty conversion (user-facing % ↔ AI_HIGH_PENALTY) ─────────────
# Reference point: ai_use = 0.50 (approximate equilibrium), average scholar capacity.
#   avg_capacity  = mean of R1(0.85), R2(0.70), Balanced(0.60), Teaching(0.50) = 0.6625
#   linear_part   = AI_LOW_PENALTY × AI_LOW_THRESHOLD = 0.05 × 0.30 = 0.015
#   nonlin_factor = (0.50 − 0.30)^AI_HIGH_EXPONENT   = 0.20^1.50   ≈ 0.08944
#   penalty_at_50 = linear_part + AI_HIGH_PENALTY × nonlin_factor
#   pct_reduction = penalty_at_50 / avg_capacity × 100
# Default AI_HIGH_PENALTY=2.50 → pct ≈ 36 %.  AI_HIGH_EXPONENT fixed at 1.50.
_AVG_CAPACITY   = (0.85 + 0.70 + 0.60 + 0.50) / 4   # 0.6625
_LINEAR_PART    = 0.05 * 0.30                          # 0.015
_NONLIN_FACTOR  = (0.50 - 0.30) ** 1.50               # 0.08944
_DEFAULT_QUAL_PCT = 36   # % corresponding to AI_HIGH_PENALTY = 2.50

def _pct_to_penalty(pct: float) -> float:
    """Convert user-facing % quality reduction → AI_HIGH_PENALTY."""
    return (pct / 100.0 * _AVG_CAPACITY - _LINEAR_PART) / _NONLIN_FACTOR

N_REPS = 50   # always run full 50 replications


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached simulation runner
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_simulation_cached(overrides_tuple: tuple) -> tuple:
    """
    Run N_REPS replications with the given parameter overrides.

    Parameters
    ----------
    overrides_tuple : tuple of (key, value) pairs — hashable for Streamlit cache.
        Pass tuple() for the baseline (all defaults).

    Returns
    -------
    period_df : pd.DataFrame  — period-level stats (all reps)
    end_df    : pd.DataFrame  — per-scholar end state (all reps)
    """
    overrides = dict(overrides_tuple)

    with _SIM_LOCK:
        saved = apply_overrides(overrides)
        try:
            all_period, all_end = [], []
            for rep in range(N_REPS):
                rng      = np.random.default_rng(RANDOM_SEED + rep)
                scholars = initialize_scholars(rng)
                journals = initialize_journals()
                s_by_id  = {s.scholar_id: s for s in scholars}
                records  = [period_zero_row(scholars)]

                for period in range(1, N_PERIODS + 1):

                    # ── Step 0: Resubmission queue (capped; top quality first) ──
                    resub_outcomes = {s.scholar_id: [] for s in scholars}
                    for scholar in scholars:
                        q_sorted = sorted(scholar.resubmission_queue,
                                          key=lambda p: p.quality, reverse=True)
                        to_eval  = q_sorted[:config.MAX_RESUB_PER_PERIOD]
                        still    = list(q_sorted[config.MAX_RESUB_PER_PERIOD:])
                        for paper in to_eval:
                            journals[paper.journal_tier].evaluate(paper, rng)
                            resub_outcomes[scholar.scholar_id].append(paper.published)
                            if paper.published:
                                scholar.record_outcome(
                                    period, paper.quality, paper.journal_tier, True)
                            else:
                                paper.tier_attempts += 1
                                if paper.tier_attempts >= config.MAX_TIER_ATTEMPTS:
                                    if paper.journal_tier < 3:
                                        next_tier = paper.journal_tier + 1
                                        if next_tier == 3:
                                            floor = config.T3_FLOOR_FRACTION[
                                                scholar.institution_type]
                                            desperate = (
                                                floor is None or
                                                scholar.total_publications
                                                < floor * config.TENURE_TARGET_MIDPOINTS[
                                                    scholar.institution_type]
                                            )
                                        else:
                                            desperate = True
                                        if desperate:
                                            paper.journal_tier  = next_tier
                                            paper.tier_attempts = 0
                                            still.append(paper)
                                else:
                                    still.append(paper)
                        scholar.resubmission_queue = still

                    # ── Step 1: New paper production ──────────────────────────
                    period_papers = []
                    for scholar in scholars:
                        n = int(rng.poisson(scholar.papers_per_period))
                        for _ in range(n):
                            quality = float(np.clip(
                                compute_quality(scholar.research_capacity,
                                                scholar.ai_use_level),
                                0.01, 0.99))
                            tier = assign_journal_tier(quality,
                                                       scholar.institution_type,
                                                       scholar.tier1_publications)
                            if tier == 3:
                                floor = config.T3_FLOOR_FRACTION[
                                    scholar.institution_type]
                                if floor is not None:
                                    if (scholar.total_publications
                                            >= floor * config.TENURE_TARGET_MIDPOINTS[
                                                scholar.institution_type]):
                                        tier = 2
                            paper = Paper(
                                scholar_id      = scholar.scholar_id,
                                period_produced = period,
                                ai_use_level    = scholar.ai_use_level,
                                quality         = quality,
                                journal_tier    = tier,
                                original_tier   = tier,
                                tier_attempts   = 0,
                            )
                            journals[tier].evaluate(paper, rng)
                            scholar.record_outcome(period, paper.quality,
                                                   tier, paper.published)
                            period_papers.append(paper)
                            if not paper.published:
                                paper.tier_attempts = 1
                                scholar.resubmission_queue.append(paper)

                    # ── Step 2: Per-paper RL updates ──────────────────────────
                    for scholar in scholars:
                        for accepted in resub_outcomes[scholar.scholar_id]:
                            scholar.update_ai_use('accepted' if accepted else 'rejected')
                        for paper in period_papers:
                            if paper.scholar_id == scholar.scholar_id:
                                scholar.update_ai_use(
                                    'accepted' if paper.published else 'rejected')

                    # ── Step 3: Collect stats ─────────────────────────────────
                    n_prod    = len(period_papers)
                    n_acc_new = sum(1 for p in period_papers if p.published)
                    n_resub_e = sum(len(v) for v in resub_outcomes.values())
                    n_resub_a = sum(sum(1 for o in v if o)
                                   for v in resub_outcomes.values())

                    row = {
                        'period':            period,
                        'mean_ai_use':       float(np.mean(
                            [s.ai_use_level for s in scholars])),
                        'mean_quality':      (float(np.mean(
                            [p.quality for p in period_papers]))
                            if period_papers else np.nan),
                        'n_produced':        n_prod,
                        'n_accepted_new':    n_acc_new,
                        'n_resub_evaluated': n_resub_e,
                        'n_resub_accepted':  n_resub_a,
                        'acceptance_rate':   (n_acc_new / n_prod
                                              if n_prod else np.nan),
                    }
                    for inst in INSTITUTION_TYPES:
                        i_sch  = [s for s in scholars
                                  if s.institution_type == inst]
                        i_paps = [p for p in period_papers
                                  if s_by_id[p.scholar_id].institution_type == inst]
                        row[f'ai_{inst}']      = float(np.mean(
                            [s.ai_use_level for s in i_sch]))
                        row[f'q_{inst}']       = (float(np.mean(
                            [p.quality for p in i_paps]))
                            if i_paps else np.nan)
                        row[f'papersps_{inst}'] = (len(i_paps) / len(i_sch)
                                                   if i_sch else 0.0)
                    records.append(row)

                period_df = pd.DataFrame(records)
                period_df['replication'] = rep

                end_rows = []
                for s in scholars:
                    rec   = s.publication_record
                    all_q = [r['quality'] for r in rec]
                    acc_q = [r['quality'] for r in rec if r['accepted']]
                    end_rows.append({
                        'scholar_id':       s.scholar_id,
                        'institution_type': s.institution_type,
                        'ai_use_final':     s.ai_use_level,
                        'total_pubs':       s.total_publications,
                        'tier1_pubs':       s.tier1_publications,
                        'tier2_pubs':       s.tier2_publications,
                        'tier3_pubs':       sum(1 for r in rec
                                               if r['tier'] == 3 and r['accepted']),
                        'acceptance_rate':  (s.total_publications / len(rec)
                                             if rec else 0.0),
                        'mean_quality_all': (float(np.mean(all_q))
                                             if all_q else np.nan),
                    })
                end_df = pd.DataFrame(end_rows)
                end_df['replication'] = rep

                all_period.append(period_df)
                all_end.append(end_df)

        finally:
            restore_overrides(saved)

    return (pd.concat(all_period, ignore_index=True),
            pd.concat(all_end,    ignore_index=True))


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure builders
# ═══════════════════════════════════════════════════════════════════════════════

_FONT = {'family': 'sans-serif', 'size': 11}

def _ax_style(ax):
    """Minimal axis styling consistent with generate_report.py."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.7)


def fig_ai_trajectory(period_df, baseline_df=None) -> plt.Figure:
    """Tab 1: AI use over time by institution (period 0–12)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def _plot_lines(df, suffix, linestyle):
        agg = df.groupby('period')
        for inst in INSTITUTION_TYPES:
            col = f'ai_{inst}'
            means = agg[col].mean()
            sems  = agg[col].sem()
            ci    = 1.96 * sems
            ax.plot(means.index, means.values,
                    color=INST_COLORS[inst], linestyle=linestyle,
                    linewidth=2, label=f'{inst} {suffix}')
            ax.fill_between(means.index,
                            means.values - ci, means.values + ci,
                            color=INST_COLORS[inst], alpha=0.12)

    _plot_lines(period_df, '(Custom)', '-')
    if baseline_df is not None:
        _plot_lines(baseline_df, '(Baseline)', '--')

    ax.set_xlim(0, N_PERIODS)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Period  (0 = initial state; 1–12 = post-RL)', **_FONT)
    ax.set_ylabel('Mean AI Use Level', **_FONT)
    ax.set_title('AI Use Trajectory by Institution Type', **_FONT)
    ax.legend(fontsize=8, ncol=2 if baseline_df is not None else 1)
    _ax_style(ax)
    fig.tight_layout()
    return fig


def fig_publications(end_df, baseline_df=None) -> plt.Figure:
    """Tab 2: Mean total publications vs. tenure target midpoint."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x      = np.arange(len(INSTITUTION_TYPES))
    width  = 0.35 if baseline_df is not None else 0.55

    def _mean_se(df, inst):
        sub = df[df['institution_type'] == inst]['total_pubs']
        return sub.mean(), sub.sem()

    for i, inst in enumerate(INSTITUTION_TYPES):
        m, se = _mean_se(end_df, inst)
        offset = -width / 2 if baseline_df is not None else 0
        ax.bar(x[i] + offset, m, width, yerr=1.96 * se,
               color=INST_COLORS[inst], label=f'{inst} (Custom)',
               capsize=4, alpha=0.9)
        if baseline_df is not None:
            mb, seb = _mean_se(baseline_df, inst)
            ax.bar(x[i] + width / 2, mb, width, yerr=1.96 * seb,
                   color=INST_COLORS[inst], label=f'{inst} (Baseline)',
                   capsize=4, alpha=0.45, hatch='//')

    # Target lines
    for i, inst in enumerate(INSTITUTION_TYPES):
        tgt = TENURE_TARGET_MIDPOINTS[inst]
        ax.hlines(tgt, x[i] - 0.45, x[i] + 0.45,
                  colors='black', linestyles='--', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(INSTITUTION_TYPES)
    ax.set_ylabel('Mean Total Publications (± 95% CI)', **_FONT)
    ax.set_title('Publications vs. Tenure Target (dashed line)', **_FONT)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))   # deduplicate
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    _ax_style(ax)
    fig.tight_layout()
    return fig


def fig_tenure_attainment(end_df, baseline_df=None) -> plt.Figure:
    """Publications tab (lower panel): % of scholars meeting tenure target."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x     = np.arange(len(INSTITUTION_TYPES))
    width = 0.35 if baseline_df is not None else 0.55

    def _pct_met(df, inst):
        sub = df[df['institution_type'] == inst]
        tgt = TENURE_TARGET_MIDPOINTS[inst]
        pct = (sub['total_pubs'] >= tgt).mean() * 100
        se  = np.sqrt(pct/100 * (1 - pct/100) / len(sub)) * 100
        return pct, se

    for i, inst in enumerate(INSTITUTION_TYPES):
        pct, se = _pct_met(end_df, inst)
        offset  = -width / 2 if baseline_df is not None else 0
        ax.bar(x[i] + offset, pct, width, yerr=1.96 * se,
               color=INST_COLORS[inst], label=f'{inst} (Custom)',
               capsize=4, alpha=0.9)
        if baseline_df is not None:
            pctb, seb = _pct_met(baseline_df, inst)
            ax.bar(x[i] + width / 2, pctb, width, yerr=1.96 * seb,
                   color=INST_COLORS[inst], label=f'{inst} (Baseline)',
                   capsize=4, alpha=0.45, hatch='//')

    ax.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(INSTITUTION_TYPES)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of Scholars Meeting Target', **_FONT)
    ax.set_title('Scholars Reaching Tenure Publication Target', **_FONT)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    _ax_style(ax)
    fig.tight_layout()
    return fig


def fig_efficiency(period_df, baseline_df=None) -> plt.Figure:
    """Efficiency tab: papers submitted per scholar per period, line colored by AI use."""
    from matplotlib.collections import LineCollection

    cmap = plt.cm.plasma
    norm = plt.Normalize(0, 1)

    n_inst  = len(INSTITUTION_TYPES)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=False)
    axes = axes.flatten()

    def _plot_inst(ax, df, inst, linestyle, alpha_ci):
        sub  = df[df['period'] >= 1]
        agg  = sub.groupby('period')
        pers = np.array(sorted(sub['period'].unique()))

        papers = agg[f'papersps_{inst}'].mean().values
        ai_use = agg[f'ai_{inst}'].mean().values
        sems   = agg[f'papersps_{inst}'].sem().values

        # Draw CI band in institution color
        ax.fill_between(pers, papers - 1.96*sems, papers + 1.96*sems,
                        color=INST_COLORS[inst], alpha=alpha_ci)

        # Draw line segments colored by AI use level
        pts  = np.array([pers, papers]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap, norm=norm,
                              linewidth=2.5, linestyle=linestyle, zorder=3)
        lc.set_array(ai_use[:-1])
        ax.add_collection(lc)

        ax.set_xlim(1, N_PERIODS)
        ax.set_ylim(bottom=0)
        ax.autoscale_view(scalex=False)

    for ax, inst in zip(axes, INSTITUTION_TYPES):
        _plot_inst(ax, period_df, inst, '-', 0.15)
        if baseline_df is not None:
            _plot_inst(ax, baseline_df, inst, '--', 0.07)

        ax.set_title(inst, color=INST_COLORS[inst], fontweight='bold', **_FONT)
        ax.set_xlabel('Period', **_FONT)
        ax.set_ylabel('Papers submitted per scholar', **_FONT)
        _ax_style(ax)

    # Shared colorbar for AI use
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Mean AI Use Level', **_FONT)

    if baseline_df is not None:
        fig.suptitle(
            'Submission Volume Over Time  (solid = Custom, dashed = Baseline)\n'
            'Line color = AI use level at that period',
            **_FONT)
    else:
        fig.suptitle(
            'Submission Volume Over Time\nLine color = AI use level at that period',
            **_FONT)

    fig.tight_layout()
    return fig


def fig_quality(period_df, baseline_df=None) -> plt.Figure:
    """Tab 4: Mean paper quality over time by institution (periods 1–12)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def _plot_lines(df, suffix, linestyle):
        sub = df[df['period'] >= 1]
        agg = sub.groupby('period')
        for inst in INSTITUTION_TYPES:
            col   = f'q_{inst}'
            means = agg[col].mean()
            sems  = agg[col].sem()
            ci    = 1.96 * sems
            ax.plot(means.index, means.values,
                    color=INST_COLORS[inst], linestyle=linestyle,
                    linewidth=2, label=f'{inst} {suffix}')
            ax.fill_between(means.index,
                            means.values - ci, means.values + ci,
                            color=INST_COLORS[inst], alpha=0.12)

    _plot_lines(period_df, '(Custom)', '-')
    if baseline_df is not None:
        _plot_lines(baseline_df, '(Baseline)', '--')

    ax.set_xlim(1, N_PERIODS)
    ax.set_xlabel('Period', **_FONT)
    ax.set_ylabel('Mean Paper Quality', **_FONT)
    ax.set_title('Paper Quality Over Time by Institution Type', **_FONT)
    ax.legend(fontsize=8, ncol=2 if baseline_df is not None else 1)
    _ax_style(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary table builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_table(end_df, label='Custom') -> pd.DataFrame:
    rows = []
    for inst in INSTITUTION_TYPES:
        sub = end_df[end_df['institution_type'] == inst]
        rows.append({
            'Institution':    inst,
            'Scenario':       label,
            'Mean AI Use':    round(sub['ai_use_final'].mean(), 3),
            'Mean Pubs':      round(sub['total_pubs'].mean(), 1),
            'Tenure Target':  TENURE_TARGET_MIDPOINTS[inst],
            '% of Target':    round(
                sub['total_pubs'].mean() / TENURE_TARGET_MIDPOINTS[inst] * 100, 1),
            'Tier 1 Pubs':    round(sub['tier1_pubs'].mean(), 1),
            'Tier 2 Pubs':    round(sub['tier2_pubs'].mean(), 1),
            'Tier 3 Pubs':    round(sub['tier3_pubs'].mean(), 1),
            'Accept Rate':    round(sub['acceptance_rate'].mean(), 3),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='AI Publishing ABM',
    page_icon='📚',
    layout='wide',
)

st.title('📚 AI Use in Academic Publishing — ABM Explorer')
st.markdown(
    'Adjust the parameters below to see how AI adoption in academia affects '
    'publication outcomes under a 6-year tenure clock (12 half-year periods, '
    '200 scholars, 50 Monte Carlo replications).'
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Model Parameters')

    with st.expander('🎯 Quality Penalty', expanded=True):
        st.caption(
            'How much AI use reduces paper quality. Estimated for a scholar '
            'who uses AI for half their work, relative to average scholar '
            'research capacity across all institution types.'
        )
        quality_pct = st.slider(
            'Quality reduction at 50% AI use (%)',
            min_value=16, max_value=70, value=_DEFAULT_QUAL_PCT, step=1,
            help=(
                'At 36% (default), a paper that would score 0.70 without AI '
                'scores about 0.45 when AI handles half the work. '
                'Higher = AI use is more costly to quality. '
                'Default calibrated to Gartenberg et al. (2026, Organization Science): '
                '~1.28 SD quality decline at high AI use.'
            ),
        )

    with st.expander('🚀 Productivity Boost', expanded=True):
        st.caption(
            'How much AI multiplies paper output. '
            '1× = no effect. 2× = a scholar using AI at full intensity '
            'writes twice as many papers as one using no AI. '
            'The boost scales smoothly with AI use level.'
        )
        ai_multiplier = st.slider(
            'AI productivity multiplier',
            min_value=1.0, max_value=4.0, value=DEFAULTS['AI_PRODUCTIVITY_MULTIPLIER'],
            step=0.25,
            help='Default: 2.0× (upper-bound estimate; Noy & Zhang 2023, Dell\'Acqua et al. 2023)',
        )

    with st.expander('🧠 Reinforcement Learning', expanded=True):
        st.caption(
            'Scholars update their AI use level (on a 0–100% scale) after each paper '
            'accepted or rejected. Steps are in percentage points (pp) — '
            'e.g., a 5 pp positive step moves a scholar from 40% to 45% AI use. '
            'The positive step is amplified when scholars are far behind on tenure targets; '
            'the negative step is intentionally smaller, reflecting that scholars '
            'don\'t quickly abandon AI after a single rejection.'
        )
        step_pos_pp = st.slider(
            'AI use increase per accepted paper (percentage points, pp)',
            min_value=1, max_value=20, value=5, step=1,
            help='Default: 5 pp. At maximum publication pressure this step is doubled.',
        )
        step_neg_pp = st.slider(
            'AI use decrease per rejected paper (pp)',
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help='Default: 1 pp. Kept small so scholars don\'t abandon AI after one rejection.',
        )

    st.divider()
    compare_baseline = st.checkbox(
        'Compare with baseline', value=True,
        help='Also run the simulation with all default parameters and overlay results.',
    )
    run_clicked = st.button('▶ Run Simulation', type='primary', use_container_width=True)

# ── Build back-end overrides from UI values ───────────────────────────────────
overrides = {
    'AI_HIGH_PENALTY':            _pct_to_penalty(quality_pct),
    'AI_HIGH_EXPONENT':           1.50,                      # fixed; not exposed in UI
    'AI_PRODUCTIVITY_MULTIPLIER': ai_multiplier,
    'BASE_STEP_POS':              step_pos_pp / 100.0,
    'BASE_STEP_NEG':              step_neg_pp / 100.0,
}

# ── Parameter diff banner (user-friendly labels and units) ────────────────────
ui_changes = []
if abs(quality_pct - _DEFAULT_QUAL_PCT) > 0.5:
    ui_changes.append(f'**Quality reduction**: {_DEFAULT_QUAL_PCT}% → {quality_pct}%')
if abs(ai_multiplier - DEFAULTS['AI_PRODUCTIVITY_MULTIPLIER']) > 0.01:
    ui_changes.append(
        f'**Productivity multiplier**: {DEFAULTS["AI_PRODUCTIVITY_MULTIPLIER"]}× → {ai_multiplier}×')
if step_pos_pp != 5:
    ui_changes.append(f'**Positive RL step**: 5 pp → {step_pos_pp} pp')
if abs(step_neg_pp - 1.0) > 0.05:
    ui_changes.append(f'**Negative RL step**: 1.0 pp → {step_neg_pp} pp')

if ui_changes:
    st.info('🔧 Parameter changes from default: ' + ' | '.join(ui_changes))
else:
    st.success('✅ All parameters at default (baseline) values')

# ── Results area ──────────────────────────────────────────────────────────────
if run_clicked:
    overrides_tuple = tuple(sorted(overrides.items()))

    # Run custom (always) and baseline (if requested)
    with st.spinner(f'Running {N_REPS} replications… this takes ~15–30 seconds the first time.'):
        period_df, end_df = run_simulation_cached(overrides_tuple)
        if compare_baseline:
            baseline_period_df, baseline_end_df = run_simulation_cached(tuple())
        else:
            baseline_period_df = baseline_end_df = None

    st.success('✅ Simulation complete!')

    # ── 4 result tabs ─────────────────────────────────────────────────────────
    tab_ai, tab_pubs, tab_efficiency, tab_quality = st.tabs([
        '📈 AI Use Trajectory',
        '📊 Publications',
        '⚡ Efficiency Over Time',
        '🔬 Quality Over Time',
    ])

    with tab_ai:
        st.subheader('AI Use Trajectory by Institution Type')
        st.caption(
            'Mean AI use level (± 95% CI across 50 replications). '
            'Period 0 = initial state drawn from survey (Nag et al., 2025); '
            'periods 1–12 reflect RL updates after each paper accepted/rejected.'
        )
        fig = fig_ai_trajectory(period_df, baseline_period_df)
        st.pyplot(fig, clear_figure=True)

    with tab_pubs:
        st.subheader('Publications vs. Tenure Target')
        st.caption(
            'Mean total publications at end of simulation (bars, ± 95% CI). '
            'Dashed horizontal line = tenure target midpoint for that institution type.'
        )
        fig = fig_publications(end_df, baseline_end_df)
        st.pyplot(fig, clear_figure=True)

        st.subheader('Scholars Reaching Tenure Target')
        st.caption(
            '% of scholars who accumulated at least the tenure target midpoint in publications '
            'by end of the simulation (± 95% CI). Dotted line at 50% for reference.'
        )
        fig = fig_tenure_attainment(end_df, baseline_end_df)
        st.pyplot(fig, clear_figure=True)

    with tab_efficiency:
        st.subheader('Submission Volume Over Time')
        st.caption(
            'Mean new papers submitted per scholar each period (± 95% CI shading). '
            'Line color shows the mean AI use level at that period — warmer colors '
            '(yellow/orange) indicate higher AI use; cooler colors (purple/dark) '
            'indicate lower AI use. Shows how AI adoption drives submission volume.'
        )
        fig = fig_efficiency(period_df, baseline_period_df)
        st.pyplot(fig, clear_figure=True)

    with tab_quality:
        st.subheader('Mean Paper Quality Over Time')
        st.caption(
            'Mean quality of new papers produced each period (± 95% CI). '
            'Quality reflects research capacity modulated by the AI quality penalty.'
        )
        fig = fig_quality(period_df, baseline_period_df)
        st.pyplot(fig, clear_figure=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader('📋 Summary Table')
    summary = build_summary_table(end_df, label='Custom')
    if baseline_end_df is not None:
        summary_base = build_summary_table(baseline_end_df, label='Baseline')
        summary = pd.concat([summary, summary_base], ignore_index=True) \
                    .sort_values(['Institution', 'Scenario']) \
                    .reset_index(drop=True)

    st.dataframe(
        summary.style.format({
            'Mean AI Use': '{:.3f}',
            'Mean Pubs':   '{:.1f}',
            '% of Target': '{:.1f}%',
            'Tier 1 Pubs': '{:.1f}',
            'Tier 2 Pubs': '{:.1f}',
            'Tier 3 Pubs': '{:.1f}',
            'Accept Rate': '{:.3f}',
        }),
        use_container_width=True,
    )

    # ── Parameter reference ───────────────────────────────────────────────────
    with st.expander('ℹ️ Parameter Reference', expanded=False):
        st.markdown("""
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Quality reduction at 50% AI use | 36% | 16–70% | Estimated quality reduction for an average scholar who uses AI for half their work (relative to no AI use). Translates internally to the nonlinear penalty coefficient. Higher = AI use is more costly to paper quality. Calibrated to Gartenberg et al. (2026). |
| AI productivity multiplier | 2× | 1–4× | At maximum AI use, scholars produce this many times more papers than at zero AI use. 1× = no effect. Calibrated to Noy & Zhang (2023), Dell'Acqua et al. (2023). |
| Positive RL step | 5 pp | 1–20 pp | Each accepted paper increases a scholar's AI use level by this many percentage points (pp). At maximum publication pressure the step is doubled. |
| Negative RL step | 1 pp | 0.1–5 pp | Each rejected paper decreases AI use by this many percentage points (pp). Kept small: scholars don't quickly abandon AI after a single rejection. |

**Break-even acceptance rate** (for AI use to grow on average at max pressure):

$$p_{\\text{accept}} > \\frac{\\text{BASE\\_STEP\\_NEG}}{\\text{BASE\\_STEP\\_POS} \\times (1 + \\text{PRESSURE\\_WEIGHT}) + \\text{BASE\\_STEP\\_NEG}}$$

With defaults: $p > 0.01 / (0.10 + 0.01) \\approx 9\\%$

**Four institution types** (50 scholars each):
- **R1**: High-research, target 8–12 total pubs, 3–6 Tier 1 required.
- **R2**: Research-active, target 6–12 total pubs, 1–3 Tier 1 required.
- **Balanced**: Teaching + research, target 4–8 pubs, Tier 2 emphasis.
- **Teaching**: Teaching-focused, target 1–3 pubs, Tier 3 norm.
        """)

else:
    st.info('👈 Adjust parameters in the sidebar and click **▶ Run Simulation** to begin.')
    st.markdown("""
**How it works:**

This webapp runs an agent-based model (ABM) of how AI adoption spreads in academia under publish-or-perish pressure.

- **200 scholar agents** across 4 institution types complete a simulated 6-year tenure clock (12 half-year periods).
- Each period, scholars produce papers and submit them to journals in one of three tiers.
- Scholars update their AI use level after **each paper** accepted or rejected (reinforcement learning with publication pressure).
- AI use **boosts productivity** (more papers) but **degrades quality** (piecewise penalty).
- Rejected papers enter a resubmission queue and cascade to lower tiers before being abandoned.
- Results shown are **means ± 95% CI across 50 Monte Carlo replications** for statistical reliability.
    """)
