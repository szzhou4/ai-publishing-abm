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
from run_scenarios import apply_overrides, restore_overrides
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

                    # ── Step 0: Resubmission queue ────────────────────────────
                    resub_outcomes = {s.scholar_id: [] for s in scholars}
                    for scholar in scholars:
                        still = []
                        for paper in scholar.resubmission_queue:
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
                        row[f'ai_{inst}'] = float(np.mean(
                            [s.ai_use_level for s in i_sch]))
                        row[f'q_{inst}']  = (float(np.mean(
                            [p.quality for p in i_paps]))
                            if i_paps else np.nan)
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


def fig_tier_breakdown(end_df, baseline_df=None) -> plt.Figure:
    """Tab 3: Stacked bar — mean T1 / T2 / T3 pubs at end of sim."""
    fig, axes = plt.subplots(1, 2 if baseline_df is not None else 1,
                             figsize=(10 if baseline_df is not None else 6, 4.5),
                             sharey=True)
    if baseline_df is None:
        axes = [axes]

    datasets = [(end_df, 'Custom Parameters')]
    if baseline_df is not None:
        datasets.append((baseline_df, 'Baseline'))

    tier_cols   = ['tier1_pubs', 'tier2_pubs', 'tier3_pubs']
    tier_labels = ['Tier 1', 'Tier 2', 'Tier 3']
    tier_colors = [COLORS['primary'], COLORS['secondary'], COLORS['gold']]

    for ax, (df, title) in zip(axes, datasets):
        x      = np.arange(len(INSTITUTION_TYPES))
        bottoms = np.zeros(len(INSTITUTION_TYPES))
        for col, label, color in zip(tier_cols, tier_labels, tier_colors):
            means = [df[df['institution_type'] == inst][col].mean()
                     for inst in INSTITUTION_TYPES]
            ax.bar(x, means, bottom=bottoms, color=color,
                   label=label, alpha=0.88)
            bottoms += np.array(means)
        ax.set_xticks(x)
        ax.set_xticklabels(INSTITUTION_TYPES)
        ax.set_title(title, **_FONT)
        ax.legend(fontsize=8)
        _ax_style(ax)

    axes[0].set_ylabel('Mean Publications', **_FONT)
    fig.suptitle('Publication Tier Breakdown by Institution', **_FONT, y=1.01)
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
    tab_ai, tab_pubs, tab_tiers, tab_quality = st.tabs([
        '📈 AI Use Trajectory',
        '📊 Publications',
        '🏆 Tier Breakdown',
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

    with tab_tiers:
        st.subheader('Publication Tier Breakdown')
        st.caption(
            'Mean Tier 1 / Tier 2 / Tier 3 publications per scholar at end of simulation.'
        )
        fig = fig_tier_breakdown(end_df, baseline_end_df)
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
