"""
run_scenarios.py — Parameter sensitivity explorer for the AI Publishing ABM.

─────────────────────────────────────────────────────────────────────────────
HOW TO USE
─────────────────────────────────────────────────────────────────────────────
1. Edit the SCENARIOS dict below.  Each entry is a named scenario with a
   dict of parameter overrides (leave a value out to keep the default).
2. Optionally adjust REPS_PER_SCENARIO and OUTPUT_DIR.
3. Run:  python3 run_scenarios.py

Outputs saved to outputs/scenarios/:
  scenario_summary.csv   — one row per scenario × institution
  fig_ai_use.png         — final AI use by scenario × institution
  fig_publications.png   — total publications vs tenure targets
  fig_tiers.png          — tier breakdown (T1 / T2 / T3) per institution
  fig_quality.png        — mean paper quality trajectory (overall)
  fig_ai_trajectory.png  — AI use over time (all scenarios, R1 + Balanced)

─────────────────────────────────────────────────────────────────────────────
TUNABLE PARAMETERS — what each one does
─────────────────────────────────────────────────────────────────────────────

  Quality penalty (how much AI use hurts paper quality)
  ─────────────────────────────────────────────────────
  AI_LOW_THRESHOLD  (default 0.30)  — AI use below this gets a mild linear
                                      penalty; above it the nonlinear term kicks in.
  AI_LOW_PENALTY    (default 0.05)  — slope of the linear (low-use) penalty.
  AI_HIGH_PENALTY   (default 2.50)  — coefficient of the nonlinear penalty above
                                      the threshold. Higher = steeper quality drop.
  AI_HIGH_EXPONENT  (default 1.50)  — exponent of the nonlinear term (>1 means
                                      accelerating penalty at very high AI use).

  Productivity boost (how much AI use increases paper output)
  ──────────────────────────────────────────────────────────
  AI_PRODUCTIVITY_MULTIPLIER (default 2.0) — papers_per_period scales as
      base_rate × multiplier^ai_use.  At ai=1.0: 2× more papers than at ai=0.
      Set to 1.0 for no productivity effect.

  Reinforcement learning (how AI use responds to accept/reject signals)
  ─────────────────────────────────────────────────────────────────────
  BASE_STEP_POS  (default 0.05)  — ai_use increase per accepted paper.
  BASE_STEP_NEG  (default 0.01)  — ai_use decrease per rejected paper (smaller
                                   = scholars don't quickly abandon AI).
  PRESSURE_WEIGHT (default 1.0)  — multiplier on the positive step at max
                                   publication pressure. Higher = more urgent
                                   AI adoption when behind on tenure targets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
#  EDIT THIS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    # ── Baseline ────────────────────────────────────────────────────────────────
    'Baseline': {},   # empty dict = use every default from config.py

    # ── Quality penalty variants ─────────────────────────────────────────────────
    'Weak quality penalty': {
        'AI_HIGH_PENALTY': 1.50,    # default 2.50  — shallower quality drop at high AI use
    },
    'Strong quality penalty': {
        'AI_HIGH_PENALTY': 3.50,    # default 2.50  — steeper quality drop at high AI use
    },

    # ── Productivity boost variants ───────────────────────────────────────────────
    'Low productivity boost': {
        'AI_PRODUCTIVITY_MULTIPLIER': 1.50,   # default 2.0  — AI only 50% more productive
    },
    'High productivity boost': {
        'AI_PRODUCTIVITY_MULTIPLIER': 3.00,   # default 2.0  — AI 3× more productive
    },

    # ── RL step size variants ─────────────────────────────────────────────────────
    'Faster RL': {
        'BASE_STEP_POS': 0.10,   # default 0.05  — larger positive step per acceptance
        'BASE_STEP_NEG': 0.02,   # default 0.01  — larger negative step per rejection
    },
    'Slower RL': {
        'BASE_STEP_POS': 0.025,  # default 0.05  — smaller positive step per acceptance
        'BASE_STEP_NEG': 0.005,  # default 0.01  — smaller negative step per rejection
    },
}

# Number of Monte Carlo replications per scenario.
# Use fewer for faster exploration; more for tighter confidence intervals.
# (Baseline uses config.py N_REPLICATIONS = 50 by default.)
REPS_PER_SCENARIO = 20

OUTPUT_DIR = 'outputs/scenarios'

# ═══════════════════════════════════════════════════════════════════════════════
#  NO NEED TO EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config                          # we patch attributes on this module object
from run_simulation import (           # reuse the simulation machinery
    initialize_scholars, initialize_journals, period_zero_row,
    INSTITUTION_TYPES, COLORS, INST_COLORS,
)
from config import (
    RANDOM_SEED, N_PERIODS, TENURE_TARGET_MIDPOINTS,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Scenario-aware parameter patching ────────────────────────────────────────

def apply_overrides(overrides: dict) -> dict:
    """
    Patch config module with override values.  Returns a dict of the original
    values so they can be restored afterwards.
    """
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


# ── Single-scenario simulation run ───────────────────────────────────────────

def run_scenario(name: str, overrides: dict, n_reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run n_reps replications with the given overrides; return (period_df, end_df)."""
    from agents.paper import Paper
    from agents.journal import Journal
    from functions.quality import compute_quality, assign_journal_tier

    saved = apply_overrides(overrides)
    try:
        all_period, all_end = [], []
        for rep in range(n_reps):
            rng      = np.random.default_rng(RANDOM_SEED + rep)
            scholars = initialize_scholars(rng)
            journals = initialize_journals()
            s_by_id  = {s.scholar_id: s for s in scholars}
            records  = [period_zero_row(scholars)]

            for period in range(1, N_PERIODS + 1):

                # Resubmission queue (capped at MAX_RESUB_PER_PERIOD; top quality first)
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
                                        floor = config.T3_FLOOR_FRACTION[scholar.institution_type]
                                        desperate = (
                                            floor is None or
                                            scholar.total_publications
                                            < floor * config.TENURE_TARGET_MIDPOINTS[scholar.institution_type]
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

                # New paper production
                period_papers = []
                for scholar in scholars:
                    n = int(rng.poisson(scholar.papers_per_period))
                    for _ in range(n):
                        quality = float(np.clip(
                            compute_quality(scholar.research_capacity, scholar.ai_use_level),
                            0.01, 0.99))
                        tier = assign_journal_tier(quality, scholar.institution_type,
                                                   scholar.tier1_publications)
                        if tier == 3:
                            floor = config.T3_FLOOR_FRACTION[scholar.institution_type]
                            if floor is not None:
                                if scholar.total_publications >= floor * config.TENURE_TARGET_MIDPOINTS[scholar.institution_type]:
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
                        scholar.record_outcome(period, paper.quality, tier, paper.published)
                        period_papers.append(paper)
                        if not paper.published:
                            paper.tier_attempts = 1
                            scholar.resubmission_queue.append(paper)

                # Per-paper RL updates
                for scholar in scholars:
                    for accepted in resub_outcomes[scholar.scholar_id]:
                        scholar.update_ai_use('accepted' if accepted else 'rejected')
                    for paper in period_papers:
                        if paper.scholar_id == scholar.scholar_id:
                            scholar.update_ai_use('accepted' if paper.published else 'rejected')

                # Collect stats
                n_prod    = len(period_papers)
                n_acc_new = sum(1 for p in period_papers if p.published)
                n_resub_e = sum(len(v) for v in resub_outcomes.values())
                n_resub_a = sum(sum(1 for o in v if o) for v in resub_outcomes.values())

                row = {
                    'period':           period,
                    'mean_ai_use':      float(np.mean([s.ai_use_level for s in scholars])),
                    'mean_quality':     float(np.mean([p.quality for p in period_papers])) if period_papers else np.nan,
                    'n_produced':       n_prod,
                    'n_accepted_new':   n_acc_new,
                    'n_resub_evaluated':n_resub_e,
                    'n_resub_accepted': n_resub_a,
                    'acceptance_rate':  n_acc_new / n_prod if n_prod else np.nan,
                }
                for inst in INSTITUTION_TYPES:
                    i_sch  = [s for s in scholars if s.institution_type == inst]
                    i_paps = [p for p in period_papers
                              if s_by_id[p.scholar_id].institution_type == inst]
                    i_acc  = sum(1 for p in i_paps if p.published)
                    row[f'ai_{inst}']   = float(np.mean([s.ai_use_level for s in i_sch]))
                    row[f'q_{inst}']    = float(np.mean([p.quality for p in i_paps])) if i_paps else np.nan
                    row[f'acc_{inst}']  = i_acc / len(i_paps) if i_paps else np.nan
                records.append(row)

            period_df = pd.DataFrame(records)
            period_df['replication'] = rep

            end_rows = []
            for s in scholars:
                rec   = s.publication_record
                all_q = [r['quality'] for r in rec]
                acc_q = [r['quality'] for r in rec if r['accepted']]
                end_rows.append({
                    'scholar_id':            s.scholar_id,
                    'institution_type':      s.institution_type,
                    'ai_use_final':          s.ai_use_level,
                    'total_pubs':            s.total_publications,
                    'tier1_pubs':            s.tier1_publications,
                    'tier2_pubs':            s.tier2_publications,
                    'tier3_pubs':            sum(1 for r in rec if r['tier']==3 and r['accepted']),
                    'acceptance_rate':       s.total_publications / len(rec) if rec else 0.0,
                    'mean_quality_all':      float(np.mean(all_q)) if all_q else np.nan,
                    'mean_quality_accepted': float(np.mean(acc_q)) if acc_q else np.nan,
                })
            end_df = pd.DataFrame(end_rows)
            end_df['replication'] = rep

            all_period.append(period_df)
            all_end.append(end_df)

    finally:
        restore_overrides(saved)   # always restore even if an error occurs

    return pd.concat(all_period, ignore_index=True), pd.concat(all_end, ignore_index=True)


# ── Run all scenarios ─────────────────────────────────────────────────────────

def run_all_scenarios():
    results = {}
    for name, overrides in SCENARIOS.items():
        print(f"  Running '{name}' ({REPS_PER_SCENARIO} reps)  overrides: {overrides or '(none)'}")
        period_df, end_df = run_scenario(name, overrides, REPS_PER_SCENARIO)
        results[name] = {'period': period_df, 'end': end_df}
    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(results: dict) -> pd.DataFrame:
    rows = []
    for name, dfs in results.items():
        end_df = dfs['end']
        for inst in INSTITUTION_TYPES:
            sub = end_df[end_df['institution_type'] == inst]
            rows.append({
                'scenario':         name,
                'institution':      inst,
                'ai_use_final':     sub['ai_use_final'].mean(),
                'total_pubs':       sub['total_pubs'].mean(),
                'tier1_pubs':       sub['tier1_pubs'].mean(),
                'tier2_pubs':       sub['tier2_pubs'].mean(),
                'tier3_pubs':       sub['tier3_pubs'].mean(),
                'acceptance_rate':  sub['acceptance_rate'].mean(),
                'mean_quality_all': sub['mean_quality_all'].mean(),
                'tenure_target':    TENURE_TARGET_MIDPOINTS[inst],
                'pct_of_target':    sub['total_pubs'].mean() / TENURE_TARGET_MIDPOINTS[inst] * 100,
            })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

SCENARIO_COLORS = [
    '#981A31', '#00546B', '#4A7C59', '#C4A35A',
    '#5E2154', '#D4856A', '#2E86AB', '#A23B72',
]

plt.rcParams.update({
    'font.family':       'sans-serif',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'legend.fontsize':   8,
})

inst_labels = {
    'R1': 'R1', 'R2': 'R2',
    'Balanced': 'Balanced', 'Teaching': 'Teaching-Focused',
}


def _scenario_colors(names):
    return {n: SCENARIO_COLORS[i % len(SCENARIO_COLORS)] for i, n in enumerate(names)}


def plot_final_ai_use(summary: pd.DataFrame, scenario_names: list):
    """Grouped bar: final AI use by scenario × institution."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    fig.suptitle('Final AI Use Level by Scenario and Institution Type',
                 fontweight='bold', color=COLORS['primary'], y=1.02)

    scolors = _scenario_colors(scenario_names)
    x       = np.arange(len(scenario_names))
    width   = 0.65

    for ax, inst in zip(axes, INSTITUTION_TYPES):
        sub = summary[summary['institution'] == inst].set_index('scenario').reindex(scenario_names)
        bars = ax.bar(x, sub['ai_use_final'], width,
                      color=[scolors[n] for n in scenario_names],
                      edgecolor='white', alpha=0.88)
        # Baseline reference line
        baseline = summary[(summary['scenario'] == scenario_names[0]) &
                           (summary['institution'] == inst)]['ai_use_final'].values
        if len(baseline):
            ax.axhline(baseline[0], color='black', lw=1.2, ls='--', alpha=0.5)
        ax.set_title(inst_labels[inst], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=7.5)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel('Mean Final AI Use Level')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_ai_use.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_ai_use.png")


def plot_publications(summary: pd.DataFrame, scenario_names: list):
    """Grouped bar: total publications vs tenure target, by scenario × institution."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle('Total Publications vs. Tenure Target Midpoint by Scenario',
                 fontweight='bold', color=COLORS['primary'], y=1.02)

    scolors = _scenario_colors(scenario_names)
    x       = np.arange(len(scenario_names))

    for ax, inst in zip(axes, INSTITUTION_TYPES):
        sub = summary[summary['institution'] == inst].set_index('scenario').reindex(scenario_names)
        ax.bar(x, sub['total_pubs'],
               color=[scolors[n] for n in scenario_names],
               edgecolor='white', alpha=0.88)
        target = TENURE_TARGET_MIDPOINTS[inst]
        ax.axhline(target, color='black', lw=2, ls='--', label=f'Target {target}')
        ax.set_title(inst_labels[inst], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=7.5)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel('Mean Total Publications')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_publications.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_publications.png")


def plot_tiers(summary: pd.DataFrame, scenario_names: list):
    """Stacked bar: T1/T2/T3 breakdown, one panel per institution."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle('Tier Breakdown by Scenario and Institution Type',
                 fontweight='bold', color=COLORS['primary'], y=1.02)

    x = np.arange(len(scenario_names))
    tier_colors = [COLORS['primary'], COLORS['secondary'], COLORS['gold']]

    for ax, inst in zip(axes, INSTITUTION_TYPES):
        sub = summary[summary['institution'] == inst].set_index('scenario').reindex(scenario_names)
        t1 = sub['tier1_pubs'].values
        t2 = sub['tier2_pubs'].values
        t3 = sub['tier3_pubs'].values
        ax.bar(x, t1, color=tier_colors[0], label='Tier 1', edgecolor='white', alpha=0.9)
        ax.bar(x, t2, bottom=t1, color=tier_colors[1], label='Tier 2', edgecolor='white', alpha=0.9)
        ax.bar(x, t3, bottom=t1+t2, color=tier_colors[2], label='Tier 3', edgecolor='white', alpha=0.9)
        ax.set_title(inst_labels[inst], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=7.5)

    axes[0].set_ylabel('Mean Publications')
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_tiers.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_tiers.png")


def plot_ai_trajectory(results: dict, scenario_names: list):
    """Line plots: AI use over time for R1 and Balanced across all scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    scolors   = _scenario_colors(scenario_names)

    for inst, ax in zip(['R1', 'Balanced'], axes):
        for name in scenario_names:
            pdf = results[name]['period']
            col = f'ai_{inst}'
            g   = pdf.groupby('period')[col]
            m   = g.mean()
            ax.plot(m.index, m, color=scolors[name], lw=2.0,
                    ls='-' if name == scenario_names[0] else '--',
                    label=name, alpha=0.9)
        ax.axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.4)
        ax.set_title(f'{inst} — AI Use Over Time', fontweight='bold')
        ax.set_xlabel('Period (0 = initial)')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, N_PERIODS)

    axes[0].set_ylabel('Mean AI Use Level')
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.suptitle('AI Use Trajectory by Scenario (R1 and Balanced)',
                 fontweight='bold', color=COLORS['primary'])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_ai_trajectory.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_ai_trajectory.png")


def plot_quality(results: dict, scenario_names: list):
    """Line plots: mean paper quality over time, all scenarios."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    scolors = _scenario_colors(scenario_names)

    for name in scenario_names:
        pdf = results[name]['period']
        g   = pdf[pdf['period'] > 0].groupby('period')['mean_quality']
        m   = g.mean()
        ax.plot(m.index, m, color=scolors[name], lw=2.0,
                ls='-' if name == scenario_names[0] else '--',
                label=name, alpha=0.9)

    ax.axhline(0.829, color='gray', lw=1, ls=':', alpha=0.5, label='T1 threshold')
    ax.axhline(0.707, color='gray', lw=1, ls=':', alpha=0.4, label='T2 threshold')
    ax.set_xlabel('Period')
    ax.set_ylabel('Mean New-Paper Quality (overall)')
    ax.set_title('Mean Paper Quality Over Time by Scenario',
                 fontweight='bold', color=COLORS['primary'])
    ax.set_xlim(1, N_PERIODS)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_quality.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR}/fig_quality.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scenario_names = list(SCENARIOS.keys())

    print(f"\nAI Publishing ABM — Scenario Explorer")
    print(f"Scenarios: {scenario_names}")
    print(f"Replications per scenario: {REPS_PER_SCENARIO}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    results = run_all_scenarios()

    print("\nBuilding summary and figures...")
    summary = build_summary(results)
    summary.to_csv(f'{OUTPUT_DIR}/scenario_summary.csv', index=False)
    print(f"Saved {OUTPUT_DIR}/scenario_summary.csv")

    plot_final_ai_use(summary, scenario_names)
    plot_publications(summary, scenario_names)
    plot_tiers(summary, scenario_names)
    plot_ai_trajectory(results, scenario_names)
    plot_quality(results, scenario_names)

    # Print console summary
    print("\n=== SCENARIO SUMMARY ===")
    for inst in INSTITUTION_TYPES:
        print(f"\n  {inst} (target = {TENURE_TARGET_MIDPOINTS[inst]}):")
        print(f"  {'Scenario':<28}  AI_final  Total_pubs  T1   T2   T3   % target")
        sub = summary[summary['institution'] == inst]
        for _, row in sub.iterrows():
            print(f"  {row['scenario']:<28}  {row['ai_use_final']:.3f}     "
                  f"{row['total_pubs']:>5.1f}    {row['tier1_pubs']:>4.1f} "
                  f"{row['tier2_pubs']:>4.1f} {row['tier3_pubs']:>4.1f}  "
                  f"{row['pct_of_target']:>5.1f}%")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")
