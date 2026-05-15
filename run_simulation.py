"""
run_simulation.py — Standalone simulation runner (v5).

Changes from v4:
  - T3_FLOOR_FRACTION policy: R1 scholars avoid Tier 3 unless total_publications
    < 40% of tenure target midpoint (< 4 pubs). Applies to both new paper tier
    assignment (redirected to T2 if not desperate) and resubmission cascade
    (abandoned rather than downgraded to T3 if not desperate).

Changes from v3:
  - RL update fires per paper (new + resubmission), not once per period.
    All papers in a period are produced/evaluated first; RL updates applied
    after, so within-period quality calculations use a consistent ai_use
    snapshot. This allows AI use to self-correct when per-paper acceptance
    rates fall below the ~9% break-even.
  - Period 0 statistics recorded (initial state before any evaluations),
    so figures show the true starting AI use distribution.
  - Balanced T2 quality threshold lowered (0.60 → 0.45) so more Balanced
    papers attempt Tier 2 before cascading to Tier 3.

Changes from v2:
  - Resubmission pipeline: rejected papers enter the scholar's resubmission_queue
    and are re-evaluated each subsequent period (same quality, new noise draw).
    After MAX_TIER_ATTEMPTS successive rejections at one tier, the paper
    downgrades to the next tier; abandoned if rejected MAX_TIER_ATTEMPTS times
    at Tier 3. Resubmissions do NOT count toward papers_per_period but DO count
    toward total_publications and the RL signal when accepted.

Changes from v1:
  - Publication lag removed; papers are produced and evaluated within the same period.
  - Base production rates raised to reflect realistic submission volumes.
  - AI productivity multiplier set to 2.0x (literature-anchored; Noy & Zhang 2023).
  - Percentile-threshold acceptance model replaces logistic function.

Runs N_REPLICATIONS of the AI Publishing ABM and saves:
  outputs/sim_results.csv        — period-level statistics (all replications)
  outputs/scholar_endstate.csv   — per-scholar state at end of simulation

Usage:
    python run_simulation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import (
    N_SCHOLARS, N_PERIODS, N_REPLICATIONS, RANDOM_SEED,
    INSTITUTION_TYPES, INSTITUTION_DISTRIBUTION,
    RESEARCH_CAPACITY_PARAMS,
    NAG_AI_VALUES, NAG_AI_PROPORTIONS,
    BASE_PRODUCTION_RATE, TENURE_TARGET_MIDPOINTS,
    AI_PRODUCTIVITY_MULTIPLIER,
    JOURNAL_TIERS, COLORS, INST_COLORS,
    TIER1_PUB_TARGETS,
    MAX_TIER_ATTEMPTS,
    T3_FLOOR_FRACTION,
)
from functions.quality import compute_quality, assign_journal_tier
from agents.scholar import Scholar
from agents.paper import Paper
from agents.journal import Journal

os.makedirs('outputs', exist_ok=True)


# ── Initialization helpers ────────────────────────────────────────────────────

def initialize_scholars(rng: np.random.Generator) -> list:
    scholars = []
    inst_counts = rng.multinomial(N_SCHOLARS, INSTITUTION_DISTRIBUTION)
    nag_probs = np.array(NAG_AI_PROPORTIONS, dtype=float)
    nag_probs = nag_probs / nag_probs.sum()
    sid = 0
    for inst, count in zip(INSTITUTION_TYPES, inst_counts):
        params = RESEARCH_CAPACITY_PARAMS[inst]
        capacities = rng.normal(params['mean'], params['sd'], count)
        capacities = np.clip(capacities, 0.01, 0.99)
        for cap in capacities:
            ai_init = float(rng.choice(NAG_AI_VALUES, p=nag_probs))
            scholars.append(Scholar(sid, float(cap), inst, ai_init))
            sid += 1
    return scholars


def initialize_journals() -> dict:
    return {tier: Journal(journal_id=tier, tier=tier) for tier in JOURNAL_TIERS}


def period_zero_row(scholars: list) -> dict:
    """Snapshot of initial scholar state before any evaluations (period 0)."""
    row = {
        'period':              0,
        'mean_ai_use':         float(np.mean([s.ai_use_level for s in scholars])),
        'mean_quality':        np.nan,
        'n_produced':          0,
        'n_accepted_new':      0,
        'n_resub_evaluated':   0,
        'n_resub_accepted':    0,
        'n_accepted_all':      0,
        'acceptance_rate':     np.nan,
        'acceptance_rate_all': np.nan,
    }
    for inst in INSTITUTION_TYPES:
        i_sch = [s for s in scholars if s.institution_type == inst]
        row[f'ai_{inst}']    = float(np.mean([s.ai_use_level for s in i_sch]))
        row[f'q_{inst}']     = np.nan
        row[f'acc_{inst}']   = np.nan
        row[f'nprod_{inst}'] = 0
    return row


# ── Single replication ────────────────────────────────────────────────────────

def run_one_replication(seed: int) -> tuple:
    rng      = np.random.default_rng(seed)
    scholars = initialize_scholars(rng)
    journals = initialize_journals()
    s_by_id  = {s.scholar_id: s for s in scholars}

    records = [period_zero_row(scholars)]   # period 0 = initial state

    for period in range(1, N_PERIODS + 1):

        # ── Step 0: Evaluate resubmission queue ──────────────────────────────
        # Evaluate all queued papers with fresh noise; track outcomes for RL.
        # Accepted resubmissions are recorded immediately; rejected ones are
        # managed in the queue (tier downgrade or abandonment).
        # RL signals are collected here but applied after new-paper production.
        resub_outcomes = {s.scholar_id: [] for s in scholars}

        for scholar in scholars:
            still_in_queue = []
            for paper in scholar.resubmission_queue:
                journals[paper.journal_tier].evaluate(paper, rng)
                resub_outcomes[scholar.scholar_id].append(paper.published)
                if paper.published:
                    scholar.record_outcome(
                        period, paper.quality, paper.journal_tier, True)
                else:
                    paper.tier_attempts += 1
                    if paper.tier_attempts >= MAX_TIER_ATTEMPTS:
                        if paper.journal_tier < 3:
                            next_tier = paper.journal_tier + 1
                            # T3 avoidance: check institutional policy before
                            # cascading into Tier 3.
                            if next_tier == 3:
                                floor = T3_FLOOR_FRACTION[scholar.institution_type]
                                desperate = (
                                    floor is None or
                                    scholar.total_publications
                                    < floor * TENURE_TARGET_MIDPOINTS[scholar.institution_type]
                                )
                            else:
                                desperate = True   # T1→T2 always allowed
                            if desperate:
                                paper.journal_tier  = next_tier
                                paper.tier_attempts = 0
                                still_in_queue.append(paper)
                            # else: not desperate → abandon rather than submit to T3
                        # else: Tier 3 exhausted → abandon
                    else:
                        still_in_queue.append(paper)
            scholar.resubmission_queue = still_in_queue

        # ── Step 1: Produce and evaluate new papers ───────────────────────────
        # All papers use the current ai_use_level snapshot (before this period's
        # RL updates fire). Quality and tier are fixed at production time.
        period_papers = []
        for scholar in scholars:
            n = int(rng.poisson(scholar.papers_per_period))
            for _ in range(n):
                quality = float(np.clip(
                    compute_quality(scholar.research_capacity, scholar.ai_use_level),
                    0.01, 0.99))
                tier = assign_journal_tier(quality, scholar.institution_type,
                                           scholar.tier1_publications)
                # T3 avoidance: redirect to T2 if institution avoids T3 and
                # scholar is not yet desperate on overall publication count.
                if tier == 3:
                    floor = T3_FLOOR_FRACTION[scholar.institution_type]
                    if floor is not None:
                        desperate = (
                            scholar.total_publications
                            < floor * TENURE_TARGET_MIDPOINTS[scholar.institution_type]
                        )
                        if not desperate:
                            tier = 2   # submit to T2 instead of T3
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

        # ── Step 2: Per-paper RL updates ──────────────────────────────────────
        # Apply one RL update per evaluated paper (resubmissions first, then new).
        # This lets AI use self-correct: scholars producing many low-quality papers
        # accumulate net-negative signals once per-paper acceptance drops below ~9%.
        for scholar in scholars:
            # Resubmission outcomes
            for accepted in resub_outcomes[scholar.scholar_id]:
                scholar.update_ai_use('accepted' if accepted else 'rejected')
            # New paper outcomes
            for paper in period_papers:
                if paper.scholar_id == scholar.scholar_id:
                    scholar.update_ai_use('accepted' if paper.published else 'rejected')

        # ── Step 3: Collect period statistics ─────────────────────────────────
        n_produced      = len(period_papers)
        n_accepted_new  = sum(1 for p in period_papers if p.published)
        n_resub_eval    = sum(len(v) for v in resub_outcomes.values())
        n_resub_acc     = sum(sum(1 for o in v if o) for v in resub_outcomes.values())
        n_accepted_all  = n_accepted_new + n_resub_acc

        ai_uses   = [s.ai_use_level for s in scholars]
        qualities = [p.quality for p in period_papers]

        by_inst = {}
        for inst in INSTITUTION_TYPES:
            i_scholars = [s for s in scholars if s.institution_type == inst]
            i_papers   = [p for p in period_papers
                          if s_by_id[p.scholar_id].institution_type == inst]
            i_acc      = sum(1 for p in i_papers if p.published)
            by_inst[inst] = {
                'ai':     float(np.mean([s.ai_use_level for s in i_scholars])),
                'q':      float(np.mean([p.quality for p in i_papers])) if i_papers else np.nan,
                'acc':    i_acc / len(i_papers) if i_papers else np.nan,
                'n_prod': len(i_papers),
            }

        row = {
            'period':              period,
            'mean_ai_use':         float(np.mean(ai_uses)),
            'mean_quality':        float(np.mean(qualities)) if qualities else np.nan,
            'n_produced':          n_produced,
            'n_accepted_new':      n_accepted_new,
            'n_resub_evaluated':   n_resub_eval,
            'n_resub_accepted':    n_resub_acc,
            'n_accepted_all':      n_accepted_all,
            'acceptance_rate':     n_accepted_new / n_produced if n_produced else np.nan,
            'acceptance_rate_all': n_accepted_all / (n_produced + n_resub_eval)
                                   if (n_produced + n_resub_eval) else np.nan,
        }
        for inst in INSTITUTION_TYPES:
            row[f'ai_{inst}']    = by_inst[inst]['ai']
            row[f'q_{inst}']     = by_inst[inst]['q']
            row[f'acc_{inst}']   = by_inst[inst]['acc']
            row[f'nprod_{inst}'] = by_inst[inst]['n_prod']

        records.append(row)

    period_df = pd.DataFrame(records)

    # End-state per scholar
    end_rows = []
    for s in scholars:
        rec   = s.publication_record
        all_q = [r['quality'] for r in rec]
        acc_q = [r['quality'] for r in rec if r['accepted']]
        t3_acc = sum(1 for r in rec if r['tier'] == 3 and r['accepted'])
        end_rows.append({
            'scholar_id':            s.scholar_id,
            'institution_type':      s.institution_type,
            'research_capacity':     s.research_capacity,
            'ai_use_final':          s.ai_use_level,
            'total_pubs':            s.total_publications,
            'tier1_pubs':            s.tier1_publications,
            'tier2_pubs':            s.tier2_publications,
            'tier3_pubs':            t3_acc,
            'total_produced':        len(rec),
            'acceptance_rate':       s.total_publications / len(rec) if rec else 0.0,
            'mean_quality_all':      float(np.mean(all_q))  if all_q  else np.nan,
            'mean_quality_accepted': float(np.mean(acc_q))  if acc_q  else np.nan,
        })

    return period_df, pd.DataFrame(end_rows)


# ── Run all replications ──────────────────────────────────────────────────────

def run_simulation():
    all_period, all_end = [], []
    print(f"Running {N_REPLICATIONS} replications x {N_PERIODS} periods x {N_SCHOLARS} scholars ...")
    for rep in range(N_REPLICATIONS):
        pdf, edf = run_one_replication(RANDOM_SEED + rep)
        pdf['replication'] = rep
        edf['replication'] = rep
        all_period.append(pdf)
        all_end.append(edf)
        if (rep + 1) % 10 == 0:
            print(f"  Completed {rep+1}/{N_REPLICATIONS}")
    period_df = pd.concat(all_period, ignore_index=True)
    end_df    = pd.concat(all_end,    ignore_index=True)
    period_df.to_csv('outputs/sim_results.csv',      index=False)
    end_df.to_csv(   'outputs/scholar_endstate.csv', index=False)
    print(f"Saved outputs/sim_results.csv  ({period_df.shape})")
    print(f"Saved outputs/scholar_endstate.csv  ({end_df.shape})")
    return period_df, end_df


# ── Summarize helper ──────────────────────────────────────────────────────────

def ci(df, col):
    """Return mean, lower CI, upper CI across replications by period."""
    g  = df.groupby('period')[col]
    m  = g.mean()
    se = g.sem()
    return m, m - 1.96*se, m + 1.96*se


# ── Quick diagnostic plots ────────────────────────────────────────────────────

def make_plots(period_df, end_df):
    periods = sorted(period_df['period'].unique())   # now includes 0
    inst_labels = {'R1':'R1','R2':'R2','Balanced':'Balanced','Teaching':'Teaching-Focused'}

    plt.rcParams.update({'font.family':'sans-serif','axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':150})

    # AI use by institution (includes period 0)
    fig, ax = plt.subplots(figsize=(8,4.5))
    for inst in INSTITUTION_TYPES:
        m,lo,hi = ci(period_df, f'ai_{inst}')
        c = INST_COLORS[inst]
        ax.plot(periods, m, color=c, lw=2.2, label=inst_labels[inst])
        ax.fill_between(periods, lo, hi, color=c, alpha=0.15)
    ax.axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_title('AI Use Level Over Time by Institution Type', fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax.set_xlabel('Period (0 = initial state; 1–12 = post-RL)');
    ax.set_ylabel('Mean AI Use Level')
    ax.set_ylim(0, 1); ax.set_xlim(0, N_PERIODS)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('outputs/fig_ai_use.png', bbox_inches='tight')
    plt.close()
    print("Saved outputs/fig_ai_use.png")

    # Tenure progress
    fig, ax = plt.subplots(figsize=(7,4.5))
    inst_pubs = end_df.groupby('institution_type')['total_pubs']
    means  = inst_pubs.mean().reindex(INSTITUTION_TYPES)
    sems   = inst_pubs.sem().reindex(INSTITUTION_TYPES)
    colors = [INST_COLORS[i] for i in INSTITUTION_TYPES]
    x      = range(len(INSTITUTION_TYPES))
    ax.bar(x, means, yerr=1.96*sems, color=colors, edgecolor='white',
           capsize=5, error_kw={'linewidth':1.5})
    for xi, inst in enumerate(INSTITUTION_TYPES):
        mid = TENURE_TARGET_MIDPOINTS[inst]
        ax.plot([xi-0.4, xi+0.4], [mid, mid], color='black', lw=2, ls='--')
    ax.set_xticks(list(x))
    ax.set_xticklabels([inst_labels[i] for i in INSTITUTION_TYPES])
    ax.set_title('Mean Total Publications by Institution Type\n(dashed = tenure target midpoint)',
                 fontweight='bold', color=COLORS['primary'], pad=10)
    ax.set_ylabel('Mean Publications')
    plt.tight_layout()
    plt.savefig('outputs/fig_tenure_progress.png', bbox_inches='tight')
    plt.close()
    print("Saved outputs/fig_tenure_progress.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    period_df, end_df = run_simulation()
    make_plots(period_df, end_df)

    print("\n=== SUMMARY STATISTICS ===")
    print("\nMean initial AI use by institution (period 0):")
    p0 = period_df[period_df['period'] == 0]
    for inst in INSTITUTION_TYPES:
        vals = p0[f'ai_{inst}']
        print(f"  {inst:12s}: M={vals.mean():.3f}")

    print("\nMean final AI use by institution (period 12):")
    for inst in INSTITUTION_TYPES:
        vals = end_df[end_df['institution_type']==inst]['ai_use_final']
        print(f"  {inst:12s}: M={vals.mean():.3f}  SD={vals.std():.3f}")

    print("\nMean total publications by institution (incl. resubmission wins):")
    for inst in INSTITUTION_TYPES:
        vals = end_df[end_df['institution_type']==inst]['total_pubs']
        tgt  = TENURE_TARGET_MIDPOINTS[inst]
        print(f"  {inst:12s}: M={vals.mean():.1f}  (target midpoint={tgt})")

    print("\nMean tier breakdown by institution:")
    for inst in INSTITUTION_TYPES:
        sub = end_df[end_df['institution_type']==inst]
        print(f"  {inst:12s}: T1={sub['tier1_pubs'].mean():.1f}  "
              f"T2={sub['tier2_pubs'].mean():.1f}  "
              f"T3={sub['tier3_pubs'].mean():.1f}")

    print("\nOverall acceptance rate — new papers (all periods):")
    acc = period_df[period_df['period'] > 0]['acceptance_rate'].dropna()
    print(f"  M={acc.mean()*100:.1f}%  range=[{acc.min()*100:.1f}%, {acc.max()*100:.1f}%]")

    print("\nResubmission volume (mean per period, across replications):")
    rsub = period_df[period_df['period'] > 0]
    print(f"  Evaluated: {rsub['n_resub_evaluated'].mean():.1f}  "
          f"Accepted: {rsub['n_resub_accepted'].mean():.1f}")

    print("\nDone.")
