"""
Full preprocessing pipeline.

Ordered steps — do not reorder:
  1.  Load + clean + prune each dataset
  2.  Harmonize column names
  3.  Normalize per-user (StudentLife and LifeSnaps only; NHANES skipped)
  4.  Save preprocessed outputs to datasets/data_preprocessed/
  5.  Discretize each source column (clinical thresholds / silhouette k-means / passthrough),
      then derive node states via k-modes clustering on the discretized column vectors;
      updates feature_node_config.json with kmeans bin_edges
  5b. Build column-question map -> configs/column_question_map.csv
  5c. Learn P(col_val | node_state) likelihood tables -> feature_node_config.json
  6.  Compute NHANES CPT priors -> write to feature_node_config.json
  7.  Build training_final.csv (concat StudentLife + LifeSnaps)
"""

import os
import pandas as pd

from data_cleaning_pruning_nhanes      import load_nhanes, clean_nhanes, prune_nhanes
from data_cleaning_pruning_studentlife import load_studentlife, clean_studentlife, prune_studentlife
from data_cleaning_pruning_lifesnaps   import load_lifesnaps, clean_lifesnaps, prune_lifesnaps
from data_harmonization                import harmonize_nhanes, harmonize_studentlife, harmonize_lifesnaps
from data_normalization                import normalize_per_user
from data_discretization               import run_discretization, clear_kmeans_config
from export_population_norm_stats      import export_population_norm_stats
from build_column_question_map         import build_map
from data_likelihood_tables            import build_likelihoods
from data_nhanes_priors                import build_priors
from data_training_set                 import build_training_set

OUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'datasets', 'data_preprocessed'
)


def _save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f'Saved {len(df)} rows, {len(df.columns)} cols -> {path}')


def run_nhanes() -> pd.DataFrame:
    print('\n=== NHANES ===')
    df = load_nhanes()
    df = clean_nhanes(df)
    df = prune_nhanes(df)
    df = harmonize_nhanes(df)
    _save(df, 'nhanes_preprocessed.csv')
    return df


def run_studentlife() -> pd.DataFrame:
    """Load, clean, prune, and harmonize StudentLife (no normalize — called from run_all)."""
    print('\n=== StudentLife ===')
    df = load_studentlife()
    df = clean_studentlife(df)
    df = prune_studentlife(df)
    df = harmonize_studentlife(df)
    return df


def run_lifesnaps() -> pd.DataFrame:
    """Load, clean, prune, and harmonize LifeSnaps (no normalize — called from run_all)."""
    print('\n=== LifeSnaps ===')
    df = load_lifesnaps()
    df = clean_lifesnaps(df)
    df = prune_lifesnaps(df)
    df = harmonize_lifesnaps(df)
    return df


def run_all() -> tuple:
    nhanes_df = run_nhanes()
    sl_df     = run_studentlife()
    ls_df     = run_lifesnaps()

    # ── Export population norm stats ────────────────────────────────
    # Must run on the pooled harmonized data (before normalization) so that
    # population mean/std reflect raw sensor distributions across all users.
    # normalize_per_user (below) will load and apply these stats.
    print('\n=== Export Population Norm Stats ===')
    combined_harmonized = pd.concat([nhanes_df, sl_df, ls_df], ignore_index=True)
    export_population_norm_stats(combined_harmonized)

    # ── Normalize (population z-score for sensor cols, per-user for rest)
    print('\n=== Normalize ===')
    sl_df = normalize_per_user(sl_df)
    _save(sl_df, 'studentlife_preprocessed.csv')
    ls_df = normalize_per_user(ls_df)
    _save(ls_df, 'lifesnaps_preprocessed.csv')

    print('\n=== Step 5: Discretization ===')
    clear_kmeans_config()
    run_discretization()
    print('\n=== Step 5b: Column-Question Map ===')
    build_map()
    print('\n=== Step 5c: Likelihood Tables ===')
    build_likelihoods()
    print('\n=== Step 6: NHANES Priors ===')
    build_priors()
    print('\n=== Step 7: Training Set ===')
    build_training_set()
    print('\nPipeline steps 1-7 complete.')
    return nhanes_df, sl_df, ls_df


if __name__ == '__main__':
    run_all()
