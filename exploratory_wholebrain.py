'''Exploratory Whole Brain Analysis'''

# Imports

import os
from pathlib import Path
import warnings
import pytensor.tensor as pt
import pytensor as pytensor

# Data
import numpy as np
import pandas as pd

# Bayes
import pymc as pm
import arviz as az

# Statistics
import pingouin as pg
import scipy.stats as stats
from scipy.stats import pearsonr, ttest_ind, ttest_rel, sem, norm, gaussian_kde, zscore
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# Neuroimaging & Networks
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

print('All imports loaded successfully.')

# Paths and Constants

# 1. Define Paths 
base_dir = Path("path/to/data/")
roi_folder = base_dir / "all_roi_data"
fig_out_dir = Path("path/to/figs/thesis_figs")

# 2. Experiment Constants
subjects = []
sessions = ["ses-01", "ses-02"]

n_subjects = len(subjects) 
n_sessions = len(sessions)

# 3. Group Assignments
hv_subjects = []
lv_subjects = [s for s in subjects if s not in hv_subjects]

# 4. Subject Exclusion Sets
motion_outliers = [] 
no_physio = []

analysis_sets = {
    'full_sample': subjects,
    'no_outlier':  [s for s in subjects if s not in motion_outliers],
    'strict':      [s for s in subjects if s not in motion_outliers and s not in no_physio],
}

# 5. Set Active Subject List
# Change this variable to swap between exclusion sets
active_subjects = analysis_sets['no_outlier']

# 6. Print Summary
print("--- Subject Group Summaries ---")
for name, subs in analysis_sets.items():
    n_lv = len([s for s in subs if s in lv_subjects])
    n_hv = len([s for s in subs if s in hv_subjects])
    print(f'{name}: N={len(subs)} (LV={n_lv}, HV={n_hv})')


# Data Loading for WHOLE-BRAIN HCP
# Each HCP parcel is its own column
import warnings

SKIP_LABELS = {'rh_HCP_???', 'lh_HCP_???'}

def load_hcp_timeseries(subject, session, roi_folder):
    """Load denoised timeseries CSV and return only the HCP columns."""
    filename = f"{subject}_{session}_denoised_timeseries.csv"
    file_path = roi_folder / filename
    if not file_path.exists():
        warnings.warn(f"File not found: {file_path}")
        return None
    raw_df = pd.read_csv(file_path)
    hcp_columns = [c for c in raw_df.columns if "HCP" in c and c not in SKIP_LABELS]
    if not hcp_columns:
        raise ValueError(f"No HCP columns found in {file_path}")
    return raw_df[hcp_columns]

print("Loading HCP time series...")
all_data = {}
missing_data_flags = []

for sub in active_subjects:
    all_data[sub] = {}
    for ses in sessions:
        try:
            df = load_hcp_timeseries(sub, ses, roi_folder)
            if df is not None:
                all_data[sub][ses] = df
            else:
                missing_data_flags.append((sub, ses))
        except Exception as e:
            print(f"Error processing {sub} {ses}: {e}")
            missing_data_flags.append((sub, ses))

complete = [s for s in all_data if all(ses in all_data[s] for ses in sessions)]
print(f"Loaded data for {len(all_data)} subjects; "
      f"{len(complete)} have both sessions.")
if missing_data_flags:
    print(f"  Warning: missing/failed for {len(missing_data_flags)} sub/ses.")

# quick check
if complete:
    n_hcp = all_data[complete[0]][sessions[0]].shape[1]
    print(f"  HCP columns per subject: {n_hcp}  (expect ~360)")


# =========================================================
# WHOLE-CONNECTOME EXPLORATORY  -  STAGE 1 (parametric + FDR)
# ---------------------------------------------------------
# For every HCP edge:  Delta z = post - pre  (per subject)
# Then Welch two-sample t-test on Delta between HV and LV groups
# (= the group x session interaction for a balanced 2x2),
# vectorized across all ~64,620 edges, with BH-FDR correction.
#
# Reuses: all_data[sub][ses] -> denoised timeseries DataFrame
#         active_subjects, sessions, hv_subjects, lv_subjects
# =========================================================

PRE, POST = sessions[0], sessions[1]

# ---------------------------------------------------------
# 1. Identify the HCP ROI columns 
# ---------------------------------------------------------
# probe one subject/session to get column names, keep only HCP columns
_probe = all_data[active_subjects[0]][PRE]
hcp_cols = [c for c in _probe.columns if "HCP" in c]
hcp_cols = sorted(hcp_cols)
n_roi = len(hcp_cols)
edge_idx = list(combinations(range(n_roi), 2))      # upper-triangle edge list
n_edge = len(edge_idx)
print(f"HCP ROIs: {n_roi}  ->  edges: {n_edge}")

# precompute upper-triangle index arrays once
iu = np.triu_indices(n_roi, k=1)

def fisher_z_edges(df):
    """Return the upper-triangle Fisher-z FC vector (length n_edge) for one df."""
    X = df[hcp_cols].values                  # (timepoints, n_roi)
    C = np.corrcoef(X, rowvar=False)         # (n_roi, n_roi)
    r = np.clip(C[iu], -0.9999, 0.9999)
    return np.arctanh(r)                      # length n_edge

# ---------------------------------------------------------
# 2. Build the Delta matrix:  rows = subjects, cols = edges
#    Delta = post - pre  (per subject, per edge)
# ---------------------------------------------------------
hv_deltas, lv_deltas, used_hv, used_lv = [], [], [], []

for sub in active_subjects:
    if sub not in all_data:
        continue
    if PRE not in all_data[sub] or POST not in all_data[sub]:
        continue
    z_pre  = fisher_z_edges(all_data[sub][PRE])
    z_post = fisher_z_edges(all_data[sub][POST])
    delta  = z_post - z_pre                  # length n_edge

    if sub in hv_subjects:
        hv_deltas.append(delta); used_hv.append(sub)
    elif sub in lv_subjects:
        lv_deltas.append(delta); used_lv.append(sub)

HV = np.vstack(hv_deltas)      # (n_hv, n_edge)
LV = np.vstack(lv_deltas)      # (n_lv, n_edge)
n_hv, n_lv = HV.shape[0], LV.shape[0]
print(f"HV subjects: {n_hv}   LV subjects: {n_lv}   (edges: {HV.shape[1]})")

# ---------------------------------------------------------
# 3. VECTORIZED Welch t-test across ALL edges at once
#    Operate down columns (axis=0): each column is one edge.
# ---------------------------------------------------------
mean_hv = HV.mean(axis=0)                 # (n_edge,)  HV mean delta per edge
mean_lv = LV.mean(axis=0)
var_hv  = HV.var(axis=0, ddof=1)          # sample variance per edge
var_lv  = LV.var(axis=0, ddof=1)

# Welch standard error and t (elementwise over all edges)
se = np.sqrt(var_hv / n_hv + var_lv / n_lv)
with np.errstate(invalid="ignore", divide="ignore"):
    t = (mean_hv - mean_lv) / se

# Welch-Satterthwaite degrees of freedom (per edge, vectorized)
with np.errstate(invalid="ignore", divide="ignore"):
    df = (var_hv/n_hv + var_lv/n_lv)**2 / (
        (var_hv/n_hv)**2/(n_hv-1) + (var_lv/n_lv)**2/(n_lv-1))

# two-sided p-values for all edges at once
p = 2 * stats.t.sf(np.abs(t), df)

# guard against any degenerate edges (zero variance in both groups)
bad = ~np.isfinite(t) | ~np.isfinite(p)
p[bad] = 1.0
t[bad] = 0.0

# ---------------------------------------------------------
# 4. FDR (Benjamini-Hochberg) across all edges
# ---------------------------------------------------------
reject, p_fdr, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")

# Cohen's d on the deltas (pooled SD), per edge, for effect-size readout
sp = np.sqrt(((n_hv-1)*var_hv + (n_lv-1)*var_lv) / (n_hv + n_lv - 2))
with np.errstate(invalid="ignore", divide="ignore"):
    d = (mean_hv - mean_lv) / sp

# ---------------------------------------------------------
# 5. Assemble results table (one row per edge) and summarize
# ---------------------------------------------------------
roi_a = [hcp_cols[i] for i, j in edge_idx]
roi_b = [hcp_cols[j] for i, j in edge_idx]
res = pd.DataFrame({
    "roi_a": roi_a, "roi_b": roi_b,
    "mean_d_HV": mean_hv, "mean_d_LV": mean_lv,
    "HV_minus_LV": mean_hv - mean_lv,
    "cohen_d": d, "t": t, "p_unc": p, "p_FDR": p_fdr,
})

n_unc = int((res["p_unc"] < 0.05).sum())
n_fdr = int((res["p_FDR"] < 0.05).sum())
print(f"\nEdges p_unc < .05 : {n_unc}  ({100*n_unc/n_edge:.2f}%)   "
      f"[~{0.05*n_edge:.0f} expected by chance if null]")
print(f"Edges surviving FDR q<.05 : {n_fdr}")

if n_fdr > 0:
    print("\nFDR-surviving edges:")
    print(res[res["p_FDR"] < 0.05]
          .sort_values("p_FDR")
          .round(4).to_string(index=False))
else:
    print("\nNo edge survives FDR. (Showing 5 smallest p_unc for inspection:)")
    print(res.nsmallest(5, "p_unc").round(4).to_string(index=False))

res.to_csv("path/to/data/whole_connectome_stage1.csv", index=False)
print("\nSaved full edge table -> whole_connectome_stage1.csv")

# =========================================================
# WHOLE-CONNECTOME EXPLORATORY - SESSION (pre->post) MAIN EFFECT
# Justified BECAUSE the interaction was null: no evidence HV/LV
# change differently, so pool all subjects for the common change.
# One-sample (paired) t-test of Delta z = post - pre vs 0, per edge.
# =========================================================
DELTA = np.vstack([HV, LV])              # (38, n_edge), reuse Stage-1 deltas
n_sub = DELTA.shape[0]
print(f"Pooled subjects: {n_sub}  edges: {DELTA.shape[1]}")

mean_d = DELTA.mean(axis=0)
sd_d   = DELTA.std(axis=0, ddof=1)
se_d   = sd_d / np.sqrt(n_sub)
with np.errstate(invalid="ignore", divide="ignore"):
    t_sess = mean_d / se_d
    dz     = mean_d / sd_d                # within-subject Cohen's dz
df_sess = n_sub - 1
p_sess  = 2 * stats.t.sf(np.abs(t_sess), df_sess)

bad = ~np.isfinite(t_sess) | ~np.isfinite(p_sess)
p_sess[bad] = 1.0; t_sess[bad] = 0.0

reject_s, p_fdr_s, _, _ = multipletests(p_sess, alpha=0.05, method="fdr_bh")

res_sess = pd.DataFrame({
    "roi_a": [hcp_cols[i] for i, j in edge_idx],
    "roi_b": [hcp_cols[j] for i, j in edge_idx],
    "mean_delta_z": mean_d, "cohen_dz": dz,
    "t": t_sess, "p_unc": p_sess, "p_FDR": p_fdr_s,
})

n_unc = int((res_sess["p_unc"] < 0.05).sum())
n_fdr = int((res_sess["p_FDR"] < 0.05).sum())
print(f"Edges p_unc < .05 : {n_unc}  ({100*n_unc/n_edge:.2f}%)  "
      f"[~{0.05*n_edge:.0f} expected if null]")
print(f"Edges surviving FDR q<.05 : {n_fdr}")

if n_fdr > 0:
    surv = res_sess[res_sess["p_FDR"] < 0.05]
    print(f"  increases: {(surv['mean_delta_z']>0).sum()}   "
          f"decreases: {(surv['mean_delta_z']<0).sum()}")
    print(surv.sort_values("p_FDR").head(10).round(4).to_string(index=False))
else:
    print("\nNo edge survives FDR. 5 smallest p_unc:")
    print(res_sess.nsmallest(5, "p_unc").round(4).to_string(index=False))

res_sess.to_csv("path/to/data/whole_connectome_session.csv", index=False)
print("\nSaved session effect table -> whole_connectome_session.csv")

# =========================================================
# NETWORK-BASED STATISTIC (NBS, Zalesky et al. 2010)
# - Interaction : permute HV/LV group labels on Delta z
# - Session     : flip within-subject signs of Delta z
# Statistic per component = extent (number of edges).
# Null = distribution of the largest component extent over permutations.
# =========================================================

rng = np.random.default_rng(42)
N_PERM = 10000
PRIMARY_P = [0.05, 0.01, 0.005, 0.001]          # primary edge thresholds (sensitivity)

# --- data: rows = subjects, cols = edges (reuse Stage-1 arrays) ----------
DELTA = np.vstack([HV, LV])                # (38, n_edge): first n_hv=HV, then LV
n_sub = DELTA.shape[0]
n_hv, n_lv = HV.shape[0], LV.shape[0]
group_mask0 = np.zeros(n_sub, dtype=bool); group_mask0[:n_hv] = True  # True = HV, boolean vector for group assignment

edge_i = np.fromiter((i for i, j in edge_idx), dtype=np.int64, count=n_edge)
edge_j = np.fromiter((j for i, j in edge_idx), dtype=np.int64, count=n_edge)

# --- per-edge test statistics (same formula for observed and permuted) ---
def t_interaction(mask):
    A, B = DELTA[mask], DELTA[~mask]
    nA, nB = A.shape[0], B.shape[0]
    vA, vB = A.var(0, ddof=1), B.var(0, ddof=1)
    sp = np.sqrt(((nA-1)*vA + (nB-1)*vB) / (nA+nB-2))
    se = sp * np.sqrt(1/nA + 1/nB)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num((A.mean(0) - B.mean(0)) / se)

def t_session(signs):
    D = DELTA * signs[:, None]
    se = D.std(0, ddof=1) / np.sqrt(n_sub)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(D.mean(0) / se)

# --- largest connected-component extent among suprathreshold edges -------
def largest_extent(tvec, thr):
    supra = np.abs(tvec) > thr
    if not supra.any():
        return 0, np.array([]), supra
    ai, bj = edge_i[supra], edge_j[supra]
    d = np.ones(ai.size)
    A = coo_matrix((np.concatenate([d, d]),
                    (np.concatenate([ai, bj]), np.concatenate([bj, ai]))),
                   shape=(n_roi, n_roi)).tocsr()
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab[ai])           # edges per component
    return sizes.max(), sizes, supra

# --- generic NBS runner --------------------------------------------------
def run_nbs(stat_fn, perm_draw, label):
    df_t = (n_hv + n_lv - 2) if label == "interaction" else (n_sub - 1)
    obs_t = stat_fn(group_mask0 if label == "interaction"
                    else np.ones(n_sub))
    print(f"\n=== NBS: {label} ({N_PERM} permutations) ===")
    for p_thr in PRIMARY_P:
        thr = stats.t.ppf(1 - p_thr/2, df_t)          # two-sided edge threshold
        obs_max, obs_sizes, supra = largest_extent(obs_t, thr)
        null_max = np.empty(N_PERM)
        for k in range(N_PERM):
            null_max[k] = largest_extent(stat_fn(perm_draw()), thr)[0]
        # p for the observed largest component vs the max-component null
        p_comp = (1 + np.sum(null_max >= obs_max)) / (1 + N_PERM)
        n_supra = int(supra.sum())
        print(f"  thr p<{p_thr:<6} (|t|>{thr:.2f}): "
              f"{n_supra:5d} supra-edges | largest comp = {obs_max:5d} edges | "
              f"p = {p_comp:.4f}"
              + ("  *" if p_comp < 0.05 else ""))
    return obs_t

# --- run both contrasts --------------------------------------------------
_ = run_nbs(
    t_interaction,
    lambda: rng.permutation(group_mask0),      # shuffle group labels
    "interaction",
)
_ = run_nbs(
    t_session,
    lambda: rng.choice([-1.0, 1.0], size=n_sub),  # random sign flips
    "session",
)
print("\nNBS complete. A component is significant if p < 0.05 at a given threshold.")