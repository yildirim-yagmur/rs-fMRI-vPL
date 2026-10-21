"""Aim 1: Bayesian Hierarchical Modeling of Visual-Parietal Functional Connectivity"""

# %%
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


# Neuroimaging & Networks
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

print('All imports loaded successfully.')

# %%
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

# %%
# Behavioral Data
df_li = pd.read_csv("path/to/data/learning_index.csv")
# Z-score the Learning Index
df_li['LI_z'] = stats.zscore(df_li['LI'])

# %%
# ROI Definitions & Network Pathways 

# 1. Base ROI Dictionary (Mapping custom names to exact CSV headers)
roi_defs = {
    # --- Localizers: Trained ---
    'v1d_training':  ['lh.v1d_trai.circle3.5'],
    'v2d_training':  ['lh.v2d_trai.circle3.5'],
    'hv4_training':  ['lh.hv4_trai.circle3.5'],
    
    # --- Localizers: Transfer (Untrained) ---
    'v1v_transfer':  ['lh.v1v_trns.circle3.5'],
    'v2v_transfer':  ['lh.v2v_trns.circle3.5'],
    'hv4_transfer':  ['lh.hv4_trns.circle3.5'],  
    
    # --- Wang Atlas: Visual ROIs ---
    'wang_v1d':      ['lh.wang15atlas.V1d'],        
    'wang_v1v':      ['lh.wang15atlas.V1v'],        
    'wang_v2v':      ['lh.wang15atlas.V2v'],        
    'wang_v2d':      ['lh.wang15atlas.V2d'],        
    'wang_v3v':      ['lh.wang15atlas.V3v'],      
    'wang_v3d':      ['lh.wang15atlas.V3d'],        
    'wang_hv4':      ['lh.wang15atlas.hV4'], 
    
    # --- Shared Across Runs: Wang Atlas IPS ROIs ---
    'IPS0':          ['lh.wang15atlas.IPS0'],
    'IPS1-3':        ['lh.wang15atlas.IPS1', 'lh.wang15atlas.IPS2', 'lh.wang15atlas.IPS3'],
    'IPS4-5':        ['lh.wang15atlas.IPS4', 'lh.wang15atlas.IPS5'],

    # hv4 transfer and training combined
    'hv4_combined':  ['lh.hv4_trai.circle3.5', 'lh.hv4_trns.circle3.5'],
}

# 2. Define the Nodes for the Three Statistical Runs
run1_training_nodes = [
    'v1d_training', 'v2d_training', 'hv4_training',
    'IPS0', 'IPS1-3', 'IPS4-5',
]

run2_wang_nodes = [
    'wang_v1d', 'wang_v1v', 'wang_v2d', 'wang_v2v', 
    'wang_v3d', 'wang_v3v', 'wang_hv4', 
    'IPS0', 'IPS1-3', 'IPS4-5'
]

run3_transfer_nodes = [
    'v1v_transfer', 'v2v_transfer', 'hv4_transfer',
    'IPS0', 'IPS1-3', 'IPS4-5'
]

run4_all_localizers_nodes = ['v1d_training', 'v2d_training', 'hv4_training',
    'IPS0', 'IPS1-3', 'IPS4-5', 'v1v_transfer', 'v2v_transfer', 'hv4_transfer', 'wang_hv4']

# 3. Define the Edges (Pathways) to test for each run
run1_training_pathways = [
    ('v2d_training', 'IPS0'), ('v2d_training', 'IPS1-3'), ('v2d_training', 'IPS4-5'),
    ('hv4_training', 'IPS0'), ('hv4_training', 'IPS1-3'), ('hv4_training', 'IPS4-5'),
    ('v2d_training', 'hv4_training'),
    ('v1d_training', 'IPS0'), ('v1d_training', 'IPS1-3'), ('v1d_training', 'IPS4-5'),
    ('v1d_training', 'v2d_training'), ('v1d_training', 'hv4_training'), 
]

run2_wang_pathways = [
    ('wang_v2d',  'IPS0'), ('wang_v2d',  'IPS1-3'), ('wang_v2d',  'IPS4-5'),
    ('wang_hv4',  'IPS0'), ('wang_hv4',  'IPS1-3'), ('wang_hv4',  'IPS4-5'),
    ('wang_v2d',  'wang_hv4'), ('wang_v1d','wang_v3d' ), ('wang_v1v','wang_v3d'),
    ('wang_v1v','wang_v2d' ), ('wang_v2d','wang_v3v' ), ('wang_v3v', 'wang_hv4'),
    ('wang_v1d', 'IPS0'), ('wang_v1d', 'IPS1-3'), ('wang_v1d', 'IPS4-5'),
    ('wang_v1d', 'wang_v2d'), ('wang_v1d', 'wang_hv4')
]

run3_transfer_pathways = [
    ('v2v_transfer', 'IPS0'), ('v2v_transfer', 'IPS1-3'), ('v2v_transfer', 'IPS4-5'),
    ('hv4_transfer', 'IPS0'), ('hv4_transfer', 'IPS1-3'), ('hv4_transfer', 'IPS4-5'),
    ('v2v_transfer', 'hv4_transfer'),('v1v_transfer', 'IPS0'), ('v1v_transfer', 'IPS1-3'), 
    ('v1v_transfer', 'IPS4-5'),('v1v_transfer', 'v2v_transfer'), ('v1v_transfer', 'hv4_transfer'),
]

print(f"Total defined custom ROIs: {len(roi_defs)}")
print(f"Run 1 (Training) nodes:  {len(run1_training_nodes)} | Edges: {len(run1_training_pathways)}")
print(f"Run 2 (Wang Only) nodes: {len(run2_wang_nodes)} | Edges: {len(run2_wang_pathways)}")
print(f"Run 3 (Transfer) nodes:  {len(run3_transfer_nodes)} | Edges: {len(run3_transfer_pathways)}")

# %%
# Data Loading and ROI Averaging

def load_and_prep_timeseries(subject, session, roi_folder, roi_dict):
    """
    Loads raw CSV, extracts defined ROIs, averages grouped ROIs, 
    and returns a clean dataframe.
    """
    filename = f"{subject}_{session}_denoised_timeseries.csv" 
    file_path = roi_folder / filename
    
    if not file_path.exists():
        warnings.warn(f"File not found: {file_path}")
        return None

    # Load the raw CSV
    raw_df = pd.read_csv(file_path)
    
    # Initialize an empty dataframe for the clean, averaged data
    clean_df = pd.DataFrame()
    
    for short_name, raw_columns in roi_dict.items():
        # Safety check: ensure the columns actually exist in the CSV
        missing = [col for col in raw_columns if col not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} for subject {subject} {session}")
            
        # Extract the columns
        extracted_data = raw_df[raw_columns]
        
        # If there are multiple columns (like IPS1-3), average them across the row (axis=1)
        if len(raw_columns) > 1:
            clean_df[short_name] = extracted_data.mean(axis=1)
        # If it's a single column, just copy it over
        else:
            clean_df[short_name] = extracted_data.iloc[:, 0]
            
    return clean_df


# Execute Loading Loop
print("Loading and preparing ROI time series...")

# Everything is stored in a nested dictionary: all_data[subject][session] = dataframe
all_data = {}
missing_data_flags = []

for sub in active_subjects:
    all_data[sub] = {}
    for ses in sessions:
        try:
            df = load_and_prep_timeseries(sub, ses, roi_folder, roi_defs)
            if df is not None:
                all_data[sub][ses] = df
            else:
                missing_data_flags.append((sub, ses))
        except Exception as e:
            print(f"Error processing {sub} {ses}: {e}")
            missing_data_flags.append((sub, ses))

print(f"Successfully loaded data for {len(all_data)} subjects.")
if missing_data_flags:
    print(f" Warning: Missing or failed data for {len(missing_data_flags)} sub/ses combinations.")

# %%
# Compute Functional Connectivity & Fisher Z-Transform

# Dictionary to hold the long-format DataFrames for statistics
stats_dfs = {}

# Bundle the run definitions to iterate over them
runs = {
    'Run1_Training': (run1_training_nodes, run1_training_pathways),
    'Run2_Wang':     (run2_wang_nodes, run2_wang_pathways),
    'Run3_Transfer': (run3_transfer_nodes, run3_transfer_pathways),
    'Run4_All_Localizers': (run4_all_localizers_nodes, run4_all_localizers_pathways)
}

print("Calculating Pearson r and applying Fisher Z-transformation...")

for run_name, (nodes, pathways) in runs.items():
    run_records = []
    
    for sub in active_subjects:
        group = 'HV' if sub in hv_subjects else 'LV'
        
        for ses in sessions:
            # 1. Get the preprocessed time series for these specific nodes
            df_ts = all_data[sub][ses][nodes]
            
            # 2. Compute Pearson correlation matrix 
            corr_matrix = df_ts.corr(method='pearson')
            
            # 3. Extract the hypothesized edges
            for node1, node2 in pathways:
                # Extract raw r-value using pandas label indexing
                r_val = corr_matrix.loc[node1, node2]
                
                # Apply Fisher Z-transformation: z = arctanh(r)
                # Clip r slightly to avoid infinity errors if r=1.0
                r_clipped = np.clip(r_val, -0.9999, 0.9999) 
                z_val = np.arctanh(r_clipped)
                
                # Store the record
                run_records.append({
                    'Subject': sub,
                    'Group': group,
                    'Session': ses,
                    'Pathway': f"{node1} <-> {node2}",
                    'r_value': r_val,
                    'z_value': z_val
                })
                
    # Convert list of records to a DataFrame
    stats_dfs[run_name] = pd.DataFrame(run_records)

print("\n--- Processing Complete ---")
for run_name, df in stats_dfs.items():
    n_edges = len(runs[run_name][1])
    n_obs = len(df)
    print(f"{run_name}: extracted {n_edges} specific edges. Total observations: {n_obs}")

# %%
# Master long-format table = the modeling granularity
# One row = one pathway × subject × session

from IPython.display import display

# Stack the three run-level long DFs into one, keeping a Run column
master_long = (
    pd.concat(stats_dfs, names=["Run", "_row"])
      .reset_index(level="Run")
      .reset_index(drop=True)
)


# Readable pre/post label (model still uses Session categorical below)
master_long["Session_label"] = master_long["Session"].map(
    {"ses-01": "pre", "ses-02": "post"}
)

# Explicit reference levels so contrasts are unambiguous at model time
master_long["Group"]   = pd.Categorical(master_long["Group"],
                                         categories=["LV", "HV"])         
master_long["Session"] = pd.Categorical(master_long["Session"],
                                         categories=["ses-01", "ses-02"]) 

master_long = master_long.merge(df_li[['Subject', 'LI_z']], on='Subject', how='left') 

master_long = master_long[
    ["Run", "Subject", "Group", "Session", "Session_label",
    "Pathway", "r_value", "z_value", "LI_z"]
]

display(master_long.head(12))

# Sanity: rows should equal n_subjects × n_sessions × total_edges
n_edges_total = sum(len(p) for _, p in runs.values())
expected = len(active_subjects) * len(sessions) * n_edges_total
print(f"Rows expected: {expected} | actual: {len(master_long)} | "
      f"{'OK' if expected == len(master_long) else 'MISMATCH — check missing scans'}")

# %%
# Set the analysis run group
analysis_run = "Run1_Training"  # Change this to Run2_Wang or Run3_Transfer as needed

# %%
# Null effect model for comparison: no session effects, only pathway and group intercepts and random effects
#
# z ~ Pathway + Group + (1 | Subject) + (1 | Subject:Pathway)

# Build the design for one statistical run, explicitly
df = master_long[master_long["Run"] == analysis_run].copy()

# 1. Create dictionaries for sum/effect coding
group_map = {"HV": 0.5, "LV": -0.5}
session_map = {"ses-02": 0.5, "ses-01": -0.5}

df["group_ind"] = df["Group"].map(group_map)
subj_idx, subjects_u = pd.factorize(df["Subject"])
path_idx, paths_u = pd.factorize(df["Pathway"])

sp_key = df["Subject"].astype(str) + "__" + df["Pathway"].astype(str)
sp_idx, sp_u = pd.factorize(sp_key)

coords = {
    "subject": subjects_u,
    "pathway": paths_u,
    "subj_pathway": sp_u,
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as aim1_null:

    path_idx_d = pm.Data("path_idx", path_idx, dims="obs_id")
    subj_idx_d = pm.Data("subj_idx", subj_idx, dims="obs_id")
    sp_idx_d = pm.Data("sp_idx", sp_idx, dims="obs_id")
    group_d = pm.Data("group", df["group_ind"].values.astype(float), dims="obs_id")
    z_obs = pm.Data("z_obs", df["z_value"].values, dims="obs_id")

    # ---- fixed effects ----
    # Global baseline connectivity per pathway
    beta_pathway = pm.Normal("beta_pathway", mu=0.3, sigma=0.3, dims="pathway")

    # Marginal Main Effect of Group
    beta_group = pm.Normal("beta_group", mu=0.0, sigma=0.10)
    b_g_p = pm.ZeroSumNormal("b_group_x_pathway", sigma=0.10, dims="pathway")
    # the "Null Model" includes the stable, trait-level grouping factors

    # ---- (1 | Subject) ----
    # Because there is no Session effect, there is no slope.
    # We use a simple random intercept for the subjects.
    sigma_subj = pm.HalfNormal("sigma_subj", 0.3)
    z_subj = pm.Normal("z_subj", 0, 1, dims="subject")
    u_int = pm.Deterministic("u_int", z_subj * sigma_subj, dims="subject")

    # ---- (1 | Subject:Pathway) ----
    sigma_subj_pathway = pm.HalfNormal("sigma_subj_pathway", 0.3)
    z_subj_pathway = pm.Normal("z_subj_pathway", 0, 1, dims="subj_pathway")
    v_subj_pathway = pm.Deterministic(
        "v_subj_pathway", z_subj_pathway * sigma_subj_pathway, dims="subj_pathway"
    )

    # ---- linear predictor ----
    mu = (
        beta_pathway[path_idx_d] # pathway-specific intercept when group = 0, session = 0 (averages)
        + beta_group * group_d # group main effect, collapsed across pathways
        + b_g_p[path_idx_d] * group_d # pathway-specific deviation from group main effect
        + u_int[subj_idx_d] # subject-specific intercept
        + v_subj_pathway[sp_idx_d] # subject-specific offset at this pathway
    )

    sigma = pm.HalfNormal("sigma", 0.15)
    pm.Normal("y", mu=mu, sigma=sigma, observed=z_obs, dims="obs_id")

   
    dt_aim1_null = pm.sample(draws=4000, tune=2000, chains=4,
                          target_accept=0.95, random_seed=42)

    pm.compute_log_likelihood(dt_aim1_null)

# %%
# Main Effects Model

# z ~ Group + Session + Pathway + (1 + Session | Subject) + (1 | Subject:Pathway)

# Build the design for one statistical run, explicitly
df = master_long[master_long["Run"] == analysis_run].copy()
print(df.shape)

# 1. Create dictionaries for sum/effect coding
group_map = {"HV": 0.5, "LV": -0.5}
session_map = {"ses-02": 0.5, "ses-01": -0.5}

# 2. Map the text labels to your float values
df["group_ind"] = df["Group"].map(group_map)
df["session_ind"] = df["Session"].map(session_map)

# Integer codes for the grouping factors (factorize keeps a label lookup)
subj_idx, subjects_u = pd.factorize(df["Subject"])
path_idx, paths_u    = pd.factorize(df["Pathway"])

# combined Subject:Pathway grouping unit
sp_key = df["Subject"].astype(str) + "__" + df["Pathway"].astype(str)
sp_idx, sp_u = pd.factorize(sp_key)

coords = {
    "subject": subjects_u,
    "pathway": paths_u,
    "subj_pathway": sp_u,         
    "re": ["intercept", "slope"],
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as aim1_informed_maineffects:

    path_idx_d = pm.Data("path_idx", path_idx, dims="obs_id")
    subj_idx_d = pm.Data("subj_idx", subj_idx, dims="obs_id")
    sp_idx_d   = pm.Data("sp_idx", sp_idx, dims="obs_id")      
    group_d    = pm.Data("group", df["group_ind"].values.astype(float), dims="obs_id")
    session_d  = pm.Data("session", df["session_ind"].values.astype(float), dims="obs_id")
    z_obs      = pm.Data("z_obs", df["z_value"].values, dims="obs_id")

    # ---- fixed effects ----
    # per-pathway intercept
    beta_pathway = pm.Normal("beta_pathway", mu=0.3, sigma=0.3, dims="pathway")

    # marginal main effects
    beta_group   = pm.Normal("beta_group",   mu=0, sigma=0.10)
    beta_session = pm.Normal("beta_session", mu=0, sigma=0.10)

    # pathway deviations, constrained to sum to zero
    b_g_p   = pm.ZeroSumNormal("b_group_x_pathway",            sigma=0.10, dims="pathway")
    b_s_p   = pm.ZeroSumNormal("b_session_x_pathway",          sigma=0.10, dims="pathway")

    # ---- (1 + Session | Subject), correlated subject intercept + slope ----
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol_subj", n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(0.3), compute_corr=True
    )
    pm.Deterministic("subj_int_slope_corr", corr[0, 1])
    z_subj = pm.Normal("z_subj", 0, 1, dims=("subject", "re"))
    u_subj = pm.Deterministic("u_subj", pt.dot(z_subj, chol.T), dims=("subject", "re"))
    u_int, u_slope = u_subj[:, 0], u_subj[:, 1]

    # ---- (1 | Subject:Pathway), intercept only ----
    sigma_subj_pathway = pm.HalfNormal("sigma_subj_pathway", 0.3)
    z_subj_pathway = pm.Normal("z_subj_pathway", 0, 1, dims="subj_pathway")
    v_subj_pathway = pm.Deterministic(
        "v_subj_pathway", z_subj_pathway * sigma_subj_pathway, dims="subj_pathway"
    )

    # ---- linear predictor ----
    mu = (
        beta_pathway[path_idx_d]    # pathway-specific intercept when group = 0, session = 0 (averages)
        + beta_group * group_d      # group main effect, collapsed across pathways and sessions
        + beta_session * session_d  # session main effect, collapsed across groups and pathways
        + b_g_p[path_idx_d] * group_d   # pathway-specific deviation from group main effect
        + b_s_p[path_idx_d] * session_d # pathway-specific deviation from session main effect
        + u_int[subj_idx_d]         # subject-specific intercept
        + u_slope[subj_idx_d] * session_d  # subject-specific slope
        + v_subj_pathway[sp_idx_d]  # subject-specific offset at this pathway
    )

    sigma = pm.HalfNormal("sigma", 0.15)
    pm.Normal("y", mu=mu, sigma=sigma, observed=z_obs, dims="obs_id")

    # ---- deliverables ----
    pm.Deterministic("sess_effect", beta_session + b_s_p, dims="pathway")


    main_effects_informed_prior_checks = pm.sample_prior_predictive(draws=1000, random_seed=42)
    dt_aim1_maineffects = pm.sample(draws=4000, tune=2000, chains=4,
                          target_accept=0.95, random_seed=42)
    main_effects_informed_posterior_checks = pm.sample_posterior_predictive(dt_aim1_maineffects, random_seed=42)

    dt_aim1_maineffects.update(main_effects_informed_prior_checks)
    dt_aim1_maineffects.update(main_effects_informed_posterior_checks)

    pm.compute_log_likelihood(dt_aim1_maineffects)

# %%
# Main Effects Model with Behavior

# z ~ Group + Session + Pathway + LI + (1 + Session | Subject) + (1 | Subject:Pathway)

# Build the design for one statistical run
df = master_long[master_long["Run"] == analysis_run].copy()
print(df.shape)

# 1. Create dictionaries for sum/effect coding
group_map = {"HV": 0.5, "LV": -0.5}
session_map = {"ses-02": 0.5, "ses-01": -0.5}

# 2. Map the text labels to your float values
df["group_ind"] = df["Group"].map(group_map)
df["session_ind"] = df["Session"].map(session_map)

# Integer codes for the grouping factors
subj_idx, subjects_u = pd.factorize(df["Subject"])
path_idx, paths_u    = pd.factorize(df["Pathway"])

# Combined Subject:Pathway grouping unit
sp_key = df["Subject"].astype(str) + "__" + df["Pathway"].astype(str)
sp_idx, sp_u = pd.factorize(sp_key)

coords = {
    "subject": subjects_u,
    "pathway": paths_u,
    "subj_pathway": sp_u,         
    "re": ["intercept", "slope"],
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as aim1_maineffects_li:

    # ---- Data Containers ----
    path_idx_d = pm.Data("path_idx", path_idx, dims="obs_id")
    subj_idx_d = pm.Data("subj_idx", subj_idx, dims="obs_id")
    sp_idx_d   = pm.Data("sp_idx", sp_idx, dims="obs_id")      
    group_d    = pm.Data("group", df["group_ind"].values.astype(float), dims="obs_id")
    session_d  = pm.Data("session", df["session_ind"].values.astype(float), dims="obs_id")
    li_d       = pm.Data("li_z", df["LI_z"].values.astype(float), dims="obs_id") # Behavioral data
    z_obs      = pm.Data("z_obs", df["z_value"].values, dims="obs_id")

    # ---- Fixed Effects (Marginal) ----
    beta_pathway      = pm.Normal("beta_pathway", mu=0.3, sigma=0.3, dims="pathway")
    beta_group        = pm.Normal("beta_group",   mu=0, sigma=0.10)
    beta_session      = pm.Normal("beta_session", mu=0, sigma=0.10)
    beta_li           = pm.Normal("beta_li",      mu=0, sigma=0.10) #  LI Main Effect

    # ---- Pathway Deviations (Zero-Sum) ----
    b_g_p    = pm.ZeroSumNormal("b_group_x_pathway", sigma=0.10, dims="pathway")
    b_s_p    = pm.ZeroSumNormal("b_session_x_pathway", sigma=0.10, dims="pathway")
    b_li_p   = pm.ZeroSumNormal("b_li_x_pathway", sigma=0.10, dims="pathway") # Pathway specific LI deviation
    

    # ---- Random Effects ----
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol_subj", n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(0.3), compute_corr=True
    )
    pm.Deterministic("subj_int_slope_corr", corr[0, 1])
    z_subj = pm.Normal("z_subj", 0, 1, dims=("subject", "re"))
    u_subj = pm.Deterministic("u_subj", pt.dot(z_subj, chol.T), dims=("subject", "re"))
    u_int, u_slope = u_subj[:, 0], u_subj[:, 1]

    sigma_subj_pathway = pm.HalfNormal("sigma_subj_pathway", 0.3)
    z_subj_pathway = pm.Normal("z_subj_pathway", 0, 1, dims="subj_pathway")
    v_subj_pathway = pm.Deterministic(
        "v_subj_pathway", z_subj_pathway * sigma_subj_pathway, dims="subj_pathway"
    )

    # ---- Linear Predictor ----
    mu = (
        beta_pathway[path_idx_d]    
        + beta_group * group_d      
        + beta_session * session_d  
        + beta_li * li_d                                # Overall effect of being a good learner
        + b_g_p[path_idx_d] * group_d   
        + b_s_p[path_idx_d] * session_d 
        + b_li_p[path_idx_d] * li_d
        + u_int[subj_idx_d]         
        + u_slope[subj_idx_d] * session_d  
        + v_subj_pathway[sp_idx_d]  
    )

    sigma = pm.HalfNormal("sigma", 0.15)
    pm.Normal("y", mu=mu, sigma=sigma, observed=z_obs, dims="obs_id")

    # ---- Deliverables ----
    pm.Deterministic("sess_effect", beta_session + b_s_p, dims="pathway")
    pm.Deterministic("LI_effect", beta_li + b_li_p, dims="pathway")  # Pathway-specific LI effect

    # ---- Sampling ----
    maineffects_li_prior = pm.sample_prior_predictive(draws=1000, random_seed=42)
    dt_aim1_maineffects_li = pm.sample(draws=4000, tune=2000, chains=4, target_accept=0.95, random_seed=42)
    li_interaction_posterior = pm.sample_posterior_predictive(dt_aim1_maineffects_li, random_seed=42)

    dt_aim1_maineffects_li.update(maineffects_li_prior)
    dt_aim1_maineffects_li.update(li_interaction_posterior)
    pm.compute_log_likelihood(dt_aim1_maineffects_li)

# %%
# All 2-Way Interactions Model
#
# z ~ (Group + Session + LI)^2 * Pathway + (1 + Session | Subject) + (1 | Subject:Pathway)

df = master_long[master_long["Run"] == analysis_run].copy()
print(df.shape)

# 1. Create dictionaries for sum/effect coding
group_map = {"HV": 0.5, "LV": -0.5}
session_map = {"ses-02": 0.5, "ses-01": -0.5}

# 2. Map the text labels to your float values
df["group_ind"] = df["Group"].map(group_map)
df["session_ind"] = df["Session"].map(session_map)

# Integer codes for the grouping factors
subj_idx, subjects_u = pd.factorize(df["Subject"])
path_idx, paths_u    = pd.factorize(df["Pathway"])

sp_key = df["Subject"].astype(str) + "__" + df["Pathway"].astype(str)
sp_idx, sp_u = pd.factorize(sp_key)

coords = {
    "subject": subjects_u,
    "pathway": paths_u,
    "subj_pathway": sp_u,
    "re": ["intercept", "slope"],
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as aim1_interaction_all2:

    # ---- Data Containers ----
    path_idx_d = pm.Data("path_idx", path_idx, dims="obs_id")
    subj_idx_d = pm.Data("subj_idx", subj_idx, dims="obs_id")
    sp_idx_d   = pm.Data("sp_idx", sp_idx, dims="obs_id")
    group_d    = pm.Data("group", df["group_ind"].values.astype(float), dims="obs_id")
    session_d  = pm.Data("session", df["session_ind"].values.astype(float), dims="obs_id")
    li_d       = pm.Data("li_z", df["LI_z"].values.astype(float), dims="obs_id") 
    z_obs      = pm.Data("z_obs", df["z_value"].values, dims="obs_id")

    # ---- Fixed Effects (Marginal Main Effects) ----
    beta_pathway = pm.Normal("beta_pathway", mu=0.3, sigma=0.3, dims="pathway") 
    beta_group   = pm.Normal("beta_group", mu=0, sigma=0.1)
    beta_session = pm.Normal("beta_session", mu=0, sigma=0.1)
    beta_li      = pm.Normal("beta_li", mu=0, sigma=0.10) 

    # ---- Marginal 2-Way Interactions (Global) ----
    b_g_s  = pm.Normal("b_group_x_session", mu=0., sigma=0.1)
    b_li_s = pm.Normal("b_li_x_session", mu=0., sigma=0.1)
    b_g_li = pm.Normal("b_group_x_li", mu=0., sigma=0.1) 

    # ---- Pathway Deviations (Zero-Sum Main Effects) ----
    b_g_p  = pm.ZeroSumNormal("b_group_x_pathway", sigma=0.1, dims="pathway")
    b_s_p  = pm.ZeroSumNormal("b_session_x_pathway", sigma=0.1, dims="pathway")
    b_li_p = pm.ZeroSumNormal("b_li_x_pathway", sigma=0.1, dims="pathway") 

    # ---- Pathway Deviations (Zero-Sum 2-Way Interactions) ----
    b_g_s_p  = pm.ZeroSumNormal("b_group_x_session_x_pathway", sigma=0.1, dims="pathway")
    b_li_s_p = pm.ZeroSumNormal("b_li_x_session_x_pathway", sigma=0.1, dims="pathway")
    b_g_li_p = pm.ZeroSumNormal("b_group_x_li_x_pathway", sigma=0.1, dims="pathway") 
   
    # ---- Random Effects ----
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol_subj", n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(0.3), compute_corr=True
    )
    pm.Deterministic("subj_int_slope_corr", corr[0, 1])
    z_subj = pm.Normal("z_subj", 0, 1, dims=("subject", "re"))
    u_subj = pm.Deterministic("u_subj", pt.dot(z_subj, chol.T), dims=("subject", "re"))
    u_int, u_slope = u_subj[:, 0], u_subj[:, 1]

    sigma_subj_pathway = pm.HalfNormal("sigma_subj_pathway", 0.3)
    z_subj_pathway = pm.Normal("z_subj_pathway", 0, 1, dims="subj_pathway")
    v_subj_pathway = pm.Deterministic(
        "v_subj_pathway", z_subj_pathway * sigma_subj_pathway, dims="subj_pathway"
    )

    # ---- Linear Predictor ----
    mu = (
        beta_pathway[path_idx_d] 
        + beta_group * group_d 
        + beta_session * session_d 
        + beta_li * li_d 
        
        # Pathway Main Effect Deviations
        + b_g_p[path_idx_d] * group_d 
        + b_s_p[path_idx_d] * session_d 
        + b_li_p[path_idx_d] * li_d 
        
        # 2-Way Global Interactions
        + b_g_s * group_d * session_d 
        + b_li_s * li_d * session_d 
        + b_g_li * group_d * li_d
        
        # 2-Way Pathway Deviations
        + b_g_s_p[path_idx_d] * group_d * session_d 
        + b_li_s_p[path_idx_d] * li_d * session_d 
        + b_g_li_p[path_idx_d] * group_d * li_d
        
        # Random Effects
        + u_int[subj_idx_d] 
        + u_slope[subj_idx_d] * session_d 
        + v_subj_pathway[sp_idx_d] 
    )

    sigma = pm.HalfNormal("sigma", 0.15)
    pm.Normal("y", mu=mu, sigma=sigma, observed=z_obs, dims="obs_id")

    # ---- Deliverables ----
    pm.Deterministic("sess_effect", beta_session + b_s_p, dims="pathway")
    pm.Deterministic("group_diff_in_sess_effect", b_g_s + b_g_s_p, dims="pathway")
    pm.Deterministic("sess_effect_LV", (beta_session + b_s_p) - 0.5 * (b_g_s + b_g_s_p), dims="pathway")
    pm.Deterministic("sess_effect_HV", (beta_session + b_s_p) + 0.5 * (b_g_s + b_g_s_p), dims="pathway")
    pm.Deterministic("group_diff_in_Learning_Effect", b_g_li + b_g_li_p, dims="pathway")
    pm.Deterministic("LI_effect_HV", (beta_li + b_li_p) + 0.5 * (b_g_li + b_g_li_p), dims="pathway")
    pm.Deterministic("LI_effect_LV", (beta_li + b_li_p) - 0.5 * (b_g_li + b_g_li_p), dims="pathway")

    # ---- Sampling ----
    all2_interaction_prior = pm.sample_prior_predictive(draws=1000, random_seed=42)
    dt_aim1_interaction_all2 = pm.sample(draws=4000, tune=2000, chains=4, target_accept=0.95, random_seed=42)
    interaction_posterior = pm.sample_posterior_predictive(dt_aim1_interaction_all2, random_seed=42)

    dt_aim1_interaction_all2.update(all2_interaction_prior)
    dt_aim1_interaction_all2.update(interaction_posterior)
    pm.compute_log_likelihood(dt_aim1_interaction_all2)

# %%
# Fullest model with behavior and all interactions

# z ~ Group * Session * LI * Pathway + (1 + Session | Subject) + (1 | Subject:Pathway)

# Build the design for one statistical run
df = master_long[master_long["Run"] == analysis_run].copy()
print(df.shape)

# 1. Create dictionaries for sum/effect coding
group_map = {"HV": 0.5, "LV": -0.5}
session_map = {"ses-02": 0.5, "ses-01": -0.5}

# 2. Map the text labels to your float values
df["group_ind"] = df["Group"].map(group_map)
df["session_ind"] = df["Session"].map(session_map)

# Integer codes for the grouping factors 
subj_idx, subjects_u = pd.factorize(df["Subject"])
path_idx, paths_u    = pd.factorize(df["Pathway"])

sp_key = df["Subject"].astype(str) + "__" + df["Pathway"].astype(str)
sp_idx, sp_u = pd.factorize(sp_key)

coords = {
    "subject": subjects_u,
    "pathway": paths_u,
    "subj_pathway": sp_u,         
    "re": ["intercept", "slope"],
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as aim1_full_3way_interaction:

    # ---- Data Containers ----
    path_idx_d = pm.Data("path_idx", path_idx, dims="obs_id")
    subj_idx_d = pm.Data("subj_idx", subj_idx, dims="obs_id")
    sp_idx_d   = pm.Data("sp_idx", sp_idx, dims="obs_id")      
    group_d    = pm.Data("group", df["group_ind"].values.astype(float), dims="obs_id")
    session_d  = pm.Data("session", df["session_ind"].values.astype(float), dims="obs_id")
    li_d       = pm.Data("li_z", df["LI_z"].values.astype(float), dims="obs_id") 
    z_obs      = pm.Data("z_obs", df["z_value"].values, dims="obs_id")

    # ---- Fixed Effects (Marginal Main Effects) ----
    beta_pathway = pm.Normal("beta_pathway", mu=0.3, sigma=0.3, dims="pathway") 
    beta_group   = pm.Normal("beta_group",   mu=0, sigma=0.1)
    beta_session = pm.Normal("beta_session", mu=0, sigma=0.1)
    beta_li      = pm.Normal("beta_li",      mu=0, sigma=0.1) 
    
    # ---- Marginal 2-Way Interactions ----
    b_g_s  = pm.Normal("b_group_x_session", mu=0., sigma=0.1) 
    b_s_li = pm.Normal("b_session_x_li",    mu=0., sigma=0.1) 
    b_g_li = pm.Normal("b_group_x_li",      mu=0., sigma=0.1) # Group x LI
    
    # ---- Marginal 3-Way Interaction ----
    b_g_s_li = pm.Normal("b_group_x_session_x_li", mu=0., sigma=0.1) # The 3-way

    # ---- Pathway Deviations (Zero-Sum Main Effects) ----
    b_g_p  = pm.ZeroSumNormal("b_group_x_pathway", sigma=0.1, dims="pathway")
    b_s_p  = pm.ZeroSumNormal("b_session_x_pathway", sigma=0.1, dims="pathway")
    b_li_p = pm.ZeroSumNormal("b_li_x_pathway", sigma=0.1, dims="pathway") 
    
    # ---- Pathway Deviations (Zero-Sum 2-Way Interactions) ----
    b_g_s_p  = pm.ZeroSumNormal("b_group_x_session_x_pathway", sigma=0.1, dims="pathway")
    b_s_li_p = pm.ZeroSumNormal("b_session_x_li_x_pathway", sigma=0.1, dims="pathway") 
    b_g_li_p = pm.ZeroSumNormal("b_group_x_li_x_pathway", sigma=0.1, dims="pathway") 
    
    # ---- Pathway Deviations (Zero-Sum 3-Way Interaction) ----
    b_g_s_li_p = pm.ZeroSumNormal("b_group_x_sess_x_li_x_pathway", sigma=0.1, dims="pathway") 

    # ---- Random Effects ----
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol_subj", n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(0.3), compute_corr=True
    )
    pm.Deterministic("subj_int_slope_corr", corr[0, 1])
    z_subj = pm.Normal("z_subj", 0, 1, dims=("subject", "re"))
    u_subj = pm.Deterministic("u_subj", pt.dot(z_subj, chol.T), dims=("subject", "re"))
    u_int, u_slope = u_subj[:, 0], u_subj[:, 1]

    sigma_subj_pathway = pm.HalfNormal("sigma_subj_pathway", 0.3)
    z_subj_pathway = pm.Normal("z_subj_pathway", 0, 1, dims="subj_pathway")
    v_subj_pathway = pm.Deterministic(
        "v_subj_pathway", z_subj_pathway * sigma_subj_pathway, dims="subj_pathway"
    )

    # ---- Linear Predictor ----
    mu = (
        beta_pathway[path_idx_d]    
        + beta_group * group_d      
        + beta_session * session_d  
        + beta_li * li_d                                
        + b_g_p[path_idx_d] * group_d   
        + b_s_p[path_idx_d] * session_d 
        + b_li_p[path_idx_d] * li_d                     
        
        # 2-Way Globals
        + b_g_s * group_d * session_d                   
        + b_s_li * session_d * li_d                     
        + b_g_li * group_d * li_d
        
        # 3-Way Global
        + b_g_s_li * group_d * session_d * li_d
        
        # 2-Way Pathway Deviations
        + b_g_s_p[path_idx_d] * group_d * session_d     
        + b_s_li_p[path_idx_d] * session_d * li_d       
        + b_g_li_p[path_idx_d] * group_d * li_d
        
        # 3-Way Pathway Deviation
        + b_g_s_li_p[path_idx_d] * group_d * session_d * li_d
        
        + u_int[subj_idx_d]         
        + u_slope[subj_idx_d] * session_d  
        + v_subj_pathway[sp_idx_d]  
    )

    sigma = pm.HalfNormal("sigma", 0.15)
    pm.Normal("y", mu=mu, sigma=sigma, observed=z_obs, dims="obs_id")

    # ---- Deliverables ----
    pm.Deterministic("sess_effect", beta_session + b_s_p, dims="pathway")
    pm.Deterministic("group_x_sess_effect", b_g_s + b_g_s_p, dims="pathway")
    pm.Deterministic("sess_x_li_effect", b_s_li + b_s_li_p, dims="pathway") 
    pm.Deterministic("sess_effect_LV",
                     (beta_session + b_s_p) - 0.5 * (b_g_s + b_g_s_p), dims="pathway")
    pm.Deterministic("sess_effect_HV",
                     (beta_session + b_s_p) + 0.5 * (b_g_s + b_g_s_p), dims="pathway")
    pm.Deterministic("LI_effect_HV", (beta_li + b_li_p) + 0.5 * (b_g_li + b_g_li_p), dims="pathway")
    pm.Deterministic("LI_effect_LV", (beta_li + b_li_p) - 0.5 * (b_g_li + b_g_li_p), dims="pathway")

    # The ultimate 3-way deliverable
    pm.Deterministic("group_x_sess_x_li_effect", b_g_s_li + b_g_s_li_p, dims="pathway")

    # ---- Sampling ----
    interaction_prior_checks = pm.sample_prior_predictive(draws=1000, random_seed=42)
    dt_aim1_full_3way = pm.sample(draws=4000, tune=2000, chains=4,
                          target_accept=0.95, random_seed=42)
    interaction_posterior_checks = pm.sample_posterior_predictive(dt_aim1_full_3way, random_seed=42)

    dt_aim1_full_3way.update(interaction_prior_checks)
    dt_aim1_full_3way.update(interaction_posterior_checks)
    pm.compute_log_likelihood(dt_aim1_full_3way)

# %%
# LOO-CV model comparison

model_comparison = az.compare({
    "Null Model": dt_aim1_null,
    "Main Effects": dt_aim1_maineffects,
    "Main Effects + LI": dt_aim1_maineffects_li,
    "All 2-Way Interactions": dt_aim1_interaction_all2,
    "Full 3-Way Interaction": dt_aim1_full_3way
})

print(model_comparison)

# %%
az.plot_compare(model_comparison)

plt.savefig("/path/to/figs/model_comparison.png", dpi=300, bbox_inches='tight')

# %%
print(az.summary(dt_aim1_maineffects, var_names=["sigma", "sigma_subj_pathway"]))

print(az.summary(dt_aim1_maineffects_li, var_names=["sigma", "sigma_subj_pathway"]))

print(az.summary(dt_aim1_null, var_names=["sigma", "sigma_subj_pathway"]))

# %%
az.plot_ppc_dist(dt_aim1_maineffects_li, group="posterior_predictive")

# %%
az.plot_ppc_dist(dt_aim1_maineffects, group="posterior_predictive")

# %%
# Session Main Effect Ridge Plot

# --- Custom Colors ---
dist_purple = "#967dff" 
dist_edge   = "#7864cc" 

# --- Plotting ---
axes = azp.plot_ridge(
    dt_aim1_maineffects_li,
    group="posterior", 
    var_names=["sess_effect"],
    backend="matplotlib",
    figure_kwargs={"figsize": (14, 8)},
    visuals={
        "face": {"color": dist_purple, "alpha": 0.6},
        "edge": {"color": dist_edge, "linewidth": 1.5}
    }
)

# --- Post-Plot Text Replacement ---
fig = plt.gcf()

# Dictionary of raw text to clean text
label_map = {
    "sess_effect": "Session Main Effect",
    "v1d_training": "V1d",
    "v2d_training": "V2d",
    "hv4_training": "hV4"
}

# Iterate through all axes to find and replace text
for ax in fig.axes:
    # 1. Update general text objects (this handles the arviz_plots table layout)
    for txt in ax.texts:
        current_text = txt.get_text()
        new_text = current_text
        for old, new in label_map.items():
            new_text = new_text.replace(old, new)
        txt.set_text(new_text)
        
        
        # If this specific text object is the y-axis title, move it left
        if new_text == "Session Main Effect":
            current_x, current_y = txt.get_position() # Subtract from current_x to move left. 
            txt.set_position((current_x - 0.1, current_y)) 
            # Optionally force it to align left from its new anchor point:
            txt.set_horizontalalignment('center') 
        # --------------------------
        
    # 2. Update y-axis tick labels (just in case any are drawn as ticks)
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    if yticklabels:
        new_yticklabels = []
        for label in yticklabels:
            new_label = label
            for old, new in label_map.items():
                new_label = new_label.replace(old, new)
            new_yticklabels.append(new_label)
        ax.set_yticklabels(new_yticklabels)

    if len(ax.collections) == 0:
        ax.grid(False)                         # Turn off the grid
        ax.set_xticks([])                      # Delete all x-ticks and labels ("variable", "pathway")
        ax.set_xlabel("")                      # Delete any hidden x-labels
        ax.spines['bottom'].set_visible(False) # Delete the rogue horizontal line!

ax.set_xlabel("Estimated Change in Fisher-z Connectivity (Post - Pre)", fontsize=13)
# Main Title
fig.suptitle("Posterior Distributions: The Effect of Training Regardless of Group", fontsize=16, fontweight='bold', x= 0.65)
#fig.suptitle("Session Main Effect", fontsize=16, fontweight='bold', y=1.02)
#plt.savefig(fig_out_dir/'session_main_effect_ridge_plot.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# Display summary
summary_df = az.summary(dt_aim1_maineffects_li, var_names=["sess_effect"], ci_kind="hdi", ci_prob=0.93)
display(summary_df)

# %%
# Session Main Effect: Posterior Probability of Direction (pd)# Session Main Effect: Posterior Probability of Direction (pd)

# 1. Grab the 16000 posterior draws for your specific pathway
posterior_draws = dt_aim1_maineffects_li.posterior["sess_effect"].sel(pathway="v1d_training <-> IPS0").values.flatten()

# 2. Count what percentage of those 16,000 draws are strictly less than 0
exact_pd = (posterior_draws < 0).mean() #.values

print(f"Exact Probability of Direction: {exact_pd:.4f}")

# %%
# Learning Index Main Effect Ridge Plot

# --- Custom Colors ---
dist_purple = "#967dff" #B1ECCE #967dff
dist_edge   = "#7864cc" #3CD186 #7864cc

# --- Generate the plot ---
axes = az.plot_ridge(
    dt_aim1_maineffects_li,
    group="posterior", 
    var_names=["LI_effect"],
    backend="matplotlib",
    figure_kwargs={"figsize": (14, 8)},
    visuals={
        "face": {"color": dist_purple, "alpha": 0.6},
        "edge": {"color": dist_edge, "linewidth": 1.5}
    }
)

# --- Post-Plot Text Replacement ---
fig = plt.gcf()

# Dictionary of raw text to clean text
label_map = {
    "LI_effect": "LI Effect",
    "v1d_training": "V1d",
    "v2d_training": "V2d",
    "hv4_training": "hV4"
}

# Iterate through all axes to find and replace text
for ax in fig.axes:
    # 1. Update general text objects (this handles the arviz_plots table layout)
    for txt in ax.texts:
        current_text = txt.get_text()
        new_text = current_text
        for old, new in label_map.items():
            new_text = new_text.replace(old, new)
        txt.set_text(new_text)
        
    # 2. Update y-axis tick labels (just in case any are drawn as ticks)
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    if yticklabels:
        new_yticklabels = []
        for label in yticklabels:
            new_label = label
            for old, new in label_map.items():
                new_label = new_label.replace(old, new)
            new_yticklabels.append(new_label)
        ax.set_yticklabels(new_yticklabels)

    if len(ax.collections) == 0:
        ax.grid(False)                         # Turn off the grid
        ax.set_xticks([])                      # Delete all x-ticks and labels ("variable", "pathway")
        ax.set_xlabel("")                      # Delete any hidden x-labels
        ax.spines['bottom'].set_visible(False) # Delete the rogue horizontal line!

ax.set_xlabel("Estimated Effect on Overall Connectivity (Fisher-z)", fontsize=13)
# Main Title
fig.suptitle("Posterior Distributions: Association Between Learning Index and Overall Network Connectivity", fontsize=16, fontweight='bold', x= 0.65)
#fig.suptitle("Session Main Effect", fontsize=16, fontweight='bold', y=1.02)
#plt.savefig(fig_out_dir/'li_effect_ridge_plot.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# Display summary
summary_df = az.summary(dt_aim1_maineffects_li, var_names=["LI_effect"], ci_prob=0.93, ci_kind="hdi")
display(summary_df)

# %%
# LI Main Effect: Posterior Probability of Direction (pd)

# 1. Grab the 16000 posterior draws for your specific pathway
posterior_draws1 = dt_aim1_maineffects_li.posterior["LI_effect"].sel(pathway="v2d_training <-> IPS0").values.flatten()
posterior_draws2 = dt_aim1_maineffects_li.posterior["LI_effect"].sel(pathway="hv4_training <-> IPS0").values.flatten()

# 2. Count what percentage of those 16,000 draws are strictly less than 0
exact_pd1 = (posterior_draws1 > 0).mean() #.values
exact_pd2 = (posterior_draws2 > 0).mean() #.values

print(f"Exact Probability of Direction for v2d <-> IPS0: {exact_pd1:.4f}")
print(f"Exact Probability of Direction for hv4 <-> IPS0: {exact_pd2:.4f}")

# %%
# Extraction the Main Effect of Session for Each Group (LV vs HV) from interaction model to check

# --- Custom Colors ---
lv_color = "#b1ecce" #BE7DFF #2effc0
#    #7ee7bb #7da5ff
hv_color = "#7d98ff"  #128df3
#65b1d7 #d77dff

# --- Extract the HDI summaries ---
df_LV = az.summary(dt_aim1_interaction_all2, var_names=["sess_effect_LV"], ci_prob=0.93)
df_HV = az.summary(dt_aim1_interaction_all2, var_names=["sess_effect_HV"], ci_prob=0.93)

lower_col, upper_col = df_LV.columns[2], df_LV.columns[3]

# --- Extract and Clean Pathway Names ---
raw_pathways = [idx.split("[")[1].replace("]", "") for idx in df_LV.index]

label_map = {
    "v1d_training": "V1d",
    "v2d_training": "V2d",
    "hv4_training": "hV4"
}

# Apply the exact same renaming logic as the ridge plot
pathways = []
for p in raw_pathways:
    clean_p = p
    for old, new in label_map.items():
        clean_p = clean_p.replace(old, new)
    pathways.append(clean_p)

y_pos = np.arange(len(pathways))

# --- Create the plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot LV Group 
LV_mean = df_LV['mean'].astype(float)
LV_lower = df_LV[lower_col].astype(float)
LV_upper = df_LV[upper_col].astype(float)
LV_err_lower = LV_mean - LV_lower 
LV_err_upper = LV_upper - LV_mean

ax.errorbar(LV_mean, y_pos + 0.15, 
            xerr=[LV_err_lower, LV_err_upper], 
            fmt='o', label='LV Group', color=lv_color,
            markersize=8, linewidth=2.5, capsize=0, zorder=2)
            
# Plot HV Group 
HV_mean = df_HV['mean'].astype(float)
HV_lower = df_HV[lower_col].astype(float)
HV_upper = df_HV[upper_col].astype(float)
HV_err_lower = HV_mean - HV_lower 
HV_err_upper = HV_upper - HV_mean

ax.errorbar(HV_mean, y_pos - 0.15, 
            xerr=[HV_err_lower, HV_err_upper], 
            fmt='o', label='HV Group', color=hv_color,
            markersize=8, linewidth=2.5, capsize=0, zorder=2)
            
# --- Formatting ---
ax.axvline(0, color='black', linestyle='--', alpha=0.6, zorder=0) 
ax.set_yticks(y_pos)
ax.set_yticklabels(pathways, fontsize=11)

ax.set_xlabel("Estimated Change in Fisher-z Connectivity (Post - Pre)", fontsize=13)
ax.set_title("Group-Specific Connectivity Changes", fontsize=16, fontweight="bold")

ax.grid(axis='y', color='#E5E5E5', linestyle='--', linewidth=1.0, alpha=0.8, zorder=0)
# Force the x-axis grid to turn off, overriding any global notebook settings
ax.grid(False, axis='x')
ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', labelcolor='black', edgecolor="#0A0909")
ax.invert_yaxis() 

plt.savefig(fig_out_dir/"session_main_effect_groups.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


