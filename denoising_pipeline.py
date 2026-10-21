import subprocess
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion
from nilearn.image import high_variance_confounds, resample_to_img
from nilearn.signal import clean


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

base_data_dir = Path("/path/to/data/")
fs_base_dir = base_data_dir / "freesurfer"
output_dir = Path("/path/to/output/")
output_dir.mkdir(parents=True, exist_ok=True)

# Subject list
subject_list = []

sessions = ["ses-01", "ses-02"]
TR = 1.5

# Subjects with known issues
missing_tapas = []

# Number of confound components
N_WM_COMP = 5
N_CSF_COMP = 5

# VIF threshold
VIF_THRESHOLD = 10

# Left hemisphere ROI definitions
custom_roi_names = [
    'lh.v1d_trai.circle3.5.mgz', 'lh.v1v_trns.circle3.5.mgz',
    'lh.v2d_trai.circle3.5.mgz', 'lh.v2v_trns.circle3.5.mgz',
    'lh.hv4_trai.circle3.5.mgz', 'lh.hv4_trns.circle3.5.mgz'
]

wang_roi_names = [
    'lh.wang15atlas.V1d.mgz', 'lh.wang15atlas.V1v.mgz',
    'lh.wang15atlas.V2d.mgz', 'lh.wang15atlas.V2v.mgz',
    'lh.wang15atlas.V3d.mgz', 'lh.wang15atlas.V3v.mgz',
    'lh.wang15atlas.hV4.mgz', 'lh.wang15atlas.IPS0.mgz',
    'lh.wang15atlas.IPS1.mgz', 'lh.wang15atlas.IPS2.mgz',
    'lh.wang15atlas.IPS3.mgz', 'lh.wang15atlas.IPS4.mgz',
    'lh.wang15atlas.IPS5.mgz'
]


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def setup_logging(output_dir):
    """Creating a log file for each pipeline run with timestamp. Terminal will show key steps and warnings/errors, while the log file will have detailed info."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"pipeline_log_{timestamp}.txt"

    logger = logging.getLogger('denoise_pipeline')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # clear any existing handlers

    # File handler — everything
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                       datefmt='%H:%M:%S'))

    # Only key steps on the terminal (WARNING and above, plus custom INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path


def print_and_log(logger, msg):
    """Print to terminal AND log to file."""
    print(msg)
    logger.info(msg)


def compute_vif(X):
    """Compute Variance Inflation Factor for each column of X."""
    vifs = []
    for i in range(X.shape[1]):
        others = np.delete(X, i, axis=1)
        target = X[:, i]
        betas, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
        predicted = others @ betas
        ss_res = np.sum((target - predicted) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        if ss_tot == 0:
            vifs.append(np.inf)
        else:
            r2 = 1 - ss_res / ss_tot
            vifs.append(1 / max(1 - r2, 1e-10))
    return vifs

def find_freesurfer_dir(fs_base_dir, sub_id):
    """Find FreeSurfer folder: f002 → hum_XXXXX.long.long002"""
    sub_num = sub_id[1:]
    matched = list(fs_base_dir.glob(f"*.long.long{sub_num}"))
    if not matched:
        raise FileNotFoundError(f"No FreeSurfer folder matching '*.long.long{sub_num}'")
    return matched[0]


def project_to_surface(func_vol_path, reg_file, hemi, fs_id, fs_base_dir, out_file, logger):
    """
    Project volumetric functional data onto native cortical surface.
    Uses --srcsubject override to ensure both sessions project onto
    the same longitudinal FreeSurfer surface.
    """
    cmd = [
        "mri_vol2surf",
        "--mov", str(func_vol_path),
        "--reg", str(reg_file),
        "--hemi", hemi,
        "--projfrac", "0.5",
        "--srcsubject", fs_id,
        "--trgsubject", fs_id,
        "--sd", str(fs_base_dir),
        "--o", str(out_file)
    ]
    logger.debug(f"mri_vol2surf: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"mri_vol2surf failed: {result.stderr}")
        raise RuntimeError(f"mri_vol2surf failed for {hemi}")


def extract_roi_timeseries(denoised_data, roi_path, n_vertices, logger):
    """Extract mean time series from a single ROI mask file."""
    results = {}
    if not roi_path.exists():
        logger.warning(f"ROI not found: {roi_path}")
        return results

    roi_data = np.squeeze(nib.load(roi_path).get_fdata())

    # Match ROI length to functional vertex count
    if roi_data.shape[0] > n_vertices:
        roi_data = roi_data[:n_vertices]
    elif roi_data.shape[0] < n_vertices:
        logger.warning(f"ROI {roi_path.name}: {roi_data.shape[0]} vertices "
                       f"vs {n_vertices} functional vertices. Padding.")
        roi_data = np.pad(roi_data, (0, n_vertices - roi_data.shape[0]))

    unique_labels = np.unique(roi_data)
    unique_labels = unique_labels[unique_labels != 0]

    for label_id in unique_labels:
        mask = (roi_data == label_id)
        n_active = np.sum(mask)
        if n_active > 0:
            roi_base = roi_path.name.replace('.mgz', '')
            col_name = roi_base if len(unique_labels) == 1 else f"{roi_base}_ROI_{int(label_id)}"
            ts = denoised_data[:, mask].mean(axis=1)
            results[col_name] = ts
            logger.debug(f"    {col_name}: {n_active} vertices, "
                         f"mean={ts.mean():.3f}, std={ts.std():.4f}")
    return results


def extract_annot_timeseries(denoised_data, annot_path, hemi, atlas_name, n_vertices, logger):
    """Extract time series from FreeSurfer annotation file."""
    results = {}
    if not annot_path.exists():
        logger.warning(f"Annotation not found: {annot_path}")
        return results

    labels, ctab, names = nib.freesurfer.io.read_annot(annot_path)
    labels_adapted = labels[:n_vertices]

    skip_names = {'unknown', 'corpuscallosum', 'background', 'medialwall',
                  'freesurfer_defined_medial_wall'}

    for idx, name_bytes in enumerate(names):
        roi_name = name_bytes.decode('utf-8')
        if roi_name.lower() in skip_names:
            continue
        mask = (labels_adapted == idx)
        n_active = np.sum(mask)
        if n_active > 0:
            col_name = f"{hemi}_{atlas_name}_{roi_name}"
            ts = denoised_data[:, mask].mean(axis=1)
            results[col_name] = ts
            logger.debug(f"    {col_name}: {n_active} vertices")
    return results


# ==============================================================================
# 3. MAIN PROCESSING LOOP
# ==============================================================================

def main():
    logger, log_path = setup_logging(output_dir)

    print_and_log(logger, "=" * 60)
    print_and_log(logger, "Surface-Based Denoising Pipeline")
    print_and_log(logger, f"Subjects: {len(subject_list)} | Sessions: {len(sessions)}")
    print_and_log(logger, f"TR: {TR}s | Bandpass: 0.008-0.09 Hz")
    print_and_log(logger, f"WM components: {N_WM_COMP} | CSF components: {N_CSF_COMP}")
    print_and_log(logger, f"Log: {log_path}")
    print_and_log(logger, "=" * 60)

    for sub in subject_list:
        print_and_log(logger, f"\nProcessing: {sub}")

        try:
            # --- Find FreeSurfer directory ---
            fs_dir = find_freesurfer_dir(fs_base_dir, sub)
            fs_id = fs_dir.name
            logger.info(f"FreeSurfer subject: {fs_id}")

            mri_dir = fs_dir / "mri"
            label_dir = fs_dir / "label"
            roi_dir = base_data_dir / sub / "rois"

            for session in sessions:
                print_and_log(logger, f"  {session}:")

                func_dir = base_data_dir / sub / session / "restingstate"
                if not func_dir.exists():
                    print_and_log(logger, f"    SKIP: session folder not found")
                    continue

                func_vol_path = func_dir / "my_hifi_images.nii.gz"
                motion_path = func_dir /f"sub-{sub}_{session}_motion_fixed.txt"
                physio_path = base_data_dir / sub / session / "physiological" / "multiple_regressors.txt"
                reg_file = func_dir / "register.dof6.lta"

                # Check required files
                for f, name in [(func_vol_path, "Functional"),
                                (motion_path, "Motion"),
                                (reg_file, "Registration")]:
                    if not f.exists():
                        raise FileNotFoundError(f"{name} missing: {f}")

                # ======================================================
                # PHASE 1: Extract confound regressors
                # ======================================================
                print_and_log(logger, "    Phase 1: Extracting confounds...")

                func_vol_img = nib.load(func_vol_path)
                n_timepoints = func_vol_img.shape[-1]
                logger.info(f"Functional shape: {func_vol_img.shape}")

                # WM/CSF masks
                c2_wm_path = mri_dir / "c2_aseg.img" # I had these images seperated from CONN otherwise integrate the aseg.mgz into the pipeline
                c3_csf_path = mri_dir / "c3_aseg.img"

                wm_resampled = resample_to_img(nib.load(c2_wm_path), func_vol_img,
                                               interpolation='continuous')
                csf_resampled = resample_to_img(nib.load(c3_csf_path), func_vol_img,
                                                interpolation='continuous')

                wm_binary = binary_erosion(wm_resampled.get_fdata() > 0.5).astype(np.uint8)
                csf_binary = binary_erosion(csf_resampled.get_fdata() > 0.5).astype(np.uint8)

                wm_mask_img = nib.Nifti1Image(wm_binary, func_vol_img.affine)
                csf_mask_img = nib.Nifti1Image(csf_binary, func_vol_img.affine)

                logger.info(f"WM mask: {np.sum(wm_binary > 0)} voxels")
                logger.info(f"CSF mask: {np.sum(csf_binary > 0)} voxels")

                # Tissue-based confounds
                wm_comp = high_variance_confounds(func_vol_img, mask_img=wm_mask_img,
                                                  n_confounds=N_WM_COMP)
                csf_comp = high_variance_confounds(func_vol_img, mask_img=csf_mask_img,
                                                   n_confounds=N_CSF_COMP)

                # ======================================================
                # Motion parameters
                # ======================================================
                motion_raw = pd.read_csv(motion_path, sep=r"\s+", header=None)
                
                ROT_RADIUS_MM = 50.0  # Power et al. 2012 — displacement on a 50 mm sphere

                m       = motion_raw.iloc[:, :6].values
                trans   = m[:, 0:3]                  # dS, dL, dP — already in mm
                rot_mm  = m[:, 3:6] * ROT_RADIUS_MM  # roll, pitch, yaw — radians -> mm-equivalent
                fd_comp = np.hstack([trans, rot_mm])

                FD = np.sum(np.abs(np.diff(fd_comp, axis=0)), axis=1)
                FD = np.concatenate([[0.0], FD])     # pad first timepoint

                motion_pure = FD.reshape(-1, 1)  # [210 x 1]
                
                # Log motion variance
                for i in range(motion_pure.shape[1]):
                    logger.debug(f"Motion param {i+1} variance: {motion_pure[:, i].var():.8f}")

                # Physio
                has_physio = physio_path.exists() and sub not in missing_tapas
                if has_physio:
                    physio_data = pd.read_csv(physio_path, sep=r"\s+", header=None).values
                    logger.info(f"Physio regressors loaded: {physio_data.shape}")
                else:
                    physio_data = None
                    reason = "known missing" if sub in missing_tapas else "file not found"
                    logger.warning(f"Physio skipped ({reason})")

                # ======================================================
                # PHASE 2: VIF check
                # ======================================================
                print_and_log(logger, "    Phase 2: Building the design matrix and VIF report...")

                # Build design matrix
                if has_physio:
                    master_confounds = np.hstack((motion_pure, wm_comp, csf_comp, physio_data))
                    confound_labels = (
                        ["FD"] +
                        [f"WM_{i+1}" for i in range(N_WM_COMP)] +
                        [f"CSF_{i+1}" for i in range(N_CSF_COMP)] +
                        [f"Physio_{i+1}" for i in range(physio_data.shape[1])]
                    )
                else:
                    master_confounds = np.hstack((motion_pure, wm_comp, csf_comp))
                    confound_labels = (
                        ["FD"] +
                        [f"WM_{i+1}" for i in range(N_WM_COMP)] +
                        [f"CSF_{i+1}" for i in range(N_CSF_COMP)]
                    )

                logger.info(f"Total regressors: {master_confounds.shape[1]}")
                logger.info(f"Labels: {confound_labels}")

                # VIF check
                vifs = compute_vif(master_confounds)
                max_vif = max(vifs)

                # Log all VIFs
                for label, vif in zip(confound_labels, vifs):
                    flag = ""
                    if vif > VIF_THRESHOLD:
                        flag = " *** SEVERE"
                    elif vif > 5:
                        flag = " * CONCERNING"
                    logger.info(f"    {label:<20} VIF: {vif:>8.1f}{flag}")

                if max_vif > VIF_THRESHOLD:
                    print_and_log(logger, f"    WARNING: Max VIF = {max_vif:.1f} (threshold: {VIF_THRESHOLD})")
                else:
                    print_and_log(logger, f"    VIF OK: max = {max_vif:.1f}")

                # Save design matrix
                dm_path = output_dir / f"{sub}_{session}_design_matrix.csv"
                pd.DataFrame(master_confounds, columns=confound_labels).to_csv(dm_path, index=False)
                logger.info(f"Design matrix saved: {dm_path.name}")

                # Save VIF report
                vif_path = output_dir / f"{sub}_{session}_vif_report.csv"
                pd.DataFrame({'Regressor': confound_labels, 'VIF': vifs}).to_csv(vif_path, index=False)
                logger.info(f"VIF report saved: {vif_path.name}")

                # ======================================================
                # PHASE 3: Surface projection
                # ======================================================
                print_and_log(logger, "    Phase 3: Projecting to surface...")

                lh_surface_out = func_dir / "lh.func_native.mgz"
                rh_surface_out = func_dir / "rh.func_native.mgz"

                project_to_surface(func_vol_path, reg_file, "lh",
                                   fs_id, fs_base_dir, lh_surface_out, logger)
                project_to_surface(func_vol_path, reg_file, "rh",
                                   fs_id, fs_base_dir, rh_surface_out, logger)

                # Load projected data
                lh_raw = np.squeeze(nib.load(lh_surface_out).get_fdata()).T
                rh_raw = np.squeeze(nib.load(rh_surface_out).get_fdata()).T

                lh_vertex_count = lh_raw.shape[1]
                rh_vertex_count = rh_raw.shape[1]
                logger.info(f"LH: {lh_vertex_count} vertices | RH: {rh_vertex_count} vertices")

                # Combine for denoising
                surf_data_raw = np.hstack((lh_raw, rh_raw))
                logger.info(f"Combined surface: {surf_data_raw.shape}")

                # ======================================================
                # PHASE 4: Denoising
                # ======================================================
                print_and_log(logger, "    Phase 4: Denoising...")

                denoised_surf_data = clean(
                    signals=surf_data_raw,
                    confounds=master_confounds,
                    detrend=True,
                    standardize=None,
                    standardize_confounds= True,
                    filter='butterworth',
                    low_pass=0.09,
                    high_pass=0.008,
                    t_r=TR
                )

                logger.info(f"Denoised shape: {denoised_surf_data.shape}")
                logger.info(f"Range: [{denoised_surf_data.min():.3f}, {denoised_surf_data.max():.3f}]")

                # Split hemispheres
                lh_denoised = denoised_surf_data[:, :lh_vertex_count]
                rh_denoised = denoised_surf_data[:, lh_vertex_count:]

                # ======================================================
                # OPTIONAL: Save full denoised surface to hard drive
                # ======================================================
                # Load the original raw images just to get their spatial headers
                lh_orig_img = nib.load(lh_surface_out)
                rh_orig_img = nib.load(rh_surface_out)
                
                # Reshape from (Time x Vertices) back to FreeSurfer's (Vertices x 1 x 1 x Time)
                lh_denoised_reshaped = lh_denoised.T[:, np.newaxis, np.newaxis, :]
                rh_denoised_reshaped = rh_denoised.T[:, np.newaxis, np.newaxis, :]
                
                # Create the new MGZ images
                lh_clean_img = nib.MGHImage(lh_denoised_reshaped, lh_orig_img.affine, lh_orig_img.header)
                rh_clean_img = nib.MGHImage(rh_denoised_reshaped, rh_orig_img.affine, rh_orig_img.header)
                
                # Save to disk next to the raw ones
                nib.save(lh_clean_img, func_dir / "lh.func_denoised.mgz")
                nib.save(rh_clean_img, func_dir / "rh.func_denoised.mgz")
                logger.info("Saved fully denoised surface arrays to disk.")

                # ======================================================
                # PHASE 5: ROI extraction
                # ======================================================
                print_and_log(logger, "    Phase 5: Extracting ROIs...")
                subject_results = {}

                # LH custom ROIs (localizer patches)
                for roi_name in custom_roi_names:
                    roi_path = roi_dir / roi_name
                    extracted = extract_roi_timeseries(lh_denoised, roi_path,
                                                       lh_vertex_count, logger)
                    subject_results.update(extracted)

                # LH Wang atlas ROIs
                for roi_name in wang_roi_names:
                    roi_path = label_dir / roi_name
                    extracted = extract_roi_timeseries(lh_denoised, roi_path,
                                                       lh_vertex_count, logger)
                    subject_results.update(extracted)

                # LH annotations (Yeo7, HCP)
                for atlas_name, annot_file in [("Yeo7", "lh.Yeo7.annot"),
                                                ("HCP", "lh.HCP-MMP1.annot")]:
                    annot_path = label_dir / annot_file
                    extracted = extract_annot_timeseries(lh_denoised, annot_path,
                                                         "lh", atlas_name,
                                                         lh_vertex_count, logger)
                    subject_results.update(extracted)

                # RH annotations (Yeo7, HCP) — no custom ROIs for RH
                for atlas_name, annot_file in [("Yeo7", "rh.Yeo7.annot"),
                                                ("HCP", "rh.HCP-MMP1.annot")]:
                    annot_path = label_dir / annot_file
                    extracted = extract_annot_timeseries(rh_denoised, annot_path,
                                                         "rh", atlas_name,
                                                         rh_vertex_count, logger)
                    subject_results.update(extracted)

                # ======================================================
                # PHASE 6: Save
                # ======================================================
                out_csv = output_dir / f"{sub}_{session}_denoised_timeseries.csv"
                pd.DataFrame(subject_results).to_csv(out_csv, index=False)

                print_and_log(logger, f"    DONE: {len(subject_results)} ROIs → {out_csv.name}")

        except Exception as e:
            print_and_log(logger, f"  ERROR: {e}")
            logger.error("Full traceback:", exc_info=True)
            continue

    print_and_log(logger, "\nPipeline Complete!")


if __name__ == "__main__":
    main()