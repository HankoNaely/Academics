# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Converted from walkthrough.ipynb, see https://soft-matter.github.io/trackpy/v0.3.2/tutorial/walkthrough.html for the original code.
# This script reproduces the TrackPy walkthrough as a standalone Python program.
# This script has been enhanced with additionnal features over the course of our project.
# To run it, make sure you have the required libraries installed (trackpy, pims, numpy, pandas, matplotlib).
# You can adjust the configuration parameters at the top of the script before running.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


# ── Imports ────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
import pandas
import pims
import sys
import cv2
import os
import re
from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.stats import norm
from pathlib import Path
from sklearn.mixture import GaussianMixture
from search_range_RMS import search_range, check_density
from theoretical_diffusion import D_from_temperature
from thermor_calib import T_from_resistance
# ───────────────────────────────────────────────────────────────────────────────────


# Set the base directory to the location of this script for relative paths.
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Folder paths ───────────────────────────────────────────────────────────────────
FRAMES_PATH = BASE_DIR / "data" / "R372_T1" 
RAW_DATA_PATH = BASE_DIR / "raw_data"
RESULTS_ROOT = BASE_DIR / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────────

# ── Dynamic Frame Counting ─────────────────────────────────────────────────────────
tif_files = list(FRAMES_PATH.glob("*.tif"))
FRAME_BATCHED = len(tif_files) 
# ───────────────────────────────────────────────────────────────────────────────────

# ── Dynamic Camera Calibration ─────────────────────────────────────────────────────
video_filename = FRAMES_PATH.name + ".avi"
video_file_path = RAW_DATA_PATH / video_filename

if video_file_path.exists() and FRAME_BATCHED > 0:
    cap = cv2.VideoCapture(str(video_file_path))
    
    # Get metadata from the original AVI
    avi_fps = cap.get(cv2.CAP_PROP_FPS)
    avi_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    # Calculate Total Duration of the original video
    # Duration (s) = Total AVI Frames / AVI FPS
    total_duration = avi_frames / avi_fps
    
    # Calculate effective FRAME_INTERVAL for your TIF sequence
    FRAME_INTERVAL = total_duration / FRAME_BATCHED
    
    print(f"Computed FRAME_INTERVAL: {FRAME_INTERVAL:.6f} s (based on {total_duration:.2f}s duration)")
# ───────────────────────────────────────────────────────────────────────────────────



# ── Parameters ─────────────────────────────────────────────────────────────────────
# These parameters can be adjusted as needed before running the script.
# They are roughly organized in the order they are used in the script.

# ── Particles characteristics ──────────────────────────────────────────────────────
DIAMETER = 13                             # particle diameter (px)
MINMASS = 1700                             # brightness threshold (a.u)

# ── Camera calibration ─────────────────────────────────────────────────────────────
PIXEL_SIZE = 0.515625                      # (µm per pixel)
EXPOSURE_TIME = 0.067                      # (s) used for dynamic error estimation in MSD fitting

# ── Additional tracking parameters ─────────────────────────────────────────────────
INVERT = True                              # algorithm looks for bright features, use invert=True if looking for dark features

# ── Motion parameters (used when linking trajectories) ─────────────────────────────
SEARCH_RANGE = 0.77                         # max displacement allowed between two frames (px), do a test run to get a rough estimate of D and then compute using probability of diffusion over a distance delta
MEMORY = 2                                 # frames a particle can disappear before being considered lost
P_MISS = 0.011                            # acceptable fraction of displacements to miss - used for SEARCH_RANGE calculation based on D

# ── Filter 1 (stubs) ───────────────────────────────────────────────────────────────
MIN_FRAMES = round(1/FRAME_INTERVAL)                          # minimum number of frames for a trajectory to be considered reliable

# ── Filter 2 (trajectory quality) ──────────────────────────────────────────────────
THRESHOLD_MASS_MAX = 2450                   # maximum mass of trajectory 
THRESHOLD_MASS_MIN = 1800                   # minimum mass of trajectory 
THRESHOLD_SIZE_MAX = 2.55                  # maximum size of trajectory 
THRESHOLD_SIZE_MIN = 2.2                    # minimum size of trajectory
THRESHOLD_ECC = 0.12                       # maximum eccentricity of trajectory 

# ── Filter 3 (static filter) ───────────────────────────────────────────────────────
STATIC_THRESHOLD = 0.187                   # initial guess for the static threshold (px), will be refined by GMM
P_BROWNIAN_MIN = 0.80                               # minimum Brownian posterior to keep a particle
    
# ── Filter 4 (sub-diffusive trajectories) ──────────────────────────────────────────
ALPHA_MIN = 0.75    # standard lower bound for free Brownian motion

# ── Filter 5 (stretching correction) ───────────────────────────────────────────────
K_STRETCH = 1.5                              # how many standard deviations above the expected Brownian stretching to set the flow-like threshold (higher = more conservative, less flow trajectories removed)

# ── MSD Fitting ────────────────────────────────────────────────────────────────────
TARGET_RELATIVE_ERROR = 0.007             # maximum acceptable relative error on D for the fit to be considered good (used to determine how many points to fit in the MSD curve)
MIN_PAIRS = int(1/TARGET_RELATIVE_ERROR**2)  # minimum number of pairs required at the largest lag time in the fit (based on relative error formula for diffusion coefficient estimation)
ERROR_THRESHOLD = 0.1                       # corrections below 10% of true MSD

# ── Saving processes ───────────────────────────────────────────────────────────────

# ── Request ────────────────────────────────────────────────────────────────────────
def ask_save_setup():

    # Asks the user if they want to save results.
    # Returns (save: bool, save_dir: str or None).

    while True:
        answer = input("\nDo you want to save those results? (Y/N) : ").strip().upper()
        if answer == 'N':
            print("Results will not be saved.")
            return False, None
        elif answer == 'Y':
            break
        else:
            print("  Please enter Y or N.")

    while True:
        resistance_value = input("Enter resistance value R (folder will be R(R)_T(T)_F(F)) : ").strip()
        try_number = input("Enter the try number T :").strip()
        filter_number = input("Enter filter number F : ").strip()
        folder_name = f"R{resistance_value}_T{try_number}_F{filter_number}"
        save_dir = RESULTS_ROOT / folder_name

        if save_dir.exists():
            print(f"\n  WARNING: folder '{folder_name}' already exists at : \n{save_dir}")
            while True:
                overwrite = input("  Overwrite / add files to it? (Y/N) : ").strip().upper()
                if overwrite == 'Y':
                    print(f"  Results will be saved in existing folder : {save_dir}")
                    return True, save_dir
                elif overwrite == 'N':
                    print("  Please choose different numbers.")
                    break
                else:
                    print("  Please enter Y or N.")
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nFolder created : {save_dir}")
            return True, save_dir

# ── Saving figures ─────────────────────────────────────────────────────────────────
def savefig(fig, name, save_dir):
    # Saves a matplotlib figure as PNG if save_dir is set.
    if save_dir:
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [saved] {name}.png")

# ── Saving CSV──────────────────────────────────────────────────────────────────────
def save_csv(df, name, save_dir):
    # Saves a DataFrame as CSV if save_dir is set.
    if save_dir:
        path = os.path.join(save_dir, f"{name}.csv")
        df.to_csv(path)
        print(f"  [saved] {name}.csv")

# ── Saving parameters ──────────────────────────────────────────────────────────────
def save_parameters(save_dir, D_full, D_mean_win, D_win_std, D_SE, sigma, perr,
                    before_f1, after_f1, before_f2, after_f2,
                    before_static, after_static, before_alpha, after_alpha,
                    before_flow, after_flow, auto_threshold, tau_low, cutoff_absolute,
                    slope_em_raw, slope_em_corr, cv_stability,
                    T_C, avi_fps, total_duration):
    if not save_dir:
        return
    path = os.path.join(save_dir, "parameters_and_results.txt")
    
    rel_error_fit = (perr[0] / D_full) * 100 if D_full != 0 else 0
    rel_error_theory = (abs(D_full - D_SE) / D_SE) * 100 if D_SE != 0 else 0
    
    lines = [
        "=" * 60,
        "TRACKPY BROWNIAN ANALYSIS REPORT",
        "=" * 60,
        f"Analysis Date     : {os.path.basename(save_dir)}",
        f"Input Folder      : {FRAMES_PATH}",
        "",
        "── VIDEO METADATA ──────────────────────────────────────────",
        f"Original FPS      : {avi_fps:.2f} frames/s",
        f"Total Duration    : {total_duration:.2f} s",
        f"Frames Analyzed   : {FRAME_BATCHED}",
        f"Effective Interval: {FRAME_INTERVAL:.6f} s",
        "",
        "── TRACKING PARAMETERS ─────────────────────────────────────",
        f"Diameter          : {DIAMETER} px",
        f"Min Mass          : {MINMASS}",
        f"Search Range      : {SEARCH_RANGE} px",
        f"Memory            : {MEMORY} frames",
        f"Pixel Size        : {PIXEL_SIZE} µm/px",
        f"Exposure Time     : {EXPOSURE_TIME} s",
        "",
        "── FILTERING PIPELINE SUMMARY ──────────────────────────────",
        f"1. Stubs (Minimum frame {MIN_FRAMES}) : {before_f1} -> {after_f1} trajectories",
        f"2. Quality ({THRESHOLD_MASS_MAX} > Mass > {THRESHOLD_MASS_MIN}) : {before_f2} -> {after_f2} trajectories",
        f"3. Static (GMM Threshold : {auto_threshold:.3f}) : {before_static} -> {after_static} trajectories",
        f"4. Alpha (Min Alpha {ALPHA_MIN}) : {before_alpha} -> {after_alpha} trajectories",
        f"5. Flow (K={K_STRETCH}) : {before_flow} -> {after_flow} trajectories",
        "",
        "── TEMPERATURE AND THEORETICAL RESULTS ───────────────────────────────────",
        f"Resistance (R)    : {R} Ohm",
        f"Temperature (T)   : {T_C:.2f} °C",
        f"Theoretical D     : {D_SE:.6f} ± {D_SE_err:.6f} µm²/s",
        "",
        "── EMSD FITTING & DIFFUSION ─────────────────────────────────",
        f"Full-video D (Global)   : {D_full:.6f} µm²/s",  
        f"Windowed mean D (WMSD)  : {D_mean_win:.6f} ± {D_win_std:.6f} µm²/s",
        f"Fitting Error (Global)  : {rel_error_fit:.2f}%",
        f"Error vs Theory         : {rel_error_theory:.2f}%",
        f"Localization σ          : {sigma*1000:.2f} nm",
        "",
        "── SYSTEM STABILITY ────────────────────────────────────────",
        f"Windowed D CV%    : {cv_stability:.2f}%", 
        f"Stability Status  : {'STABLE' if cv_stability < 5 else 'DRIFTING/UNSTABLE'}",
        "",
        "── SCALING & CUTOFFS ───────────────────────────────────────",
        f"Lower Fit Cutoff  : {tau_low:.4f} s",
        f"Upper Fit Cutoff  : {cutoff_absolute:.2f} s",
        f"Alpha (Corrected) : {slope_em_corr:.3f}",
        "=" * 60,
    ]
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  [saved] parameters_and_results.txt")
# ───────────────────────────────────────────────────────────────────────────────────



# ── Checking "particles" parameters ────────────────────────────────────────────────
# Functions defined in this part are meant to be run beforehand, targeting basic particles characteristics.
# They can be used to adjust the DIAMETER and MINMASS parameters, and get a basic visual check of any loaded frame.
# They can be run multiple times on different frames to check for consistency across the video.

# ── Frame comparison ───────────────────────────────────────────────────────────────
def frame_comparison(frames, indexes):

    # Displays multiple frames side-by-side for comparison.
    # Parameters :
    # frames : pims object containing the video frames
    # indexes : list of ints, corresponding to the frame indices you want to display (e.g. [0, 10, 20])

    num_plots = len(indexes)
    fig, axes = plt.subplots(1, num_plots,
                             figsize=(4 * num_plots, 4),
                             constrained_layout=True)
    if num_plots == 1:
        axes = [axes]
    for ax, idx in zip(axes, indexes):
        ax.imshow(frames[idx], cmap='gray')
        ax.set_title(f"Frame {idx}", fontsize=12, pad=5)
        ax.axis('off')
    plt.subplots_adjust(wspace=0.01)
    plt.show()
    
    sys.exit()

# ── MINMASS verification ───────────────────────────────────────────────────────────
def target_minmass(frames, index, diameter, minmass, invert):
    
    # This function allows you to visually check the mass distribution of particles.
    # Use this and the visualization of detected features to adjust the MINMASS parameter to detect proper particles while avoiding noise.
    
    f = tp.locate(frames[index], diameter, invert=invert, minmass=minmass)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tp.annotate(f, frames[index], ax=ax1)
    ax1.set_title(f"Detection (minmass={minmass})")
    ax2.hist(f['mass'], bins=40, color='skyblue', edgecolor='black')
    ax2.axvline(minmass, color='red', linestyle='--', label=f'Current min: {minmass}')
    ax2.set(xlabel='Mass', ylabel='Amount of particle', title='Mass (brightness) distribution')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    print(f.head())
    
    sys.exit()

# ── DIAMETER verification ──────────────────────────────────────────────────────────
def target_diameter(frames, index, diameter, minmass, invert):
    
    # This function allows you to visually check the distribution of sub-pixel (fractional part) position of particles.
    # The histrogram shouldn't have any dips (or peaks), if so adjust the diameter parameter to ensure it is a relativly flat plateau.
    # Dips or peaks suggest that the algorithm is biased towards certain sub-pixel positions (integers).
    # This will affect the accuracy of particle localization and tracking.
    plt.figure(figsize=(12, 4)) 
    
    # Increase font sizes even more to compensate for scaling
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', labelsize=20)
    plt.rc('axes', titlesize=20)

    f = tp.locate(frames[index], diameter, invert=invert, minmass=minmass)
    
    # --- 2. RUN BIAS CHECK ---
    # Use the current figure we just created
    res = tp.subpx_bias(f)
    
    # Labeling as before...
    axes = np.array(res).ravel()
    for i, ax in enumerate(axes):
        ax.set_xlabel("Fractional part (pxl)")
        ax.set_ylabel("Count")
        ax.set_title("X-Coordinate" if i == 0 else "Y-Coordinate")

    # --- 3. SAVE WITH TIGHT MARGINS ---
    plt.tight_layout()
    # bbox_inches='tight' removes the white "dead space" around the plot
    plt.savefig(f"bias_d{diameter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.rcdefaults()
    sys.exit()
    
# ── THRESHOLD_ECC verification ─────────────────────────────────────────────────────
def target_ecc(frames, index, diameter, minmass, invert, threshold_ecc):
    
    # This function allows you to visually check the eccentricity distribution of detected features.
    # Eccentricity = 0 is a perfect circle, eccentricity = 1 is a line.
    # The bulk of genuine spherical particles should cluster near 0.
    # Set THRESHOLD_ECC just above the peak of the distribution.
    
    f = tp.locate(frames[index], diameter, invert=invert, minmass=minmass)
    
    kept    = f[f['ecc'] <  threshold_ecc]
    removed = f[f['ecc'] >= threshold_ecc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.imshow(frames[index], cmap='gray')
    for _, row in removed.iterrows():
        c = plt.Circle((row['x'], row['y']), diameter/2, 
                       color='red', fill=False, linewidth=1)
        ax1.add_patch(c)
    ax1.set_title(f"Red = removed")
    ax1.axis('off')
    
    ax2.hist(f['ecc'], bins=40, color='skyblue', edgecolor='black')
    ax2.axvline(threshold_ecc, color='red', linestyle='--',
                label=f'Current threshold: {threshold_ecc}')
    ax2.set_xlabel('Eccentricity')
    ax2.set_ylabel('Count')
    ax2.set_title('Eccentricity distribution\n(0 = perfect circle, 1 = line)')
    ax2.legend()
    
    print(f"  Total features detected : {len(f)}")
    print(f"  Kept with ecc < {threshold_ecc} : {len(kept)}  ({len(kept)/len(f):.1%})")
    print(f"  Median eccentricity : {f['ecc'].median():.3f}")
    print(f"  95th percentile     : {f['ecc'].quantile(0.95):.3f}")
    
    plt.tight_layout()
    plt.show()
    
    sys.exit()
# ───────────────────────────────────────────────────────────────────────────────────



# ── Trajectory stretching calculation───────────────────────────────────────────────
# Those functions are used in the Filter 4 (flow-like trajectories) to calculate the stretching ratio of each trajectory and compare it to expected values.
# It is the ratio of the end-to-end distance to the total path length.
def trajectory_stretching(tm):
    df = tm.copy()
    if 'particle' in df.index.names:
        df = df.reset_index(drop=True)

    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    df['step'] = np.sqrt(df['dx']**2 + df['dy']**2)

    stats = df.groupby('particle').agg(
        x0=('x', 'first'),
        y0=('y', 'first'),
        x1=('x', 'last'),
        y1=('y', 'last'),
        path_length=('step', 'sum'),
        mass_mean=('mass', 'mean'),
        n_frames=('frame', 'count')
    )
    stats['end_to_end'] = np.sqrt(
        (stats['x1'] - stats['x0'])**2 +
        (stats['y1'] - stats['y0'])**2
    )
    stats['stretching'] = (
        stats['end_to_end'] /
        stats['path_length'].replace(0, np.nan)
    )
    return stats
    # Stats contains the stretching ratio for each trajectory, as well as other useful information (mean mass, number of frames, etc.).
    # Stretching ratio close to 1 indicates a straight trajectory (flow-like), while a ratio close to 0 indicates a highly convoluted trajectory (Brownian).
    # This stretching ratio is used to filter out flow-like trajectories in Filter 5.

def is_flow_like(stretching, n_frames, k):
    return stretching > k * np.sqrt(np.pi / (2 * n_frames))


ALPHA_FIT_POINTS = 4   # number of points to use for alpha, starting from index 2
# This function calculates the diffusive exponent alpha from the MSD curve of a trajectory.
def calculate_alpha(msd_series):
    msd_series = msd_series.dropna()
    end_idx = min(2 + ALPHA_FIT_POINTS, len(msd_series))
    valid = msd_series.iloc[3:end_idx]
    if len(valid) < 3:
        return np.nan
    x = np.log(valid.index.values)
    y = np.log(valid.values)
    slope, _ = np.polyfit(x, y, 1)
    return slope

# ── Core loop ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    
    # ── Theoretical diffusion coefficient calculation ──────────────────────────────────
    folder_name = FRAMES_PATH.name                          # "R727_T1"
    match = re.search(r'R(\d+)', folder_name)
    if match:
        R = int(match.group(1)) * 10          # 727 -> 7270 Ohm
    else:
        raise ValueError(f"Could not extract R value from folder name: {folder_name}")

    T_SAMPLE, T_err_K = T_from_resistance(R)  
    T_C = T_SAMPLE - 273.15
    D_SE, D_SE_err    = D_from_temperature(T_SAMPLE, T_err_K)
    print(f"Estimated sample temperature from resistance: {T_C:.1f}°C")
    # ───────────────────────────────────────────────────────────────────────────────────


    # ── Loading frames ─────────────────────────────────────────────────────────────
    frames = pims.as_grey(pims.ImageSequence(str(FRAMES_PATH / "*.tif")))
    print(f"Loaded {len(frames)} frames successfully from : {FRAMES_PATH}")


    # Optional diagnostic calls - to uncomment as needed. 
    # BEWARE : they will stop the script after displaying results, allowing for parameters adjustment before proceeding.
    
    #frame_comparison(frames, [0, 100, 200])
    #target_minmass(frames, 0, DIAMETER, MINMASS, INVERT)
    #target_diameter(frames, 0, DIAMETER, MINMASS, INVERT)
    #target_ecc(frames, 0, DIAMETER, MINMASS, INVERT, THRESHOLD_ECC)


    # ── Saving request at startup ──────────────────────────────────────────────────
    SAVE, SAVE_DIR = ask_save_setup()


    # ── Locate features and link trajectories ──────────────────────────────────────
    f = tp.batch(frames[:FRAME_BATCHED], DIAMETER, minmass=MINMASS, invert=INVERT, engine='numba', processes='auto')
    f_low = tp.batch(frames[:FRAME_BATCHED], DIAMETER, minmass=100, invert=INVERT, engine='numba', processes='auto')
    t = tp.link_df(f, SEARCH_RANGE, memory=MEMORY)
    # f contains the detected features (x, y, mass, size, ecc, etc.) for each frame.
    # t contains the same data as f, with an added 'particle' column indicating for each particle their ID (what trajectory they belong to).
    
    # MINMASS visual verification - this is to check if the chosen MINMASS value is appropriate for the detected features, and adjust it if necessary.
    fig_filter, ax1 = plt.subplots()
    tp.mass_size(f[f['frame'] == 0], ax=ax1)
    tp.mass_size(f_low[f_low['frame'] == 0], ax=ax1)
    ax1.axvline(MINMASS, color='r', linestyle='--', label=f'Min Mass: {MINMASS}')
    ax1.set_title('Minmass per particle')
    savefig(fig_filter, "00_filter_minmass", SAVE_DIR)
    plt.show()

    # ── Filter 1 : Stub filter ──────────────────────────────────────────────────────
    # This filter removes trajectories that last less than MIN_FRAMES frames, as they are likely to be noise or unreliable detections.
    before_f1 = t['particle'].nunique()
    t1 = tp.filter_stubs(t, MIN_FRAMES)
    after_f1 = t1['particle'].nunique()
    print(f"Filter 1 (stubs) - before: {before_f1}  after: {after_f1}")
    # t1 now contains only filtered trajectories from t.
    # ────────────────────────────────────────────────────────────────────────────────

    # ── Filter 2 : Trajectory quality filter ────────────────────────────────────────
    # This filter removes trajectories that do not meet criteria in terms of mass, size, and eccentricity.
    # As opposed to the first mass and size filters applied during feature detection, this one is applied on the trajectories as a whole. 
    # It uses the mean mass, size and eccentricity of all the features in the trajectory.
    stats = t1.groupby('particle').mean()

    fig_filter, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    tp.mass_size(stats, ax=ax1)
    ax1.axvline(THRESHOLD_MASS_MIN, color='r', linestyle='--')
    ax1.axvline(THRESHOLD_MASS_MAX, color='r', linestyle='--')
    ax1.axhline(THRESHOLD_SIZE_MAX, color='r', linestyle='--')
    ax1.axhline(THRESHOLD_SIZE_MIN, color='r', linestyle='--')
    ax1.set_title('Mean Size vs Mass per Trajectory')

    selection = stats[
        (stats['mass'] > THRESHOLD_MASS_MIN) &
        (stats['mass'] < THRESHOLD_MASS_MAX) &
        (stats['size'] < THRESHOLD_SIZE_MAX) &
        (stats['size'] > THRESHOLD_SIZE_MIN) &
        (stats['ecc'] < THRESHOLD_ECC)
    ].index
    t2 = t1[t1['particle'].isin(selection)]

    tp.annotate(t2[t2['frame'] == 0], frames[0], ax=ax2)
    ax2.set_title(f'Frame 0: {len(selection)} Valid Trajectories')

    ax3.hist(stats['mass'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(THRESHOLD_MASS_MIN, color='r', linestyle='--', label='Min')
    ax3.axvline(THRESHOLD_MASS_MAX, color='r', linestyle='--', label='Max')
    ax3.set_title('Distribution of Mean Mass')
    ax3.set_xlabel('Mass')
    ax3.set_ylabel('Count')
    ax3.legend()

    ax4.hist(stats['size'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax4.axvline(THRESHOLD_SIZE_MIN, color='r', linestyle='--', label='Min')
    ax4.axvline(THRESHOLD_SIZE_MAX, color='r', linestyle='--', label='Max')
    ax4.set_title('Distribution of Mean Size ($R_g$)')
    ax4.set_xlabel('Size (pixels)')
    ax4.set_ylabel('Count')
    ax4.legend()

    plt.tight_layout()

    before_f2 = t1['particle'].nunique()
    after_f2  = t2['particle'].nunique()
    print(f"Filter 2 (mass/size/ecc) - before: {before_f2}  after: {after_f2}")

    savefig(fig_filter, "02_filter_mass_size_dist", SAVE_DIR)
    plt.show()

    save_csv(stats, "filter_stats_mean_per_particle", SAVE_DIR)
    # ─────────────────────────────────────────────────────────────────────────────────


    stds = t2.groupby('particle')[['x', 'y']].std().mean(axis=1)

    # ── Filter 3 : Static filter ───────────────────────────────────────────────────────
    # ── GMM fit on the std distribution (log-space) ──────────────────────────────────
    # The Brownian population has a long right tail in linear space — fitting GMM there
    # causes it to ignore the sharp static peak at low std values entirely.
    # Fitting in log space compresses the tail and makes both populations roughly Gaussian,
    # so the GMM separates them correctly.
    # The threshold is found as the Bayes-optimal crossing point of the two log-space
    # Gaussians (where their weighted PDFs are equal), then converted back with exp().
    stds = t2.groupby('particle')[['x', 'y']].std().mean(axis=1)

    log_stds     = np.log(stds.values)
    log_stds_arr = log_stds.reshape(-1, 1)

    # ── Physics-informed initial means ───────────────────────────────────────────────
    # Static particles should fluctuate only due to localization noise (~30–80 nm).
    # Brownian particles have much larger positional std depending on D and frame interval.
    # Seeding the GMM with physically motivated initial means prevents it from
    # converging to an arbitrary partition when the populations are close.
    sigma_loc_px    = 0.08                           # estimated localization noise (px), adjust if needed
    sigma_brown_px  = 0.8                            # rough expected Brownian std (px), adjust if needed
    initial_means   = np.array([[np.log(sigma_loc_px)], [np.log(sigma_brown_px)]])

    gmm = GaussianMixture(
        n_components=2,
        means_init=initial_means,
        random_state=0
    ).fit(log_stds_arr)

    # Sort components so component 0 is always the static (lower mean) population
    order       = np.argsort(gmm.means_.flatten())
    log_means   = gmm.means_.flatten()[order]           # [log(µ_static), log(µ_brownian)]
    log_sigmas  = np.sqrt(gmm.covariances_.flatten()[order])
    gmm_weights = gmm.weights_[order]

    # Convert means back to linear space for reporting
    gmm_means_px = np.exp(log_means)                    # [µ_static px, µ_brownian px]

    # ── Bayes-optimal threshold ───────────────────────────────────────────────────────
    # Find the crossing point of the two weighted Gaussians in log space.
    # This minimises total misclassification probability, accounting for unequal
    # weights and variances (unlike the geometric midpoint, which assumes both are equal).
    def gmm_diff(x):
        p0 = gmm_weights[0] * norm.pdf(x, log_means[0], log_sigmas[0])
        p1 = gmm_weights[1] * norm.pdf(x, log_means[1], log_sigmas[1])
        return p0 - p1

    try:
        log_threshold = brentq(gmm_diff, log_means[0], log_means[1])
        auto_threshold = np.exp(log_threshold)
        print(f"  Bayes-optimal threshold found at {auto_threshold:.3f} px")
    except ValueError:
        # No crossing found between the two means — populations fully overlap
        # Fall back to geometric midpoint and warn the user
        log_threshold  = log_means.mean()
        auto_threshold = np.exp(log_threshold)
        print(f"  WARNING: No GMM crossing found between the two means.")
        print(f"           Populations may overlap too strongly for reliable separation.")
        print(f"           Falling back to geometric midpoint: {auto_threshold:.3f} px")

    print(f"  GMM means        : {gmm_means_px[0]:.3f} px (static)  |  {gmm_means_px[1]:.3f} px (Brownian)")
    print(f"  GMM weights      : {gmm_weights[0]:.2f} (static)  |  {gmm_weights[1]:.2f} (Brownian)")
    print(f"  GMM auto threshold : {auto_threshold:.3f} px   (current STATIC_THRESHOLD = {STATIC_THRESHOLD})")

    # ── Posterior probability (soft) classification ───────────────────────────────────
    # When populations overlap, a hard threshold will always misclassify edge particles.
    # Instead we assign each particle a Brownian posterior probability and keep only
    # those above P_BROWNIAN_MIN, giving a principled, overlap-aware filter.
    # For reference the hard-threshold result is also computed for comparison.

    posteriors    = gmm.predict_proba(log_stds_arr)     # shape (n_particles, 2)
    p_brownian    = posteriors[:, order[1]]             # posterior of Brownian component, re-ordered
    p_static      = posteriors[:, order[0]]

    prob_df = pandas.DataFrame({
        'std':        stds.values,
        'p_static':   p_static,
        'p_brownian': p_brownian,
    }, index=stds.index)

    soft_brownian_ids = prob_df[prob_df['p_brownian'] >= P_BROWNIAN_MIN].index
    hard_brownian_ids = stds[stds > STATIC_THRESHOLD].index                 # manual hard cut for comparison

    print(f"  Soft filter (p_Brownian ≥ {P_BROWNIAN_MIN}) keeps : {len(soft_brownian_ids)} / {len(stds)} particles")
    print(f"  Hard filter (std > {STATIC_THRESHOLD} px)    keeps : {len(hard_brownian_ids)} / {len(stds)} particles")

    # ── Build smooth overlay curves ───────────────────────────────────────────────────
    x_smooth  = np.linspace(max(stds.min(), 1e-6), stds.max(), 400)
    bin_width = (stds.max() - stds.min()) / 50
    n_total   = len(stds)

    def lognormal_counts(x, log_mu, log_sigma, weight):
        return weight * n_total * bin_width * norm.pdf(np.log(x), log_mu, log_sigma) / x

    curve_static   = lognormal_counts(x_smooth, log_means[0], log_sigmas[0], gmm_weights[0])
    curve_brownian = lognormal_counts(x_smooth, log_means[1], log_sigmas[1], gmm_weights[1])
    curve_combined = curve_static + curve_brownian

    # ── Plot ─────────────────────────────────────────────────────────────────────────
    fig, (ax, ax_log, ax_post) = plt.subplots(3, 1, figsize=(15, 18))

    # Panel 1 — linear space with GMM overlay and both threshold lines
    ax.hist(stds, bins=50, color='skyblue', edgecolor='black', alpha=0.6, label='Observed distribution')
    ax.plot(x_smooth, curve_static,   color='mediumpurple', lw=2, label=f'GMM — static (µ={gmm_means_px[0]:.3f} px,  w={gmm_weights[0]:.2f})')
    ax.plot(x_smooth, curve_brownian, color='teal',         lw=2, label=f'GMM — Brownian (µ={gmm_means_px[1]:.3f} px,  w={gmm_weights[1]:.2f})')
    ax.plot(x_smooth, curve_combined, color='gray',         lw=1.5, linestyle='--', alpha=0.7, label='Combined fit')
    ax.axvline(STATIC_THRESHOLD, color='red',    linestyle='--', lw=1.5, label=f'Manual threshold ({STATIC_THRESHOLD:.3f} px)')
    ax.axvline(auto_threshold,   color='orange', linestyle='-',  lw=2,   label=f'Bayes-optimal GMM threshold ({auto_threshold:.3f} px)')
    ax.set_xlabel('Standard Deviation of Position (pixels)')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Linear Space: Std distribution & Log-Normal Fits')
    ax.legend(fontsize=9)

    # Panel 2 — log space (what the GMM actually sees)
    log_x_smooth  = np.linspace(log_stds.min(), log_stds.max(), 400)
    log_bin_width = (log_stds.max() - log_stds.min()) / 50

    def gaussian_counts(x, mu, sigma, weight):
        return weight * n_total * log_bin_width * norm.pdf(x, mu, sigma)

    ax_log.hist(log_stds, bins=50, color='lightgray', edgecolor='black', alpha=0.6, label='Log-transformed distribution')
    ax_log.plot(log_x_smooth, gaussian_counts(log_x_smooth, log_means[0], log_sigmas[0], gmm_weights[0]),
                color='mediumpurple', lw=2, label=f'Static component (log µ={log_means[0]:.2f})')
    ax_log.plot(log_x_smooth, gaussian_counts(log_x_smooth, log_means[1], log_sigmas[1], gmm_weights[1]),
                color='teal',         lw=2, label=f'Brownian component (log µ={log_means[1]:.2f})')
    ax_log.axvline(log_threshold, color='orange', lw=2, label=f'Bayes threshold (log-space = {log_threshold:.2f})')
    ax_log.axvline(np.log(STATIC_THRESHOLD), color='red', linestyle='--', lw=1.5, label=f'Manual threshold (log-space = {np.log(STATIC_THRESHOLD):.2f})')
    ax_log.set_xlabel(r'Log of Standard Deviation $\ln(\sigma_{pos})$')
    ax_log.set_ylabel('Particle count')
    ax_log.set_title('Log-Space: Normal Distributions (where GMM is fitted)')
    ax_log.legend(fontsize=9)

    # Panel 3 — posterior probability scatter: p(Brownian) vs std
    # Points above the dashed line are kept by the soft filter.
    ax_post.scatter(stds.values, p_brownian, s=8, alpha=0.4, color='steelblue', label='p(Brownian) per particle')
    ax_post.axhline(P_BROWNIAN_MIN, color='green',  linestyle='--', lw=1.5, label=f'Soft-filter threshold (p={P_BROWNIAN_MIN})')
    ax_post.axvline(STATIC_THRESHOLD, color='red',  linestyle='--', lw=1.5, label=f'Manual hard threshold ({STATIC_THRESHOLD:.3f} px)')
    ax_post.axvline(auto_threshold,   color='orange', linestyle='-', lw=1.5, label=f'Bayes hard threshold ({auto_threshold:.3f} px)')
    ax_post.set_xlabel('Standard Deviation of Position (pixels)')
    ax_post.set_ylabel('p(Brownian)')
    ax_post.set_title(f'Posterior probability — soft filter keeps {len(soft_brownian_ids)} particles  (hard manual: {len(hard_brownian_ids)})')
    ax_post.legend(fontsize=9)
    ax_post.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    savefig(fig, "02_static_threshold_GMM_comparison", SAVE_DIR)
    plt.show()
    
    before_static = t2['particle'].nunique()
    t3 = t2[t2['particle'].isin(soft_brownian_ids)]     # soft GMM posterior filter
    after_static  = t3['particle'].nunique()
    print(f"Filter 3 (static, soft p≥{P_BROWNIAN_MIN}) — before: {before_static}  after: {after_static}")
    # ─────────────────────────────────────────────────────────────────────────────────

    # ── Plot static-filtered trajectories ───────────────────────────────────────────
    fig_traj, (ax1, ax2) = plt.subplots(1, 2)
    tp.plot_traj(t2, ax=ax1)
    tp.plot_traj(t3, ax=ax2)
    ax1.set_title('Trajectories before static filter')
    ax2.set_title('Trajectories after static filter')
    savefig(fig_traj, "03_filtered_trajectories", SAVE_DIR)
    plt.show()
    # ────────────────────────────────────────────────────────────────────────────────
        
    # ── Filter 4 : Remove sub-diffusive / confined trajectories ─────────────────────
    # Particles near the glass or transiently adsorbed show a saturating MSD (alpha << 1).
    # Free Brownian motion has alpha ≈ 1. We remove trajectories whose scaling exponent
    # falls below ALPHA_MIN, following the standard classification subdiffusive
    im_t3  = tp.imsd(t3, PIXEL_SIZE, 1 / FRAME_INTERVAL)
    alphas = im_t3.apply(calculate_alpha).dropna()

    before_alpha = t3['particle'].nunique()
    valid_alpha_ids = alphas[alphas >= ALPHA_MIN].index
    t4 = t3[t3['particle'].isin(valid_alpha_ids)]
    after_alpha = t4['particle'].nunique()
    print(f"Filter 4 (sub-diffusive) — before: {before_alpha}  after: {after_alpha}")

    # Plot alpha distribution with threshold 
    fig_alpha, ax = plt.subplots(figsize=(10, 5))
    ax.hist(alphas, bins=np.linspace(-0.5, 2.5, 60), color='teal', edgecolor='black', alpha=0.7)
    ax.axvline(ALPHA_MIN, color='red',  lw=2, linestyle='--', label=f'Min threshold α={ALPHA_MIN}')
    ax.axvline(1.0,       color='gray', lw=1.5, linestyle=':', label='Perfect Brownian (α=1)')
    removed_alpha = before_alpha - after_alpha
    ax.set_xlabel(r'Diffusive Exponent $\alpha$')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Sub-diffusive filter — {removed_alpha} particles removed  ({after_alpha} kept)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig_alpha, "04_alpha_filter", SAVE_DIR)
    plt.show()
    
    t_alpha_removed = t3[~t3['particle'].isin(valid_alpha_ids)]
    t_alpha_kept    = t3[t3['particle'].isin(valid_alpha_ids)]

    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if not t_alpha_removed.empty:
        tp.plot_traj(t_alpha_removed, ax=ax1, color='red', alpha=0.5, linewidth=1)
    ax1.set_aspect('equal')

    if not t_alpha_kept.empty:
        tp.plot_traj(t_alpha_kept, ax=ax2, color='green', alpha=0.5, linewidth=1)
    ax2.set_aspect('equal')
    plt.tight_layout()
    
    savefig(fig_comp, "05_alpha_comparison_side_by_side", SAVE_DIR)
    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────


 # ── Filter 4 : Remove flow-like trajectories ────────────────────────────────────
    stats_stretch = trajectory_stretching(t4)

    flow_ids = stats_stretch[
    stats_stretch.apply(lambda row: is_flow_like(row['stretching'], row['n_frames'], K_STRETCH), axis=1) & (stats_stretch['n_frames'] > MIN_FRAMES)].index

    before_flow = t4['particle'].nunique()
    t4_flow     = t4[t4['particle'].isin(flow_ids)]          # trajectories which ID are in flow_ids, hence flow trajectories (for plotting)
    t4_brownian = t4[~t4['particle'].isin(flow_ids)]         # Brownian only  (for MSD)
    after_flow  = t4_brownian['particle'].nunique()
    print(f"Filter 4 (flow-like) - before: {before_flow}  after (Brownian only): {after_flow}")
    # t4_brownian now contains only the Brownian trajectories from t4, with flow-like trajectories removed based on their stretching ratio.

    # ── Plot all vs flow-like vs Brownian ────────────────────────────────────────────
    fig_flow, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), sharex=True, sharey=True)
    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    tp.plot_traj(t4, ax=ax1)
    ax1.set_title('All trajectories (drift not corrected yet)')

    if not t4_flow.empty:
        tp.plot_traj(t4_flow, ax=ax2)
        ax2.set_title(f'Flow-like only (k = {K_STRETCH})')

    tp.plot_traj(t4_brownian, ax=ax3)
    ax3.set_title('Brownian only  (flow-like removed)')

    plt.tight_layout()
    savefig(fig_flow, "03_flow_vs_brownian_trajectories", SAVE_DIR)
    plt.show()


    # ── Plot Size / Mass scatter : all vs flow-like ───────────────────────────────────
    # This plot allows to visually check where are the flow-like trajectories in the size/mass space, and if they correspond to specific values (e.g. bigger mass or size).
    if not t4_flow.empty:
        t4_clean  = t4_flow.reset_index(drop=True)
        stats_sm  = t4_clean.groupby('particle').mean()

        fig_sm, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

        ax1.scatter(stats_sm['mass'], stats_sm['size'], s=15, alpha=0.5)
        ax1.axvline(THRESHOLD_MASS_MIN, color='r', ls='--')
        ax1.axhline(THRESHOLD_SIZE_MIN, color='r', ls='--')
        ax1.set_title('Size / Mass — all trajectories')
        ax1.set_xlabel('Mass')
        ax1.set_ylabel('Size')
        ax1.grid(alpha=0.2)

        ax2.scatter(
            stats_sm.loc[flow_ids, 'mass'],
            stats_sm.loc[flow_ids, 'size'],
            s=40, color='red', alpha=0.9
        )
        ax2.axvline(THRESHOLD_MASS_MIN, color='r', ls='--')
        ax2.axhline(THRESHOLD_SIZE_MIN, color='r', ls='--')
        ax2.set_title('Size / Mass — flow-like trajectories highlighted')
        ax2.set_xlabel('Mass')
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        savefig(fig_sm, "04_size_mass_flow_highlighted", SAVE_DIR)
        plt.show()
    # ──────────────────────────────────────────────────────────────────────────────────
    
    
    # ── Filter 5 : Drift correction ─────────────────────────────────────────────────
    d = tp.compute_drift(t4_brownian)

    fig_drift, ax_d = plt.subplots()
    d.plot(ax=ax_d)
    ax_d.set_title('Drift in x and y directions')
    ax_d.set_xlabel('Frame')
    ax_d.set_ylabel('Drift [px]')
    savefig(fig_drift, "05_drift", SAVE_DIR)
    plt.show()

    save_csv(d, "drift_xy_per_frame", SAVE_DIR)

    t_final = tp.subtract_drift(t4_brownian.copy(), d) 
    # t_final now contains the drift-corrected trajectories from t4_brownian.

    # ── Plot drift-filtered trajectories ───────────────────────────────────────────
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    tp.plot_traj(t4_brownian, ax=ax1)
    tp.plot_traj(t_final, ax=ax2)
    ax1.set_title('Raw trajectories (with overall drift)')
    ax2.set_title('Final drift-corrected trajectories')
    savefig(fig_comp, "06_trajectories_drift_correction", SAVE_DIR)
    plt.show()
    # ────────────────────────────────────────────────────────────────────────────────

    


    # ── MSD analysis on cleared trajectories ──────────────────────────────────────────
    # This is the core of the program, now that we have cleared trajectories through different filters, we can analyze the Brownian motion through MSD.
    # First, the individual MSD is plotted for each trajectory, this allows to check for consistency across trajectories and identify any outliers.
    # Then, the ensemble MSD is plotted and fitted with a power law to extract the diffusion coefficient D and the power law exponent n.
    
    # ── Individual MSD & Alpha Distribution ──────────────────────────────────────────
    im = tp.imsd(t_final, PIXEL_SIZE, 1 / FRAME_INTERVAL)
    em_detail = tp.emsd(t_final, PIXEL_SIZE, 1 / FRAME_INTERVAL, detail=True) # We define them here because it's used for the visual cut in the iMSD plot.
    em_detail.index = em_detail.index * FRAME_INTERVAL # Convert lag times from frames to seconds (because detail=True gives lag times in frames).
    em = em_detail['msd'] # We will use the 'msd' column of the em_detail for the ensemble MSD plot, and the index for the lag times.
    counts = em_detail['N'] # We will use the 'N' column of the em_detail for the pair counts in the ensemble MSD plot.
    
    # We use the first few points (e.g., first 5 lag times) to get the local scaling
    cutoff_absolute = counts[counts >= MIN_PAIRS].index.max() # seconds
    cutoff_absolute_idx_im = np.searchsorted(im.index, cutoff_absolute)  # for alpha
    cutoff_absolute_idx_em = np.searchsorted(em.index, cutoff_absolute)  # for EMSD fit


    fig_diag, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Individual MSD
    ax1.plot(im.index, im, 'k-', alpha=0.1)
    ax1.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$ [s]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Individual MSD (Brownian Only)')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
           
    ax1.axvline(cutoff_absolute, color='blue', linestyle='--', alpha=0.8, label=f'Fit cutoff ({cutoff_absolute:.2f} s,  min pairs = {MIN_PAIRS})')

    
    # Panel 3: EMSD with pair counts

    ax3.plot(em.index, em.values, 'o', markersize=3, color='steelblue', label='EMSD')
    ax3.axvline(cutoff_absolute, color='blue', linestyle='--', alpha=0.8, label=f'Fit cutoff ({cutoff_absolute:.2f} s,  min pairs = {MIN_PAIRS})')

    threshold = counts.max() * 0.5

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('lag time $t$ [s]')
    ax3.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    ax3.set_title('EMSD — fit cutoff vs pair count')
    ax3.grid(True, which="both", ls="-", alpha=0.2)

    ax3b = ax3.twinx()
    ax3b.plot(counts.index, counts.values, color='orange', lw=1.2, alpha=0.7, label='Pair count')
    ax3b.axhline(MIN_PAIRS, color='orange', linestyle=':', alpha=0.6, label=f'{MIN_PAIRS} pairs threshold')
    ax3b.set_ylabel('Number of pairs', color='orange')
    ax3b.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    savefig(fig_diag, "07_individual_MSD_and_alpha_and_counts", SAVE_DIR)
    plt.show()

    save_csv(alphas.to_frame(name='alpha'), "individual_alphas", SAVE_DIR)
    # ──────────────────────────────────────────────────────────────────────────────────

    # ── Ensemble MSD ──────────────────────────────────────────────────────────────────

    def msd_model(t, D, sigma2):
        return 4 * D * (t - EXPOSURE_TIME/3) + 4 * sigma2     # Stokes-Einstein model with offset to account for localization error (sigma²) and dynamic error (EXPOSURE_TIME)


    # Pass 1 — rough fit with arbitrary low cutoff
    em_fit_rough = em.iloc[:cutoff_absolute_idx_em]
    popt_rough, _ = curve_fit(msd_model, em_fit_rough.index.values,
                            em_fit_rough.values, p0=[D_SE, (30e-3)**2],
                            bounds=(0, np.inf))
    D_rough, sigma2_rough = popt_rough

    # Compute principled lower cutoff from rough estimates
    tau_low = abs(sigma2_rough/D_rough - EXPOSURE_TIME/3) / ERROR_THRESHOLD
    low_cutoff_idx_em = np.searchsorted(em.index, tau_low)
    low_cutoff_idx_im = np.searchsorted(im.index, tau_low)

    # Pass 2 — final fit with principled cutoffs on both ends
    em_fit = em.iloc[low_cutoff_idx_em:cutoff_absolute_idx_em]
    t_fit   = em_fit.index.values
    msd_fit = em_fit.values

    popt, pcov = curve_fit(msd_model, t_fit, msd_fit,
                        p0=[D_rough, sigma2_rough],
                        bounds=(0, np.inf))
    
    D      = popt[0]                      # µm²/s  — diffusion coefficient
    sigma2 = popt[1]                      # µm²    — variance of localization error
    sigma  = np.sqrt(sigma2)              # µm     — localization precision 
    perr   = np.sqrt(np.diag(pcov))       # standard errors on D and sigma2
    
    n = 1.0
    A = 4 * D 
    # ──────────────────────────────────────────────────────────────────────────────────
    

    # ── Plot of the EMSD ──────────────────────────────────────────────────────────────
    fig_emsd, ax = plt.subplots()
    ax.plot(em.index, em, 'o', alpha=0.5, label='EMSD data')

    # Plot the corrected fit
    t_plot = em.index.values
    ax.plot(t_plot, 4*D*(t_plot - EXPOSURE_TIME/3) + 4*sigma2, 'r-', lw=2,
            label=f'Linear fit  D = {D:.4f} µm²/s,  sigma = {sigma*1000:.1f} nm')

    # Show what the pure 4Dt line looks like (no offset)
    ax.plot(t_plot, 4*D*t_plot, 'g--', lw=1, alpha=0.6, label='4Dt  (offset removed)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    ax.set_xlabel('lag time $t$ [s]')
    ax.legend()
    ax.set_title(f'EMSD fit  (Brownian only) — D = {D:.4f} µm²/s,  sigma = {sigma*1000:.1f} nm')
    
    savefig(fig_emsd, "08_ensemble_MSD", SAVE_DIR)
    plt.show()

    save_csv(em.to_frame(name='emsd'), "ensemble_MSD", SAVE_DIR)
    # ──────────────────────────────────────────────────────────────────────────────────
    
# ── Windowed MSD — temporal stability check ───────────────────────────────────────
    # The full-video MSD uses all frames but only short lag times.
    # Here we split the video into non-overlapping temporal windows and fit D
    # independently in each one, using the same tau_low / cutoff_absolute window
    # already determined above. This tests whether D is stationary across time —
    # a drift in D across windows is a direct fingerprint of convection building up
    # or thermal equilibration effects not captured by drift correction.
    #
    # WINDOW_DURATION is set automatically to 2.5 × cutoff_absolute — the minimum
    # that guarantees enough displacement pairs at the upper lag-time cutoff within
    # each window. This maximises the number of independent windows from the video.

    MIN_TRAJ_PER_WIN = 10    # minimum trajectories in a window to attempt a fit

    # ── Safe index flatten ────────────────────────────────────────────────────────────
    # tp.subtract_drift can leave 'particle' as both an index level AND a column,
    # which makes reset_index() and groupby() raise errors. We handle all cases:
    _tf = t_final.copy()
    # Case 1: particle is an index level — drop it from index (it stays as column)
    if 'particle' in _tf.index.names:
        _tf = _tf.reset_index(level='particle', drop=True)
    # Case 2: any remaining named index levels that are NOT already columns
    _remaining = [n for n in _tf.index.names if n and n not in _tf.columns]
    if _remaining:
        _tf = _tf.reset_index(level=_remaining)
    else:
        _tf = _tf.reset_index(drop=True)
    # Case 3: deduplicate columns (safety net)
    tw_base = _tf.loc[:, ~_tf.columns.duplicated()].copy()

    if 'particle' not in tw_base.columns or 'frame' not in tw_base.columns:
        print("  Windowed MSD: could not resolve 'particle'/'frame' columns — skipping.")
    else:
        t_max_s         = tw_base['frame'].max() * FRAME_INTERVAL
        WINDOW_DURATION = cutoff_absolute * 5
        n_windows_max   = max(1, int(t_max_s / WINDOW_DURATION))

        print(f"\n  ── Windowed MSD setup ──")
        print(f"  cutoff_absolute  : {cutoff_absolute:.3f} s")
        print(f"  WINDOW_DURATION  : {WINDOW_DURATION:.3f} s  (= 2.5 × cutoff)")
        print(f"  Video duration   : {t_max_s:.1f} s")
        print(f"  Expected windows : {n_windows_max}")
        if n_windows_max < 3:
            print(f"  ⚠ Fewer than 3 windows — stability check will be limited.")

        D_windows = []

        for i in range(n_windows_max):
            f_start = int(i       * WINDOW_DURATION / FRAME_INTERVAL)
            f_end   = int((i + 1) * WINDOW_DURATION / FRAME_INTERVAL)

            mask = (tw_base['frame'] >= f_start) & (tw_base['frame'] < f_end)
            tw   = tw_base[mask].copy()
            tw['frame'] = tw['frame'] - f_start    # re-index frames to 0

            frame_counts = tw.groupby('particle', sort=False)['frame'].count()
            valid_ids    = frame_counts[frame_counts >= MIN_FRAMES].index
            tw           = tw[tw['particle'].isin(valid_ids)]

            if tw['particle'].nunique() < MIN_TRAJ_PER_WIN:
                continue

            try:
                em_w     = tp.emsd(tw, PIXEL_SIZE, 1 / FRAME_INTERVAL)
                low_w    = np.searchsorted(em_w.index, tau_low)
                high_w   = np.searchsorted(em_w.index, cutoff_absolute)
                em_fit_w = em_w.iloc[low_w:high_w]

                if len(em_fit_w) < 4:
                    continue

                # Use the 'sigma2' calculated from the high-confidence full-video fit
                popt_w, pcov_w = curve_fit(
                    lambda t, D_val: msd_model(t, D_val, sigma2), # Only D_val is a variable here
                    em_fit_w.index.values, 
                    em_fit_w.values, 
                    p0=[D], # Guess for D only
                    bounds=(0, np.inf)
                )

                # Extract only the D value
                D_w     = popt_w[0]
                D_w_err = np.sqrt(np.diag(pcov_w))[0]

                D_windows.append({
                    'window':  i,
                    't_start': i * WINDOW_DURATION,
                    't_mid':   (i + 0.5) * WINDOW_DURATION,
                    'D':       D_w,
                    'D_err':   D_w_err,
                    'N_traj':  tw['particle'].nunique()
                })

            except (RuntimeError, ValueError):
                continue

        if len(D_windows) >= 2:
            df_win = pandas.DataFrame(D_windows)

            w_win = 1.0 / df_win['D_err'].replace(0, np.nan) ** 2
            w_win = w_win.fillna(0)
            if w_win.sum() > 0:
                D_win_mean = np.sum(w_win * df_win['D']) / w_win.sum()
                D_win_std  = np.sqrt(1.0 / w_win.sum())
            else:
                D_win_mean = df_win['D'].mean()
                D_win_std  = df_win['D'].std()

            fig_win, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax_d.errorbar(df_win['t_mid'], df_win['D'],
                          yerr=df_win['D_err'],
                          fmt='o-', capsize=4, color='steelblue',
                          linewidth=1.5, markersize=6,
                          label=f'D per {WINDOW_DURATION:.1f}s window')
            ax_d.axhline(D,          color='red',   lw=2,   linestyle='--',
                         label=f'Full-video D = {D:.4f} µm²/s')
            ax_d.axhline(D_win_mean, color='green', lw=1.5, linestyle=':',
                         label=f'Windowed mean = {D_win_mean:.4f} ± {D_win_std:.4f} µm²/s')
            ax_d.fill_between(df_win['t_mid'],
                              D_win_mean - D_win_std,
                              D_win_mean + D_win_std,
                              alpha=0.12, color='green')
            ax_d.set_ylabel('D (µm²/s)')
            ax_d.set_title(
                f'Temporal stability of D — {len(df_win)} × {WINDOW_DURATION:.1f}s windows\n'
                f'Flat = stationary Brownian   |   Drift = convection artifact'
            )
            ax_d.legend(fontsize=9)
            ax_d.grid(True, alpha=0.3)

            ax_n.bar(df_win['t_mid'], df_win['N_traj'],
                     width=WINDOW_DURATION * 0.8, color='teal', alpha=0.7,
                     label='Trajectories per window')
            ax_n.axhline(MIN_TRAJ_PER_WIN, color='red', linestyle=':', lw=1.5,
                         label=f'Min threshold ({MIN_TRAJ_PER_WIN})')
            ax_n.set_xlabel('Window midpoint (s)')
            ax_n.set_ylabel('N trajectories')
            ax_n.legend(fontsize=9)
            ax_n.grid(True, alpha=0.3)

            plt.tight_layout()
            savefig(fig_win, "09_windowed_D_stability", SAVE_DIR)
            plt.show()
            
            cv_percent        = df_win['D'].std() / df_win['D'].mean() * 100

            print(f"\n  ── Windowed MSD stability ──")
            print(f"  Windows computed      : {len(df_win)}")
            print(f"  Windowed mean D       : {D_win_mean:.4f} ± {D_win_std:.4f} µm²/s")
            print(f"  Classic first D       : {D:.4f} µm²/s")
            print(f"  Window-to-window CV   : {cv_percent:.2f}%  (< 5% = stable Brownian)")

            save_csv(df_win[['t_start', 'D', 'D_err', 'N_traj']], "windowed_D", SAVE_DIR)

        else:
            print(f"  Windowed MSD: not enough valid windows (got {len(D_windows)}).")
            print(f"  Video may be too short relative to cutoff_absolute ({cutoff_absolute:.3f} s).")
    # ─────────────────────────────────────────────────────────────────────────────────
    
    # ── Self-consistency checks ─────────────────────────────────────────────────
    
    # The static threshold should be close to sigma (localization noise).
    # If they diverge significantly, the filter may be too aggressive or too loose.
    threshold_px_as_um = auto_threshold * PIXEL_SIZE   # convert back to µm for comparison
    ratio = threshold_px_as_um / sigma                     # should be close to K_STATIC


# ── Per-trajectory alpha (corrected) ─────────────────────────────────────────────
    def calculate_alpha_corrected(msd_series, D, sigma2, exposure_time):
        msd_series = msd_series.dropna()
        t = msd_series.index.values
        
        # 2D offset: 4ε² - 4D(σ_exp/3)  — Savin & Doyle 2005, Eq.14 extended to 2D
        offset = 4 * sigma2 - 4 * D * (exposure_time / 3)
        msd_corrected = msd_series.values - offset
        
        valid_mask = (msd_corrected > 0) & (t >= tau_low) & (t <= cutoff_absolute)
        t_valid   = t[valid_mask]
        msd_valid = msd_corrected[valid_mask]
        
        if len(t_valid) < 3:
            return np.nan
        
        slope, _ = np.polyfit(np.log(t_valid), np.log(msd_valid), 1)
        return slope

    im_final = tp.imsd(t_final, PIXEL_SIZE, 1 / FRAME_INTERVAL)
    alphas_corrected = im_final.apply(
        lambda s: calculate_alpha_corrected(s, D, sigma2, EXPOSURE_TIME)
    ).dropna()

    # ── Ensemble alpha (raw) ──────────────────────────────────────────────────────────
    em_alpha_fit = em.iloc[low_cutoff_idx_em:cutoff_absolute_idx_em]
    slope_em_raw, _ = np.polyfit(
        np.log(em_alpha_fit.index.values),
        np.log(em_alpha_fit.values),
        1
    )

    # ── Ensemble alpha (corrected) ────────────────────────────────────────────────────
    offset_em      = 4 * sigma2 - 4 * D * (EXPOSURE_TIME / 3)
    em_corrected   = em_alpha_fit.values - offset_em
    valid_em       = em_corrected > 0
    p, cov = np.polyfit(
        np.log(em_alpha_fit.index.values[valid_em]),
        np.log(em_corrected[valid_em]),
        1,
        full=False, cov=True
    )
    
    slope_em_corr = p[0]
    slop_err_corr = np.sqrt(cov[0,0])
     

    print(f"  Ensemble alpha (raw)       : {slope_em_raw:.3f}")
    print(f"  Ensemble alpha (corrected) : {slope_em_corr:.3f} ± {slop_err_corr:.3f}  (expected ≈ 1.0)")

    # ── Plot ──────────────────────────────────────────────────────────────────────────
    fig_alpha, ax = plt.subplots(figsize=(12, 5))

    ax.hist(alphas_corrected, bins=np.linspace(-0.5, 2.5, 60),
            color='teal', edgecolor='black', alpha=0.7,
            label='Per-trajectory α (corrected)')

    ax.axvline(1.0,           color='gray',   lw=1.5, linestyle=':',
               label='Perfect Brownian (α=1)')
    ax.axvline(slope_em_raw,  color='orange', lw=2,   linestyle='-.',
               label=f'Ensemble MSD α raw ({slope_em_raw:.3f})')
    ax.axvline(slope_em_corr, color='green',  lw=2,   linestyle='-',
               label=f'Ensemble MSD α corrected ({slope_em_corr:.3f})')

    ax.set_xlabel(r'Diffusive Exponent $\alpha$')
    ax.set_ylabel('Particle count')
    ax.set_title(f'Alpha distribution — {len(alphas_corrected)} trajectories')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig_alpha, "alpha_distribution_corrected", SAVE_DIR)
    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────


# ── Printing results ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 50)
    print('FINAL SUMMARY REPORT')
    print('=' * 50)
    print(f"{'Parameter':<25} | {'Value'}")
    print('-' * 50)
    print(f"{'Detected Temperature':<25} | {T_C:.2f}°C")
    print('-' * 50)
    print(f"{'Experimental D':<25} | {D:.4f} ± {perr[0]:.4f} µm²/s")
    print(f"{'Theoretical D (S-E)':<25} | {D_SE:.4f} ± {D_SE_err:.4f} µm²/s")
    print(f"{'Rel. Error vs Theory':<25} | {abs(D-D_SE)/D_SE*100:.2f}%")
    print('-' * 50)
    print(f"Localization σ (from MSD fit)     : {sigma*1000:.2f} nm")
    print(f"Localization σ (from static auto threshold)      : {threshold_px_as_um*1000:.2f} nm  (= {ratio:.2f}σ)")
    print('-' * 50)
    print(f"{'MSD Fit Lower cutoff':<25} | {tau_low:.4f} s | ≈ {low_cutoff_idx_em} lag times skipped")
    print(f"{'MSD Fit Higher cutoff':<25} | {cutoff_absolute:.2f} s")
    print(f"{'MSD Fit window':<25} | {(cutoff_absolute - tau_low):.4f}  s")
    print(f"  Points in window : {(cutoff_absolute - tau_low)/ FRAME_INTERVAL:.0f} lag times (from {low_cutoff_idx_em} to {cutoff_absolute_idx_em})")
    print(f"{'Final Particle Count':<25} | {after_flow}")
    print('-' * 50)
    print(f"{'Advised Search Range':<25} | {search_range(D, FRAME_INTERVAL, PIXEL_SIZE, P_MISS):.1f} px")
    print(f"{'Manual Search Range':<25} | {SEARCH_RANGE} px")
    check_density(f[f['frame'] == 0], SEARCH_RANGE)
    print('-' * 50)
    print(f"  Points used       : {len(em_fit_rough)} (indices 2 to {cutoff_absolute_idx_em})")
    print('=' * 50)


    # ── Saving data Call ───────────────────────────────────────────────────────
    if SAVE:
        save_parameters(
            SAVE_DIR, 
            D,               
            D_win_mean,      
            D_win_std, D_SE, sigma, perr, 
            before_f1, after_f1,
            before_f2, after_f2,
            before_static, after_static,
            before_alpha, after_alpha,
            before_flow, after_flow,
            auto_threshold, tau_low, cutoff_absolute,
            slope_em_raw, slope_em_corr,        
            cv_percent,                       
            T_C, avi_fps, total_duration
        )
        print(f"\nAll results and plots archived in: {SAVE_DIR}")
    # ───────────────────────────────────────────────────────────────────────────────────
    