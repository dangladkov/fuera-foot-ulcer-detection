#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_wounds_both.py

Color/priority (exclusive, one color per tile):
  1) WOUND CLUSTER        = #9900FF
  2) SUSPECT (contra_sus ∩ uni_sus per foot) = fuchsia #FF00FF
  3) BOTH TRIGGERS TODAY  = green  #00FF66
  4) CONTRALAT TRIGGER    = blue   #0066FF
  5) UNILAT TRIGGER       = cyan   #00FFFF

- All overlay alphas initialized to 1.0.
- Overlays drawn BEHIND the black grid so grid is always visible.
- Baseline rendering preserved: 'hot' heatmap, NaN background (no black), right foot mirrored,
  grid only on valid tiles, contralateral engine as baseline.

Terminal status stays verbose; patient-friendly messages written to patient_log.txt.
"""

import os, glob, warnings, random
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.io.matlab.miobase import MatReadWarning

# ==========================
# IO roots (baseline layout)
# ==========================
INPUT_ROOT         = "output_images_wound_modes_segmented"
DETECT_OUTPUT_ROOT = "output_images_detect_wounds"
PATIENT_FILTER = ""
VARIANT_FILTER = ""
RUNDIR_FILTER  = ""

# ==========================
# Visuals (match baseline)
# ==========================
CHUNK_PX        = 5
CONSISTENT_GRID = True

HEATMAP_CMAP = "hot"  # baseline

# ---- Colors (HEX) + alpha = 1.0 (tweakable later) ----
COLOR_UNI      = "#84D7D7"  # unilateral trigger (cyan)
ALPHA_UNI      = 1.0

COLOR_CONTRA   = "#0066FF"  # contralateral trigger (blue)
ALPHA_CONTRA   = 1.0

COLOR_BOTH     = "#00D015"  # both triggers today (green)
ALPHA_BOTH     = 1.0

COLOR_SUS      = "#9900FF"  # suspect (windowed, royal purple)
ALPHA_SUS      = 1.0

COLOR_WOUND    = "#FF00F7"  # wound clusters (fuchsia)
ALPHA_WOUND    = 1.0

GRID_COLOR     = "black"

# z-order so overlays are BEHIND the grid
Z_HEATMAP = 1
Z_OVERLAY = 3
Z_GRID    = 5

warnings.filterwarnings("ignore", category=MatReadWarning)

# Regions for patient-friendly messages
SEG_TOP_FRAC  = 0.65
SEG_HEEL_FRAC = 0.40

# ==========================
# Detection params
# ==========================
# Contralateral (baseline)
DIFF_THRESH_C    = 2.0
WINDOW_DAYS_C    = 5
MIN_TRIG_COUNT_C = 3
MIN_CLUSTER_SIZE = 3

# Unilateral
UNILAT_FOOT_MODE             = "both"  # "left" | "right" | "both"
UNILAT_TRAIN_DAYS            = 5
UNILAT_SIGMA_K               = 2.0
UNILAT_CONSECUTIVE_ANOMALIES = 3
RANDOMIZE_TRAINING_FOR_TESTING = True
RANDOM_SEED                  = 69420

# Robust unilateral baseline
UNILAT_ROBUST          = True
UNILAT_SIGMA_FLOOR     = 0.25
UNILAT_QUANTILE_BOUNDS = (0.05, 0.95)
UNILAT_WINSOR_FRAC     = 0.10

# ==========================
# Utils
# ==========================
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def discover_runs(input_root: str) -> List[Tuple[str, str, str]]:
    runs = []
    for patient in sorted(glob.glob(os.path.join(input_root, "*"))):
        if not os.path.isdir(patient): continue
        if PATIENT_FILTER and os.path.basename(patient) != PATIENT_FILTER: continue
        for variant in sorted(glob.glob(os.path.join(patient, "*"))):
            if not os.path.isdir(variant): continue
            if VARIANT_FILTER and os.path.basename(variant) != VARIANT_FILTER: continue
            for run_dir in sorted(glob.glob(os.path.join(variant, "*"))):
                if not os.path.isdir(run_dir): continue
                if RUNDIR_FILTER and os.path.basename(run_dir) != RUNDIR_FILTER: continue
                if glob.glob(os.path.join(run_dir, "mat", "*.mat")):
                    runs.append((os.path.basename(patient), os.path.basename(variant), run_dir))
    return runs

def load_day_mat(mat_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        md = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[WARN] Failed to load {mat_path}: {e}")
        return None, None

    def _find_key(d: dict, choices: Iterable[str]) -> Optional[np.ndarray]:
        for k in d.keys():
            kl = k.lower()
            for c in choices:
                if c in kl:
                    try:
                        arr = np.array(d[k], dtype=float)
                        if arr.ndim >= 2:
                            return np.nan_to_num(arr, nan=0.0).astype(float)
                    except Exception:
                        pass
        return None

    left  = _find_key(md, ["left_crop","left_foot","left"])
    right = _find_key(md, ["right_crop","right_foot","right"])

    if left is None or right is None:
        for k, v in md.items():
            if hasattr(v, "__dict__"):
                for kk, vv in v.__dict__.items():
                    if isinstance(vv, np.ndarray) and vv.ndim >= 2:
                        name = kk.lower()
                        if left is None and "left" in name:  left  = vv
                        if right is None and "right" in name: right = vv

    if left is None or right is None: return None, None

    left  = left.astype(float);  left[left  <= 0] = np.nan
    right = right.astype(float); right[right <= 0] = np.nan
    return left, right

def compute_boxes_and_indices(h: int, w: int, chunk_px: int):
    boxes, indices = [], []
    for y0 in range(0, h - chunk_px + 1, chunk_px):
        for x0 in range(0, w - chunk_px + 1, chunk_px):
            boxes.append((x0, y0, chunk_px, chunk_px))
            indices.append((x0 // chunk_px, y0 // chunk_px))
    return boxes, indices

def valid_tiles(img: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> Set[Tuple[int,int]]:
    mask = np.isfinite(img)
    val = set()
    for (x0,y0,ww,hh) in boxes:
        reg = mask[y0:y0+hh, x0:x0+ww]
        if reg.size and np.all(reg):
            val.add((x0//ww, y0//hh))
    return val

def components_4n(indices: Set[Tuple[int,int]]) -> List[Set[Tuple[int,int]]]:
    indices = set(indices)
    comps, dirs = [], [(1,0),(-1,0),(0,1),(0,-1)]
    while indices:
        start = indices.pop()
        stack, comp = [start], {start}
        while stack:
            x,y = stack.pop()
            for dx,dy in dirs:
                nb = (x+dx, y+dy)
                if nb in indices:
                    indices.remove(nb); comp.add(nb); stack.append(nb)
        comps.append(comp)
    return comps

def tile_means(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], valid_idx: Set[Tuple[int,int]]) -> Dict[Tuple[int,int], float]:
    means = {}
    for (x0,y0,w0,h0) in boxes:
        ij = (x0//w0, y0//h0)
        if ij not in valid_idx: continue
        region = img[y0:y0+h0, x0:x0+w0]
        if region.size == 0: continue
        means[ij] = float(np.nanmean(region))
    return means

def segment_labels_for_indices(indices: Iterable[Tuple[int,int]], foot_h: int, chunk_px: int) -> Set[str]:
    if not indices: return set()
    top_h = int(round(foot_h * SEG_TOP_FRAC))
    heel_h = int(round(top_h * SEG_HEEL_FRAC))
    labels = set()
    for (ix, iy) in indices:
        y_px = iy * chunk_px
        if y_px < heel_h: labels.add("heel")
        elif y_px < top_h: labels.add("mid-foot")
        else: labels.add("upper-foot")
    return labels

# ==========================
# CONTRALATERAL (baseline)
# ==========================
@dataclass
class ContraState:
    window: Dict[Tuple[int,int], deque]
    def __init__(self): self.window = defaultdict(lambda: deque(maxlen=WINDOW_DAYS_C))

def contralateral_check(left: np.ndarray, right: np.ndarray, chunk_px: int, state: ContraState):
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left_c, right_c = left[:h,:w], right[:h,:w]
    boxes, indices = compute_boxes_and_indices(h, w, chunk_px)

    mask_left, mask_right = np.isfinite(left_c), np.isfinite(right_c)
    valid_today = set()
    for (x0,y0,ww,hh), (ix,iy) in zip(boxes, indices):
        regL = mask_left[y0:y0+hh, x0:x0+ww]
        regR = mask_right[y0:y0+hh, x0:x0+ww]
        if regL.size and regR.size and np.all(regL) and np.all(regR):
            valid_today.add((ix,iy))

    means_left  = tile_means(left_c,  boxes, valid_today)
    means_right = tile_means(right_c, boxes, valid_today)

    triggers_today = set()
    for ij in valid_today:
        if ij in means_left and ij in means_right:
            if abs(means_left[ij] - means_right[ij]) >= DIFF_THRESH_C:
                triggers_today.add(ij)

    suspects = set()
    for ij in valid_today:
        dq = state.window[ij]
        dq.append(1 if ij in triggers_today else 0)
        if sum(dq) >= MIN_TRIG_COUNT_C:
            suspects.add(ij)

    wound_clusters = [c for c in components_4n(suspects) if len(c) >= MIN_CLUSTER_SIZE]
    return triggers_today, suspects, wound_clusters, boxes, (h, w)

# ==========================
# UNILATERAL (robust; per-foot)
# ==========================
@dataclass
class UnilatBaseline:
    stats: Dict[Tuple[int,int], Tuple[float,float,float,float]]  # (mu, sigma, lo, hi)
    valid_tiles: Set[Tuple[int,int]]
    def __init__(self):
        self.stats = {}
        self.valid_tiles = set()

@dataclass
class UnilatState:
    consec: Dict[Tuple[int,int], int]
    def __init__(self):
        self.consec = defaultdict(int)

def _winsorized(values: np.ndarray, frac: float) -> np.ndarray:
    if values.size == 0 or frac <= 0: return values
    lo = np.quantile(values, frac); hi = np.quantile(values, 1.0 - frac)
    return np.clip(values, lo, hi)

def _robust_stats(vals: List[float]) -> Optional[Tuple[float,float,float,float]]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0: return None
    if UNILAT_ROBUST:
        aw = _winsorized(a, UNILAT_WINSOR_FRAC) if UNILAT_WINSOR_FRAC > 0 else a
        mu  = float(np.median(aw))
        mad = float(np.median(np.abs(aw - mu)))
        sd  = float(max(1.4826 * mad, UNILAT_SIGMA_FLOOR))
        qlo, qhi = np.quantile(aw, UNILAT_QUANTILE_BOUNDS)
        lo, hi = float(qlo), float(qhi)
    else:
        mu = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        sd = float(max(sd, UNILAT_SIGMA_FLOOR if a.size > 1 else 0.0))
        lo, hi = float(np.min(a)), float(np.max(a))
    return mu, sd, lo, hi

def build_unilateral_baseline(frames: List[np.ndarray], chunk_px: int) -> UnilatBaseline:
    if not frames: return UnilatBaseline()
    h, w = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (h, w):
            h = min(h, f.shape[0]); w = min(w, f.shape[1])
    frames = [f[:h,:w] for f in frames]
    boxes, indices = compute_boxes_and_indices(h, w, chunk_px)

    valid_all = set()
    for (x0,y0,ww,hh), (ix,iy) in zip(boxes, indices):
        ok = True
        for f in frames:
            reg = f[y0:y0+hh, x0:x0+ww]
            if reg.size == 0 or not np.all(np.isfinite(reg)): ok = False; break
        if ok: valid_all.add((ix,iy))

    vals_per_tile = {ij: [] for ij in valid_all}
    for f in frames:
        means = tile_means(f, boxes, valid_all)
        for ij, m in means.items(): vals_per_tile[ij].append(m)

    baseline = UnilatBaseline()
    for ij, vals in vals_per_tile.items():
        r = _robust_stats(vals)
        if r is not None: baseline.stats[ij] = r
    baseline.valid_tiles = set(baseline.stats.keys())
    return baseline

def unilateral_check(foot_today: np.ndarray, baseline: UnilatBaseline, chunk_px: int, state: UnilatState):
    h, w = foot_today.shape[:2]
    boxes, indices = compute_boxes_and_indices(h, w, chunk_px)

    valid_today = set()
    mask = np.isfinite(foot_today)
    for (x0,y0,ww,hh), (ix,iy) in zip(boxes, indices):
        if (ix,iy) not in baseline.valid_tiles: continue
        reg = mask[y0:y0+hh, x0:x0+ww]
        if reg.size and np.all(reg): valid_today.add((ix,iy))

    means_today = tile_means(foot_today, boxes, valid_today)

    triggers_today = set()
    for ij in valid_today:
        if ij not in baseline.stats or ij not in means_today: continue
        mu, sd, lo, hi = baseline.stats[ij]
        x = means_today[ij]
        if (sd > 0 and abs(x - mu) > UNILAT_SIGMA_K * sd) or (x < lo) or (x > hi):
            triggers_today.add(ij)

    suspects = set()
    for ij in valid_today:
        state.consec[ij] = state.consec[ij] + 1 if ij in triggers_today else 0
        if state.consec[ij] >= UNILAT_CONSECUTIVE_ANOMALIES:
            suspects.add(ij)

    wound_clusters = [c for c in components_4n(suspects) if len(c) >= MIN_CLUSTER_SIZE]
    return triggers_today, suspects, wound_clusters, boxes, (h, w)

# ==========================
# Rendering (exclusive colors behind grid)
# ==========================
def render_day_png(out_png_path: str,
                   left: np.ndarray,
                   right: np.ndarray,
                   boxes: List[Tuple[int,int,int,int]],
                   left_valid_tiles: Set[Tuple[int,int]],
                   right_valid_tiles: Set[Tuple[int,int]],
                   gap_cols: int,
                   # category sets (exclusive already):
                   uni_left: Set[Tuple[int,int]],
                   uni_right: Set[Tuple[int,int]],
                   contra_left: Set[Tuple[int,int]],
                   contra_right: Set[Tuple[int,int]],
                   both_left: Set[Tuple[int,int]],
                   both_right: Set[Tuple[int,int]],
                   sus_left: Set[Tuple[int,int]],
                   sus_right: Set[Tuple[int,int]],
                   wound_tiles: Set[Tuple[int,int]],
                   chunk_px: int):
    left_v  = left.copy()
    right_v = right.copy()
    right_disp = np.fliplr(right_v)

    h = max(left_v.shape[0], right_v.shape[0])
    wL, wR = left_v.shape[1], right_v.shape[1]

    # NaN canvas so outside is transparent (no black)
    canvas = np.full((h, wL + gap_cols + wR), np.nan, dtype=float)
    canvas[:left_v.shape[0], :wL] = left_v
    canvas[:right_disp.shape[0], wL + gap_cols:wL + gap_cols + wR] = right_disp

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    im = ax.imshow(np.ma.masked_invalid(canvas), cmap=HEATMAP_CMAP, zorder=Z_HEATMAP)
    ax.set_facecolor("white")
    ax.axis("off")

    # Draw overlays BEHIND the grid, using linewidth=0 (no extra outlines).
    x_off = wL + gap_cols

    def _draw_set_left(S, color, alpha):
        for (ix,iy) in S:
            x0, y0 = ix*chunk_px, iy*chunk_px
            ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px,
                                   linewidth=0, edgecolor=color, facecolor=color,
                                   alpha=alpha, zorder=Z_OVERLAY))

    def _draw_set_right(S, color, alpha):
        for (ix,iy) in S:
            x0_m, y0 = x_off + (wR - (ix+1)*chunk_px), iy*chunk_px
            ax.add_patch(Rectangle((x0_m, y0), chunk_px, chunk_px,
                                   linewidth=0, edgecolor=color, facecolor=color,
                                   alpha=alpha, zorder=Z_OVERLAY))

    # Order doesn't matter now because sets are exclusive, but we keep wound>sus>both>contra>uni
    _draw_set_left (uni_left,    COLOR_UNI,    ALPHA_UNI)
    _draw_set_right(uni_right,   COLOR_UNI,    ALPHA_UNI)

    _draw_set_left (contra_left, COLOR_CONTRA, ALPHA_CONTRA)
    _draw_set_right(contra_right,COLOR_CONTRA, ALPHA_CONTRA)

    _draw_set_left (both_left,   COLOR_BOTH,   ALPHA_BOTH)
    _draw_set_right(both_right,  COLOR_BOTH,   ALPHA_BOTH)

    _draw_set_left (sus_left,    COLOR_SUS,    ALPHA_SUS)
    _draw_set_right(sus_right,   COLOR_SUS,    ALPHA_SUS)

    # wound tiles are drawn on BOTH feet
    _draw_set_left (wound_tiles, COLOR_WOUND,  ALPHA_WOUND)
    _draw_set_right(wound_tiles, COLOR_WOUND,  ALPHA_WOUND)

    # Grid LAST so it's on top of colors
    for (ix,iy) in left_valid_tiles:
        x0, y0 = ix*chunk_px, iy*chunk_px
        ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px, linewidth=0.7,
                               edgecolor=GRID_COLOR, facecolor='none', zorder=Z_GRID))
    for (ix,iy) in right_valid_tiles:
        x0_m, y0 = x_off + (wR - (ix+1)*chunk_px), iy*chunk_px
        ax.add_patch(Rectangle((x0_m, y0), chunk_px, chunk_px, linewidth=0.7,
                               edgecolor=GRID_COLOR, facecolor='none', zorder=Z_GRID))

    # Colorbar with integer tick labels (no decimals)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Temperature (°C)")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    cbar.locator = MaxNLocator(integer=True); cbar.update_ticks()

    plt.savefig(out_png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def patient_log_line(fp: str, text: str):
    with open(fp, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def log_status_terminal(text: str): print(text)

# ==========================
# Main per-run
# ==========================
@dataclass
class ContraState:
    window: Dict[Tuple[int,int], deque]
    def __init__(self): self.window = defaultdict(lambda: deque(maxlen=WINDOW_DAYS_C))

def _directional_hot_sets_for_today(left_img: np.ndarray,
                                    right_img: np.ndarray,
                                    boxes: List[Tuple[int,int,int,int]]) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    """
    Returns (left_hot_today, right_hot_today) sets of (ix,iy) where today's left>right or right>left.
    This is used ONLY for patient_log phrasing to list independent 'hotter' areas.
    """
    left_hot_today, right_hot_today = set(), set()
    # valid if both feet have finite data for the tile
    maskL, maskR = np.isfinite(left_img), np.isfinite(right_img)
    valid = set()
    for (x0,y0,ww,hh) in boxes:
        subL = maskL[y0:y0+hh, x0:x0+ww]
        subR = maskR[y0:y0+hh, x0:x0+ww]
        if subL.size and subR.size and np.all(subL) and np.all(subR):
            valid.add((x0//ww, y0//hh))

    meansL = tile_means(left_img,  boxes, valid)
    meansR = tile_means(right_img, boxes, valid)

    for ij in valid:
        l = meansL.get(ij, np.nan); r = meansR.get(ij, np.nan)
        if not np.isfinite(l) or not np.isfinite(r): continue
        if l > r:
            left_hot_today.add(ij)
        elif r > l:
            right_hot_today.add(ij)
    return left_hot_today, right_hot_today

def run_on_directory(patient: str, variant: str, run_dir: str):
    mat_dir = os.path.join(run_dir, "mat")
    mat_files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    if not mat_files:
        print(f"[INFO] No .mat files in {mat_dir}"); return

    out_root   = os.path.join(DETECT_OUTPUT_ROOT, patient, variant, os.path.basename(run_dir))
    out_png    = os.path.join(out_root, "png")
    out_log    = os.path.join(out_root, "log")
    ensure_dir(out_png); ensure_dir(out_log)
    patient_log_fp = os.path.join(out_log, "patient_log.txt")
    with open(patient_log_fp, "w", encoding="utf-8"): pass  # reset

    contra_state = ContraState()
    uni_state = {"left": UnilatState(), "right": UnilatState()}
    uni_base  = {"left": UnilatBaseline(), "right": UnilatBaseline()}
    baseline_built = False

    days = [load_day_mat(p) for p in mat_files]
    valid_indices = [i for i,(L,R) in enumerate(days) if (L is not None and R is not None)]
    if not valid_indices:
        patient_log_line(patient_log_fp, "We couldn't read enough images to start. Please try again.")
        print("No valid days with left+right data."); return

    left0, right0 = days[valid_indices[0]]
    H = min(left0.shape[0], right0.shape[0]); W = min(left0.shape[1], right0.shape[1])
    gap_cols = max(10, W // 20)

    # training selection (optional randomization kept)
    train_pool = [i for i in valid_indices if i < valid_indices[0] + 10]
    if RANDOMIZE_TRAINING_FOR_TESTING and len(train_pool) >= UNILAT_TRAIN_DAYS:
        random.seed(RANDOM_SEED)
        train_indices = sorted(random.sample(train_pool, k=UNILAT_TRAIN_DAYS))
    else:
        train_indices = [i for i in valid_indices[:UNILAT_TRAIN_DAYS]]

    processed_count = 0
    wound_found = False  # <-- for closing line at the end

    for day_i, (L_raw, R_raw) in enumerate(days, start=1):
        if L_raw is None or R_raw is None:
            log_status_terminal(f"[DAY {day_i:02d}] Missing data, skipping.")
            continue

        L, R = L_raw[:H,:W], R_raw[:H,:W]

        # --- Contralateral (first)
        contra_trig, contra_sus, _, boxes, _ = contralateral_check(L, R, CHUNK_PX, contra_state)

        # valid tiles for tight grid
        left_valid_tiles  = valid_tiles(L, boxes)
        right_valid_tiles = valid_tiles(R, boxes)

        # --- Build unilateral baselines after 5 processed days
        processed_count += 1
        if (not baseline_built) and (processed_count == UNILAT_TRAIN_DAYS):
            L_train = [days[ti][0][:H,:W] for ti in train_indices if days[ti][0] is not None]
            R_train = [days[ti][1][:H,:W] for ti in train_indices if days[ti][1] is not None]
            if UNILAT_FOOT_MODE in ("left","both"):  uni_base["left"]  = build_unilateral_baseline(L_train, CHUNK_PX)
            if UNILAT_FOOT_MODE in ("right","both"): uni_base["right"] = build_unilateral_baseline(R_train, CHUNK_PX)
            baseline_built = True
            # NOTE: terminal message preserved; we no longer log a patient line here to avoid duplicating Day 05.

        # --- Unilateral (second)
        uni_trig_L=set(); uni_sus_L=set()
        uni_trig_R=set(); uni_sus_R=set()
        if baseline_built:
            if uni_base["left"].stats:
                uni_trig_L, uni_sus_L, _, _, _ = unilateral_check(L, uni_base["left"], CHUNK_PX, uni_state["left"])
            if uni_base["right"].stats:
                uni_trig_R, uni_sus_R, _, _, _ = unilateral_check(R, uni_base["right"], CHUNK_PX, uni_state["right"])

        # --- True suspects per foot (windowed)
        true_sus_left  = contra_sus.intersection(uni_sus_L) if baseline_built else set()
        true_sus_right = contra_sus.intersection(uni_sus_R) if baseline_built else set()

        # --- Wound clusters from union of true suspects
        wound_union = true_sus_left | true_sus_right
        wound_clusters_union = [c for c in components_4n(wound_union) if len(c) >= MIN_CLUSTER_SIZE]
        wound_tiles = set().union(*wound_clusters_union) if wound_clusters_union else set()

        # --- Today's BOTH triggers per foot
        both_L = contra_trig & uni_trig_L
        both_R = contra_trig & uni_trig_R

        # --- Build EXCLUSIVE sets for drawing (unchanged visuals)
        uni_left    = set(uni_trig_L)
        contra_left = set(contra_trig)
        both_left   = set(both_L)
        uni_right    = set(uni_trig_R)
        contra_right = set(contra_trig)
        both_right   = set(both_R)

        uni_left    -= both_left;  contra_left -= both_left
        uni_right   -= both_right; contra_right-= both_right

        if baseline_built:
            for S in (uni_left, contra_left, both_left):
                S -= (true_sus_left | wound_tiles)
            for S in (uni_right, contra_right, both_right):
                S -= (true_sus_right | wound_tiles)

        sus_left  = set(true_sus_left)  - wound_tiles
        sus_right = set(true_sus_right) - wound_tiles

        # --- Render (unchanged)
        out_png_path = os.path.join(out_png, f"wound_detect_day{day_i:02d}.png")
        render_day_png(out_png_path, L, R, boxes,
                       left_valid_tiles, right_valid_tiles, gap_cols,
                       uni_left=uni_left, uni_right=uni_right,
                       contra_left=contra_left, contra_right=contra_right,
                       both_left=both_left, both_right=both_right,
                       sus_left=sus_left, sus_right=sus_right,
                       wound_tiles=wound_tiles,
                       chunk_px=CHUNK_PX)

        # ============================================================
        #                PATIENT LOG  (ONLY SECTION CHANGED)
        # ============================================================

        # Days 1–5: always training message; do not add other lines for these days
        if day_i <= UNILAT_TRAIN_DAYS:
            patient_log_line(patient_log_fp, f"Day {day_i:02d}: We're learning your normal temperature pattern. No concerns today.")
        else:
            # Direction of 'hotter than comparator' for today's copy (independent abnormal areas)
            left_hot_today, right_hot_today = _directional_hot_sets_for_today(L, R, boxes)

            if wound_clusters_union:
                # WOUND: list only sides that are hotter than their comparator today
                wound_left  = wound_tiles & left_hot_today
                wound_right = wound_tiles & right_hot_today

                # Fallback: if direction is ambiguous today, fall back to true-sus per-foot
                if not wound_left and not wound_right:
                    wound_left  = wound_tiles & true_sus_left
                    wound_right = wound_tiles & true_sus_right

                msg_parts = []
                if wound_left:
                    segs = sorted(segment_labels_for_indices(wound_left, H, CHUNK_PX))
                    msg_parts.append(f"Left foot ({', '.join(segs)})")
                if wound_right:
                    segs = sorted(segment_labels_for_indices(wound_right, H, CHUNK_PX))
                    msg_parts.append(f"Right foot ({', '.join(segs)})")

                where_txt = " and ".join(msg_parts) if msg_parts else "foot area"
                patient_log_line(patient_log_fp, f"Day {day_i:02d}: We see signs that may indicate a wound on your {where_txt}.")
                # Closing line for wound case
                patient_log_line(patient_log_fp, "Detection finished. Please contact your clinician as soon as possible to assess the area.")
                wound_found = True
                # (terminal messages + break stay unchanged)
                log_status_terminal((f"[DAY {day_i:02d}] WOUND DETECTED by AND | "
                                     f"Contra: trig={len(contra_trig)}, sus={len(contra_sus)} | "
                                     f"UniL: trig={len(uni_trig_L)}, sus={len(uni_sus_L)} | "
                                     f"UniR: trig={len(uni_trig_R)}, sus={len(uni_sus_R)} | "
                                     f"AND clusters={len(wound_clusters_union)} | "
                                     f"PNG -> {os.path.basename(out_png_path)}"))
                break

            else:
                # WATCH status: list only independent hot areas (per foot) among TRUE SUSPECTS
                watch_left  = true_sus_left  & left_hot_today
                watch_right = true_sus_right & right_hot_today

                if watch_left or watch_right:
                    parts = []
                    if watch_left:
                        segs = sorted(segment_labels_for_indices(watch_left, H, CHUNK_PX))
                        parts.append(f"Left foot ({', '.join(segs)})")
                    if watch_right:
                        segs = sorted(segment_labels_for_indices(watch_right, H, CHUNK_PX))
                        parts.append(f"Right foot ({', '.join(segs)})")
                    patient_log_line(patient_log_fp, f"Day {day_i:02d}: We’re watching a warm area on your {' and '.join(parts)}.")
                else:
                    patient_log_line(patient_log_fp, f"Day {day_i:02d}: No suspicious hotspots today.")

        # ============================================================
        #            END PATIENT LOG SECTION (terminal unchanged)
        # ============================================================

        # Terminal status (unchanged)
        if not baseline_built:
            log_status_terminal((f"[DAY {day_i:02d}] Training (contra only) | "
                                 f"Contra: trig={len(contra_trig)}, sus(win{WINDOW_DAYS_C})={len(contra_sus)} | "
                                 f"PNG -> {os.path.basename(out_png_path)}"))
        else:
            log_status_terminal((f"[DAY {day_i:02d}] No AND cluster (≥{MIN_CLUSTER_SIZE}) | "
                                 f"Contra: trig={len(contra_trig)}, sus={len(contra_sus)} | "
                                 f"UniL: trig={len(uni_trig_L)}, sus={len(uni_sus_L)} | "
                                 f"UniR: trig={len(uni_trig_R)}, sus={len(uni_sus_R)} | "
                                 f"PNG -> {os.path.basename(out_png_path)}"))

    # Final closing line if we finished the loop without finding a wound
    if not wound_found:
        patient_log_line(patient_log_fp, "Detection finished. No wound was detected in this run. Keep monitoring daily, and contact your clinician if you notice any changes.")


def main():
    runs = discover_runs(INPUT_ROOT)
    if not runs:
        print(f"[INFO] No runs found under {INPUT_ROOT}."); return
    for (patient, variant, run_dir) in runs:
        print(f"[RUN] {patient}/{variant}/{os.path.basename(run_dir)}")
        run_on_directory(patient, variant, run_dir)

if __name__ == "__main__":
    main()
