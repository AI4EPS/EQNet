"""
metrics.py — Unified seismic phase and event detection evaluation.

Sections:
    A: Constants & CLI argument parsing
    B: extract_gt_from_h5 — H5 to CSV extraction
    C: load_gt & load_predictions — Data loading
    D: match_detections, filter_predictions_main_event, cluster_and_match_events
    E: compute_metrics, compute_metrics_vs_param, sweep_thresholds
    F: Plotting functions
    G: CLI main
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Constants & CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 100
FEATURE_SCALE = 16
DEGREE2KM = 111.32
EARTH_RADIUS_KM = 6371.0
DEFAULT_PICK_DT_BAR = 0.5        # seconds
DEFAULT_EVENT_DT_BAR = 3.0       # seconds
DEFAULT_EVENT_CENTER_DT_BAR = 1.0 # seconds
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_POLARITY_THRESHOLD = 0.3
DEFAULT_DBSCAN_EPSILON = 16
DEFAULT_MIN_STATIONS = 3
DEFAULT_STATION_THRESHOLD = 10
DEFAULT_TIME_SPACE_WEIGHT = [1, 0.8, 0.8, 2]
VAPP = 8.2   # km/s, apparent P velocity for estimating missing arrivals
MAX_TRACE_SAMPLES = 12000  # hard-coded trace length in samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Seismic phase picking and event detection metrics",
        add_help=True,
    )
    # --- Input paths ---
    parser.add_argument("--h5_file", type=str, default=None,
                        help="H5 file path (for GT extraction)")
    parser.add_argument("--gt_csv_dir", type=str, default=None,
                        help="Directory with gt_events.csv and gt_picks.csv")
    parser.add_argument("--pick_dir", type=str, default=None,
                        help="Directory with prediction pick CSVs")
    parser.add_argument("--event_dir", type=str, default=None,
                        help="Directory with prediction event CSVs")
    # --- Output ---
    parser.add_argument("--save_path", type=str, default="./metrics",
                        help="Output directory for results")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output filenames")
    # --- Model/format ---
    parser.add_argument("--model", type=str, default="eqnet",
                        choices=["eqnet", "phasenet_plus"],
                        help="Model that produced predictions")
    # --- Thresholds ---
    parser.add_argument("--score_threshold", type=float,
                        default=DEFAULT_SCORE_THRESHOLD,
                        help="Phase/event score threshold")
    parser.add_argument("--polarity_threshold", type=float,
                        default=DEFAULT_POLARITY_THRESHOLD,
                        help="Polarity score threshold for D/N/U mapping")
    parser.add_argument("--pick_dt_bar", type=float,
                        default=DEFAULT_PICK_DT_BAR,
                        help="Max residual (s) for pick TP")
    parser.add_argument("--event_dt_bar", type=float,
                        default=DEFAULT_EVENT_DT_BAR,
                        help="Max residual (s) for clustered event TP")
    parser.add_argument("--event_center_dt_bar", type=float,
                        default=DEFAULT_EVENT_CENTER_DT_BAR,
                        help="Max residual (s) for event center TP")
    parser.add_argument("--distribution_bar", type=float,
                        default=DEFAULT_PICK_DT_BAR,
                        help="Residual window for distribution statistics")
    # --- Matching ---
    parser.add_argument("--match_by", type=str, default="index",
                        choices=["index", "time"],
                        help="Match by sample index or absolute time")
    parser.add_argument("--max_residual_samples", type=float, default=500,
                        help="Max residual in samples for matching window")
    # --- Clustering ---
    parser.add_argument("--association_method", type=str, default="dbscan",
                        choices=["dbscan", "associator"],
                        help="Event association method")
    parser.add_argument("--dbscan_epsilon", type=float,
                        default=DEFAULT_DBSCAN_EPSILON)
    parser.add_argument("--min_stations", type=int,
                        default=DEFAULT_MIN_STATIONS)
    parser.add_argument("--time_space_weight", type=float, nargs=4,
                        default=DEFAULT_TIME_SPACE_WEIGHT)
    # --- Filters ---
    parser.add_argument("--main_event_only", action="store_true",
                        help="Filter predictions to main event duration only")
    parser.add_argument("--filter_events", type=str, default=None,
                        help="JSON file with event IDs to filter")
    parser.add_argument("--include_events", action="store_true",
                        help="Include (not exclude) the filtered events")
    parser.add_argument("--station_threshold", type=int,
                        default=DEFAULT_STATION_THRESHOLD,
                        help="Min stations for station-count subset metrics")
    # --- Threshold sweep ---
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Run threshold sweep for PR curves")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                 0.85, 0.9])
    # --- Mode ---
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract GT from H5, then exit")

    args = parser.parse_args(argv)
    if args.suffix and not args.suffix.startswith("_"):
        args.suffix = f"_{args.suffix}"
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Helpers & extract_gt_from_h5
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_time(s):
    """Parse ISO time string to datetime."""
    return datetime.fromisoformat(str(s))


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (degrees)."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return float(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * EARTH_RADIUS_KM)


def haversine_azimuth_deg(sta_lat, sta_lon, evt_lat, evt_lon):
    """Azimuth in degrees from station to event (bearing), 0-360."""
    dlon = np.radians(evt_lon - sta_lon)
    y = np.sin(dlon) * np.cos(np.radians(evt_lat))
    x = (np.cos(np.radians(sta_lat)) * np.sin(np.radians(evt_lat))
         - np.sin(np.radians(sta_lat)) * np.cos(np.radians(evt_lat))
         * np.cos(dlon))
    return float(np.degrees(np.arctan2(y, x)) % 360)


def polarity_label(score, threshold):
    """Map continuous polarity score to D/N/U."""
    if score < -threshold:
        return "D"
    elif score > threshold:
        return "U"
    return "N"


def extract_gt_from_h5(h5_path, output_dir):
    """Extract ground truth metadata from H5 to CSV files.

    Produces:
        gt_events.csv — one row per pick_event_id
        gt_picks.csv  — one row per (event_id, station_id, pick_event_id,
                         phase_type) where phase_type is "P", "S", or "event"
    """
    os.makedirs(output_dir, exist_ok=True)

    event_rows = {}   # keyed by pick_event_id
    pick_rows = []

    with h5py.File(h5_path, "r") as fp:
        event_ids = list(fp.keys())
        for event_id in tqdm(event_ids, desc="Extracting GT"):
            event = fp[event_id]
            event_attrs = dict(event.attrs)

            main_event_time_str = event_attrs.get("event_time", None)
            main_event_time_index = event_attrs.get("event_time_index", None)
            if main_event_time_str is None or main_event_time_index is None:
                continue
            main_event_time_index = int(main_event_time_index)
            main_event_time = _parse_time(main_event_time_str)

            # Register main event in gt_events
            if event_id not in event_rows:
                event_rows[event_id] = {
                    "pick_event_id": event_id,
                    "latitude": float(event_attrs.get("latitude", np.nan)),
                    "longitude": float(event_attrs.get("longitude", np.nan)),
                    "depth_km": float(event_attrs.get("depth_km", np.nan)),
                    "magnitude": float(event_attrs.get("magnitude", np.nan)),
                    "event_time": str(main_event_time_str),
                    "event_time_index": main_event_time_index,
                    "num_stations": len(list(event.keys())),
                }

            station_ids = list(event.keys())
            for sta_id in station_ids:
                sta = event[sta_id]
                attrs = dict(sta.attrs)

                phase_type_arr = np.array(attrs.get("phase_type", []),
                                          dtype=str)
                phase_index_arr = np.array(attrs.get("phase_index", []),
                                           dtype=float)
                pick_event_id_arr = np.array(attrs.get("event_id", []),
                                             dtype=str)
                raw_pol = attrs.get("phase_polarity", None)
                if raw_pol is not None:
                    phase_polarity_arr = np.array(raw_pol, dtype=str)
                else:
                    phase_polarity_arr = np.array(
                        ["N"] * len(phase_type_arr), dtype=str
                    )
                dt_s = float(attrs.get("dt_s", 1.0 / SAMPLE_RATE))
                snr_arr = np.array(attrs.get("snr", [0.0]), dtype=float)
                snr_pos = snr_arr[snr_arr > 0]
                snr_max = float(snr_pos.max()) if len(snr_pos) > 0 else 0.0
                sta_lat = float(attrs.get("latitude", np.nan))
                sta_lon = float(attrs.get("longitude", np.nan))
                elevation_m = float(attrs.get("elevation_m", 0.0))

                # Detect automatic-only stations (kept in CSV but marked non-manual)
                phase_status = attrs.get("phase_status", None)
                is_automatic = (phase_status is not None
                                and str(phase_status) == "automatic")

                # Trace start time
                trace_start_time = (
                    main_event_time
                    - timedelta(seconds=main_event_time_index * dt_s)
                )

                # Group by pick_event_id
                unique_peids = sorted(set(pick_event_id_arr))
                for peid in unique_peids:
                    mask = pick_event_id_arr == peid
                    types = phase_type_arr[mask]
                    indices = phase_index_arr[mask]
                    polarities = phase_polarity_arr[mask]

                    has_p = np.any(types == "P")
                    has_s = np.any(types == "S")
                    is_manual = bool(has_p and has_s and not is_automatic)

                    # Look up pick_event metadata
                    try:
                        if peid == event_id:
                            pe_attrs = event_attrs
                        else:
                            pe_attrs = dict(fp[peid].attrs)
                        pe_lat = float(pe_attrs.get("latitude", np.nan))
                        pe_lon = float(pe_attrs.get("longitude", np.nan))
                        pe_depth = float(pe_attrs.get("depth_km", np.nan))
                        pe_mag = float(pe_attrs.get("magnitude", np.nan))
                        pe_time_str = pe_attrs.get("event_time", None)
                        pe_time_index = pe_attrs.get("event_time_index", None)
                        if pe_time_index is not None:
                            pe_time_index = int(pe_time_index)
                    except KeyError:
                        pe_lat = pe_lon = pe_depth = pe_mag = np.nan
                        pe_time_str = None
                        pe_time_index = None

                    # Register pick_event in gt_events if new
                    if peid not in event_rows and pe_time_str is not None:
                        try:
                            pe_num_stas = len(list(fp[peid].keys()))
                        except KeyError:
                            pe_num_stas = 0
                        event_rows[peid] = {
                            "pick_event_id": peid,
                            "latitude": pe_lat,
                            "longitude": pe_lon,
                            "depth_km": pe_depth,
                            "magnitude": pe_mag,
                            "event_time": str(pe_time_str),
                            "event_time_index": pe_time_index,
                            "num_stations": pe_num_stas,
                        }

                    # begin_time_index_shift for secondary events
                    if peid != event_id and pe_time_str is not None:
                        pe_time = _parse_time(pe_time_str)
                        begin_time_shift = int(round(
                            (pe_time - main_event_time).total_seconds() / dt_s
                        ))
                    else:
                        begin_time_shift = 0

                    # Distance and azimuth from station to pick_event
                    if not (np.isnan(pe_lat) or np.isnan(pe_lon)
                            or np.isnan(sta_lat) or np.isnan(sta_lon)):
                        dist_km = haversine_distance_km(
                            sta_lat, sta_lon, pe_lat, pe_lon)
                        azi_deg = haversine_azimuth_deg(
                            sta_lat, sta_lon, pe_lat, pe_lon)
                    else:
                        dist_km = float(attrs.get("distance_km", np.nan))
                        azi_deg = float(attrs.get("azimuth", np.nan))

                    # Shared row template
                    base = {
                        "event_id": event_id,
                        "station_id": sta_id,
                        "pick_event_id": peid,
                        "is_manual": is_manual,
                        "snr_max": snr_max,
                        "station_latitude": sta_lat,
                        "station_longitude": sta_lon,
                        "elevation_m": elevation_m,
                        "dt_s": dt_s,
                        "event_magnitude": pe_mag,
                        "event_latitude": pe_lat,
                        "event_longitude": pe_lon,
                        "event_depth_km": pe_depth,
                        "distance_km": dist_km,
                        "azimuth_deg": azi_deg,
                    }

                    # Emit P and S rows
                    for pt in ("P", "S"):
                        pt_mask = types == pt
                        if not np.any(pt_mask):
                            continue
                        for idx, pol in zip(indices[pt_mask],
                                            polarities[pt_mask]):
                            phase_time = (
                                trace_start_time
                                + timedelta(seconds=float(idx) * dt_s)
                            ).isoformat(timespec="milliseconds")
                            row = base.copy()
                            row.update({
                                "phase_type": pt,
                                "phase_index": float(idx),
                                "phase_time": phase_time,
                                "phase_polarity": str(pol),
                                "travel_time_s": np.nan,
                            })
                            pick_rows.append(row)

                    # Emit event center rows (only if both P and S)
                    if is_manual:
                        p_idx = np.sort(indices[types == "P"])
                        s_idx = np.sort(indices[types == "S"])
                        n_pairs = min(len(p_idx), len(s_idx))
                        for k in range(n_pairs):
                            center = (p_idx[k] + s_idx[k]) / 2.0
                            if pe_time_index is not None:
                                t0 = pe_time_index + begin_time_shift
                                tt_s = float((center - t0) * dt_s)
                            else:
                                tt_s = np.nan
                            phase_time = (
                                trace_start_time
                                + timedelta(seconds=float(center) * dt_s)
                            ).isoformat(timespec="milliseconds")
                            row = base.copy()
                            row.update({
                                "phase_type": "event",
                                "phase_index": center,
                                "phase_time": phase_time,
                                "phase_polarity": "N",
                                "travel_time_s": tt_s,
                            })
                            pick_rows.append(row)

    gt_events_df = pd.DataFrame(list(event_rows.values()))
    gt_picks_df = pd.DataFrame(pick_rows)

    gt_events_path = os.path.join(output_dir, "gt_events.csv")
    gt_picks_path = os.path.join(output_dir, "gt_picks.csv")
    gt_events_df.to_csv(gt_events_path, index=False)
    gt_picks_df.to_csv(gt_picks_path, index=False)

    print(f"Extracted {len(gt_events_df)} events, {len(gt_picks_df)} picks")
    print(f"  gt_events.csv: {gt_events_path}")
    print(f"  gt_picks.csv:  {gt_picks_path}")
    return gt_events_df, gt_picks_df


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: load_gt & load_predictions
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt(gt_dir):
    """Load GT CSVs into DataFrames."""
    gt_events = pd.read_csv(os.path.join(gt_dir, "gt_events.csv"))
    gt_picks = pd.read_csv(os.path.join(gt_dir, "gt_picks.csv"))
    return gt_events, gt_picks


def _load_pred_csvs(directory, model):
    """Collect prediction CSV file paths for a given model format."""
    if model == "eqnet":
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory) if f.endswith(".csv")
        ]
    elif model == "phasenet_plus":
        paths = []
        for d in os.listdir(directory):
            dp = os.path.join(directory, d)
            if os.path.isdir(dp):
                for f in os.listdir(dp):
                    if f.endswith(".csv"):
                        paths.append(os.path.join(dp, f))
        return paths
    return []


def load_predictions(pick_dir, event_dir, model):
    """Load all prediction CSVs into unified DataFrames.

    Returns (pick_preds_df, event_preds_df).
    Both have an 'event_id' column derived from the CSV filename.
    """
    pick_preds = _load_pick_preds(pick_dir, model) if pick_dir else pd.DataFrame()
    event_preds = _load_event_preds(event_dir, model) if event_dir else pd.DataFrame()
    return pick_preds, event_preds


def _load_pick_preds(pick_dir, model):
    """Load pick prediction CSVs into a single DataFrame."""
    frames = []
    if model == "eqnet":
        for f in os.listdir(pick_dir):
            if not f.endswith(".csv"):
                continue
            fpath = os.path.join(pick_dir, f)
            if os.path.getsize(fpath) == 0:
                continue
            event_id = re.sub(r"\.csv$", "", f)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            df["event_id"] = event_id
            frames.append(df)
    elif model == "phasenet_plus":
        for d in os.listdir(pick_dir):
            dp = os.path.join(pick_dir, d)
            if not os.path.isdir(dp):
                continue
            event_id = d
            for f in os.listdir(dp):
                if not f.endswith(".csv"):
                    continue
                fpath = os.path.join(dp, f)
                if os.path.getsize(fpath) == 0:
                    continue
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
                df["event_id"] = event_id
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # Normalize columns
    if "phase_polarity" not in result.columns:
        result["phase_polarity"] = 0.0
    return result


def _load_event_preds(event_dir, model):
    """Load event prediction CSVs into a single DataFrame."""
    frames = []
    if model == "eqnet":
        for f in os.listdir(event_dir):
            if not f.endswith(".csv"):
                continue
            fpath = os.path.join(event_dir, f)
            if os.path.getsize(fpath) == 0:
                continue
            event_id = re.sub(r"\.csv$", "", f)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if "event_id" in df.columns:
                df = df.rename(columns={"event_id": "event_id_pred"})
            df["event_id"] = event_id
            frames.append(df)
    elif model == "phasenet_plus":
        for d in os.listdir(event_dir):
            dp = os.path.join(event_dir, d)
            if not os.path.isdir(dp):
                continue
            event_id = d
            sub_frames = []
            for f in os.listdir(dp):
                if not f.endswith(".csv"):
                    continue
                fpath = os.path.join(dp, f)
                if os.path.getsize(fpath) == 0:
                    continue
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
                sub_frames.append(df)
            if sub_frames:
                df = pd.concat(sub_frames, ignore_index=True)
                df["event_id"] = event_id
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Matching functions
# ═══════════════════════════════════════════════════════════════════════════════

def _match_sorted(gt_vals, pred_vals, max_residual):
    """Greedy nearest-neighbor matching on sorted arrays.

    Both gt_vals and pred_vals must be sorted ascending.
    Returns list of (gt_idx | None, pred_idx | None, residual | NaN).
    """
    matches = []
    used_pred = set()

    for g_idx in range(len(gt_vals)):
        g_val = gt_vals[g_idx]
        lo = bisect_left(pred_vals, g_val - max_residual)
        hi = bisect_right(pred_vals, g_val + max_residual)
        best_p, best_res = None, max_residual + 1
        for p_idx in range(lo, hi):
            if p_idx in used_pred:
                continue
            res = abs(pred_vals[p_idx] - g_val)
            if res < best_res:
                best_res = res
                best_p = p_idx
        if best_p is not None:
            used_pred.add(best_p)
            matches.append((g_idx, best_p, pred_vals[best_p] - g_val))
        else:
            matches.append((g_idx, None, np.nan))

    # Remaining unmatched predictions
    for p_idx in range(len(pred_vals)):
        if p_idx not in used_pred:
            matches.append((None, p_idx, np.nan))

    return matches


def match_detections(pred_df, gt_df, phase_types, max_residual_s,
                     match_by="index", sample_rate=SAMPLE_RATE,
                     polarity_threshold=DEFAULT_POLARITY_THRESHOLD):
    """Unified matching for picks and event centers.

    Parameters
    ----------
    pred_df : DataFrame
        Predictions with columns: event_id, station_id, phase_index,
        phase_time, phase_score, phase_type, phase_polarity.
    gt_df : DataFrame
        Ground truth (gt_picks.csv subset) with columns: event_id, station_id,
        pick_event_id, phase_type, phase_index, phase_time, ...
    phase_types : list of str
        Phase types to match, e.g. ["P", "S"] or ["event"].
    max_residual_s : float
        Maximum residual in seconds for matching window.
    match_by : str
        "index" (match phase_index in sample units, grouped by event_id+station_id)
        or "time" (match phase_time, grouped by station_id).
    sample_rate : float
        Sample rate for index-to-seconds conversion.
    polarity_threshold : float
        Threshold for polarity D/N/U mapping.

    Returns
    -------
    matched_df : DataFrame
        Rows for tp_candidate, unmatched_gt, and unmatched_pred.
    """
    if pred_df.empty and gt_df.empty:
        return pd.DataFrame()

    # Filter to requested phase types
    if not gt_df.empty and "phase_type" in gt_df.columns:
        gt_sub = gt_df[gt_df["phase_type"].isin(phase_types)].copy()
    else:
        gt_sub = pd.DataFrame()
    if not pred_df.empty and "phase_type" in pred_df.columns:
        pred_sub = pred_df[pred_df["phase_type"].isin(phase_types)].copy()
    else:
        pred_sub = pd.DataFrame()

    # Filter to manual GT only (is_manual == True)
    if not gt_sub.empty and "is_manual" in gt_sub.columns:
        gt_sub = gt_sub[gt_sub["is_manual"] == True].copy()

    if gt_sub.empty and pred_sub.empty:
        return pd.DataFrame()

    # Determine grouping and value columns
    if match_by == "index":
        group_cols = ["event_id", "station_id", "phase_type"]
        val_col_gt = "phase_index"
        val_col_pred = "phase_index"
        max_residual_val = max_residual_s * sample_rate  # convert to samples
        residual_to_s = 1.0 / sample_rate
    else:  # "time"
        group_cols = ["station_id", "phase_type"]
        val_col_gt = "phase_time"
        val_col_pred = "phase_time"
        max_residual_val = max_residual_s
        residual_to_s = 1.0
        # Convert time strings to epoch seconds for numeric comparison
        gt_sub = gt_sub.copy()
        pred_sub = pred_sub.copy()
        gt_sub["_time_epoch"] = pd.to_datetime(
            gt_sub["phase_time"]).astype(np.int64) / 1e9
        pred_sub["_time_epoch"] = pd.to_datetime(
            pred_sub["phase_time"]).astype(np.int64) / 1e9
        val_col_gt = "_time_epoch"
        val_col_pred = "_time_epoch"
        residual_to_s = 1.0

    # Index both DataFrames for fast lookup
    gt_sub = gt_sub.reset_index(drop=True)
    pred_sub = pred_sub.reset_index(drop=True)

    # Build groups
    gt_groups = {}
    for idx, row in gt_sub.iterrows():
        key = tuple(row[c] for c in group_cols)
        gt_groups.setdefault(key, []).append(idx)

    pred_groups = {}
    for idx, row in pred_sub.iterrows():
        key = tuple(row[c] for c in group_cols)
        pred_groups.setdefault(key, []).append(idx)

    all_keys = set(gt_groups.keys()) | set(pred_groups.keys())
    result_rows = []

    for key in tqdm(all_keys, desc=f"Matching detections for {phase_types}"):
        gt_idxs = gt_groups.get(key, [])
        pred_idxs = pred_groups.get(key, [])

        if not gt_idxs and not pred_idxs:
            continue

        # Get sorted values
        if gt_idxs:
            gt_vals = gt_sub.loc[gt_idxs, val_col_gt].values
            gt_order = np.argsort(gt_vals)
            gt_vals_sorted = gt_vals[gt_order]
            gt_idxs_sorted = [gt_idxs[i] for i in gt_order]
        else:
            gt_vals_sorted = np.array([])
            gt_idxs_sorted = []

        if pred_idxs:
            pred_vals = pred_sub.loc[pred_idxs, val_col_pred].values
            pred_order = np.argsort(pred_vals)
            pred_vals_sorted = pred_vals[pred_order]
            pred_idxs_sorted = [pred_idxs[i] for i in pred_order]
        else:
            pred_vals_sorted = np.array([])
            pred_idxs_sorted = []

        if len(gt_vals_sorted) == 0:
            # All preds are unmatched
            for pi in pred_idxs_sorted:
                result_rows.append(_make_unmatched_pred_row(
                    pred_sub.iloc[pi], polarity_threshold))
            continue
        if len(pred_vals_sorted) == 0:
            # All GT are unmatched
            for gi in gt_idxs_sorted:
                result_rows.append(_make_unmatched_gt_row(gt_sub.iloc[gi]))
            continue

        # Run matching
        raw_matches = _match_sorted(
            gt_vals_sorted.tolist(),
            pred_vals_sorted.tolist(),
            max_residual_val
        )

        for g_local, p_local, residual in raw_matches:
            if g_local is not None and p_local is not None:
                gi = gt_idxs_sorted[g_local]
                pi = pred_idxs_sorted[p_local]
                result_rows.append(_make_matched_row(
                    gt_sub.iloc[gi], pred_sub.iloc[pi],
                    residual * residual_to_s, polarity_threshold))
            elif g_local is not None:
                gi = gt_idxs_sorted[g_local]
                result_rows.append(_make_unmatched_gt_row(gt_sub.iloc[gi]))
            else:
                pi = pred_idxs_sorted[p_local]
                result_rows.append(_make_unmatched_pred_row(
                    pred_sub.iloc[pi], polarity_threshold))

    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows)


def _make_matched_row(gt_row, pred_row, residual_s, polarity_threshold):
    """Build a tp_candidate row from matched GT and pred."""
    row = {
        "match_type": "tp_candidate",
        "phase_type": gt_row.get("phase_type", ""),
        "residual_s": residual_s,
        # GT side
        "gt_event_id": gt_row.get("event_id", ""),
        "gt_station_id": gt_row.get("station_id", ""),
        "gt_pick_event_id": gt_row.get("pick_event_id", ""),
        "gt_phase_index": gt_row.get("phase_index", np.nan),
        "gt_phase_time": gt_row.get("phase_time", ""),
        "gt_phase_polarity": gt_row.get("phase_polarity", "N"),
        # Pred side
        "pred_event_id": pred_row.get("event_id", ""),
        "pred_station_id": pred_row.get("station_id", ""),
        "pred_phase_index": pred_row.get("phase_index", np.nan),
        "pred_phase_time": pred_row.get("phase_time", ""),
        "pred_phase_score": pred_row.get("phase_score", np.nan),
        "pred_polarity_score": pred_row.get("phase_polarity", 0.0),
        # GT metadata
        "magnitude": gt_row.get("event_magnitude", np.nan),
        "snr": gt_row.get("snr_max", np.nan),
        "distance_km": gt_row.get("distance_km", np.nan),
        "azimuth_deg": gt_row.get("azimuth_deg", np.nan),
        "event_latitude": gt_row.get("event_latitude", np.nan),
        "event_longitude": gt_row.get("event_longitude", np.nan),
        "event_depth_km": gt_row.get("event_depth_km", np.nan),
    }
    # Polarity prediction label
    try:
        pol_score = float(pred_row.get("phase_polarity", 0.0))
        row["pred_polarity_label"] = polarity_label(pol_score, polarity_threshold)
    except (ValueError, TypeError):
        row["pred_polarity_label"] = "N"

    # Event-center-specific columns
    if gt_row.get("phase_type") == "event":
        row["travel_time_gt"] = gt_row.get("travel_time_s", np.nan)
        row["travel_time_pred"] = pred_row.get("travel_time", np.nan)
        tt_gt = row["travel_time_gt"]
        tt_pred = row["travel_time_pred"]
        if pd.notna(tt_gt) and pd.notna(tt_pred):
            row["travel_time_residual"] = float(tt_pred) - float(tt_gt)
        else:
            row["travel_time_residual"] = np.nan
        row["distance_pred"] = pred_row.get("epicentral_distance", np.nan)
        dist_gt = gt_row.get("distance_km", np.nan)
        dist_pred = row["distance_pred"]
        if pd.notna(dist_gt) and pd.notna(dist_pred):
            row["distance_residual"] = float(dist_pred) - float(dist_gt)
        else:
            row["distance_residual"] = np.nan
        row["azimuth_pred"] = pred_row.get("azimuth", np.nan)
        azi_gt = gt_row.get("azimuth_deg", np.nan)
        azi_pred = row["azimuth_pred"]
        if pd.notna(azi_gt) and pd.notna(azi_pred):
            diff = (float(azi_pred) - float(azi_gt) + 180) % 360 - 180
            row["azimuth_residual"] = diff
        else:
            row["azimuth_residual"] = np.nan
        row["depth_pred"] = pred_row.get("event_location_z", np.nan)
        row["magnitude_pred"] = pred_row.get("magnitude", np.nan)

    return row


def _make_unmatched_gt_row(gt_row):
    """Build an unmatched_gt (false negative) row."""
    return {
        "match_type": "unmatched_gt",
        "phase_type": gt_row.get("phase_type", ""),
        "residual_s": np.nan,
        "gt_event_id": gt_row.get("event_id", ""),
        "gt_station_id": gt_row.get("station_id", ""),
        "gt_pick_event_id": gt_row.get("pick_event_id", ""),
        "gt_phase_index": gt_row.get("phase_index", np.nan),
        "gt_phase_time": gt_row.get("phase_time", ""),
        "gt_phase_polarity": gt_row.get("phase_polarity", "N"),
        "pred_event_id": np.nan,
        "pred_station_id": np.nan,
        "pred_phase_index": np.nan,
        "pred_phase_time": np.nan,
        "pred_phase_score": np.nan,
        "pred_polarity_score": np.nan,
        "pred_polarity_label": np.nan,
        "magnitude": gt_row.get("event_magnitude", np.nan),
        "snr": gt_row.get("snr_max", np.nan),
        "distance_km": gt_row.get("distance_km", np.nan),
        "azimuth_deg": gt_row.get("azimuth_deg", np.nan),
        "event_latitude": gt_row.get("event_latitude", np.nan),
        "event_longitude": gt_row.get("event_longitude", np.nan),
        "event_depth_km": gt_row.get("event_depth_km", np.nan),
    }


def _make_unmatched_pred_row(pred_row, polarity_threshold):
    """Build an unmatched_pred (false positive) row."""
    try:
        pol_score = float(pred_row.get("phase_polarity", 0.0))
        pol_label = polarity_label(pol_score, polarity_threshold)
    except (ValueError, TypeError):
        pol_score = 0.0
        pol_label = "N"
    return {
        "match_type": "unmatched_pred",
        "phase_type": pred_row.get("phase_type", ""),
        "residual_s": np.nan,
        "gt_event_id": np.nan,
        "gt_station_id": np.nan,
        "gt_pick_event_id": np.nan,
        "gt_phase_index": np.nan,
        "gt_phase_time": np.nan,
        "gt_phase_polarity": np.nan,
        "pred_event_id": pred_row.get("event_id", ""),
        "pred_station_id": pred_row.get("station_id", ""),
        "pred_phase_index": pred_row.get("phase_index", np.nan),
        "pred_phase_time": pred_row.get("phase_time", ""),
        "pred_phase_score": pred_row.get("phase_score", np.nan),
        "pred_polarity_score": pol_score,
        "pred_polarity_label": pol_label,
        "magnitude": np.nan,
        "snr": np.nan,
        "distance_km": np.nan,
        "azimuth_deg": np.nan,
        "event_latitude": np.nan,
        "event_longitude": np.nan,
        "event_depth_km": np.nan,
    }


def normalize_event_preds_for_matching(event_preds):
    """Normalize event prediction columns to match pick prediction format.

    Converts event_center_index (feature-scale) to phase_index (sample-scale),
    maps event_score to phase_score, and sets phase_type to "event".
    """
    if event_preds.empty:
        return event_preds
    df = event_preds.copy()
    df["phase_index"] = df["event_center_index"] * FEATURE_SCALE
    df["phase_score"] = df["event_score"]
    df["phase_type"] = "event"
    if "event_center_time" in df.columns:
        df["phase_time"] = df["event_center_time"]
    elif "event_time" in df.columns:
        df["phase_time"] = df["event_time"]
    if "phase_polarity" not in df.columns:
        df["phase_polarity"] = 0.0
    return df


def filter_predictions_main_event(pred_df, gt_picks, is_event_pred=False,
                                  sample_rate=SAMPLE_RATE,
                                  feature_scale=FEATURE_SCALE,
                                  keep_unmatched=False):
    """Filter predictions to only keep those near main event GT picks.

    Main event = pick_event_id == event_id.

    Parameters
    ----------
    keep_unmatched : bool
        If True, predictions at (event_id, station_id) combos with no GT
        windows are kept (matches old event-level behavior). If False, they
        are discarded (matches old pick-level behavior).

    Performance: O(N_gt + N_pred) hash-join instead of the naive
    O(N_gt_groups × N_pred) per-group scan.
    """
    if pred_df.empty:
        return pred_df

    # Use P/S picks from gt_picks to define duration windows.
    # For event preds: only main event picks (pick_event_id == event_id),
    #   matching old event code which uses `for e_id in [event_id]`.
    # For pick preds: all events' P/S picks,
    #   matching old pick code which uses `for e_id in set(e_ids)`.
    gt_ps = gt_picks[gt_picks["phase_type"].isin(["P", "S"])]
    if is_event_pred:
        gt_ps = gt_ps[gt_ps["pick_event_id"] == gt_ps["event_id"]]

    # ── Step 1: Compute duration windows per (event_id, station_id) ──────────
    windows: list[tuple] = []
    for (eid, sid), grp in tqdm(gt_ps.groupby(["event_id", "station_id"],
                                          sort=False), desc="Computing GT windows for filtering"):
        p_arr = grp.loc[grp["phase_type"] == "P", "phase_index"].values
        s_arr = grp.loc[grp["phase_type"] == "S", "phase_index"].values
        if len(p_arr) == 0 and len(s_arr) == 0:
            continue
        dist = grp["distance_km"].iloc[0]
        if np.isnan(dist):
            dist = 0.0
        if len(s_arr) == 0:
            s_arr = p_arr + dist / VAPP * sample_rate
        if len(p_arr) == 0:
            p_arr = s_arr - dist / VAPP * sample_rate
        n = min(len(p_arr), len(s_arr))
        p_s = np.sort(p_arr)[:n]
        s_s = np.sort(s_arr)[:n]
        lo = np.clip(p_s - 5 * sample_rate, 0, MAX_TRACE_SAMPLES)
        hi = np.clip(p_s + 3 * (s_s - p_s), 0, MAX_TRACE_SAMPLES)
        for l, h in zip(lo, hi):
            windows.append((eid, sid, l, h))

    if not windows:
        if keep_unmatched:
            return pred_df.copy()
        return pd.DataFrame(columns=pred_df.columns)

    windows_df = pd.DataFrame(
        windows, columns=["event_id", "station_id", "_lo", "_hi"])

    # ── Step 2: Determine pred index column and scaling ──────────────────────
    # Raw event preds use event_center_index (feature-scale units);
    # normalized preds / pick preds use phase_index (sample units).
    
    if is_event_pred:
        windows_df["_lo"] = windows_df["_lo"] / feature_scale
        windows_df["_hi"] = windows_df["_hi"] / feature_scale
        if "event_center_index" in pred_df.columns and "phase_index" not in pred_df.columns:
            pred_index_col = "event_center_index"
        else:
            pred_index_col = "phase_index"
    else:
        pred_index_col = "phase_index"

    # ── Step 3: Vectorized merge-filter ──────────────────────────────────────
    pred_tagged = pred_df.reset_index(drop=True)
    pred_tagged = pred_tagged.assign(_row=np.arange(len(pred_tagged)))

    merged = pred_tagged.merge(
        windows_df, on=["event_id", "station_id"], how="inner")

    if merged.empty:
        matched_rows = np.array([], dtype=int)
    else:
        idx_vals = merged[pred_index_col].values
        mask = ((idx_vals >= merged["_lo"].values)
                & (idx_vals <= merged["_hi"].values))
        matched_rows = np.sort(merged.loc[mask, "_row"].unique())

    # ── Step 4: Optionally keep pred rows with no GT windows ─────────────────
    if keep_unmatched:
        pred_keys = set(zip(pred_tagged["event_id"],
                            pred_tagged["station_id"]))
        win_keys = set(zip(windows_df["event_id"],
                           windows_df["station_id"]))
        unmatched_keys = pred_keys - win_keys
        if unmatched_keys:
            um_mask = pred_tagged.apply(
                lambda r: (r["event_id"], r["station_id"]) in unmatched_keys,
                axis=1)
            extra = pred_tagged.loc[um_mask, "_row"].values
            matched_rows = np.sort(np.union1d(matched_rows, extra))

    if len(matched_rows) == 0:
        return pd.DataFrame(columns=pred_df.columns)
    return pred_df.iloc[matched_rows].reset_index(drop=True)


def cluster_and_match_events(event_preds, gt_events,
                             association_method="dbscan",
                             dbscan_epsilon=DEFAULT_DBSCAN_EPSILON,
                             min_stations=DEFAULT_MIN_STATIONS,
                             time_space_weight=None,
                             event_dt_bar=DEFAULT_EVENT_DT_BAR,
                             sample_rate=SAMPLE_RATE,
                             feature_scale=FEATURE_SCALE,
                             gt_picks=None):
    """Cluster event predictions per event_id and match to GT events.

    This reproduces the original metrics.py behavior: DBSCAN clustering
    within each event_id, then matching the best cluster to the GT event.

    Parameters
    ----------
    gt_picks : DataFrame or None
        If provided, used to look up per-station GT distance/azimuth for
        computing per-station residuals (matching old EventMetric behavior).

    Returns a DataFrame with match results for clustered events.
    """
    if time_space_weight is None:
        time_space_weight = DEFAULT_TIME_SPACE_WEIGHT
    weights = np.array(time_space_weight)

    if event_preds.empty or gt_events.empty:
        return pd.DataFrame()

    dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=min_stations)

    # Build per-station GT distance/azimuth lookup from gt_picks
    # Key: (event_id, station_id) → (distance_km, azimuth_deg)
    gt_sta_lookup = {}
    if gt_picks is not None and not gt_picks.empty:
        # Use P picks to get station-level metadata (one per station per event)
        gt_ps = gt_picks[gt_picks["phase_type"] == "P"]
        for _, row in gt_ps.iterrows():
            key = (row["event_id"], row["station_id"])
            if key not in gt_sta_lookup:
                sta_lat = row.get("station_latitude", np.nan)
                sta_lon = row.get("station_longitude", np.nan)
                evt_lat = row.get("event_latitude", np.nan)
                evt_lon = row.get("event_longitude", np.nan)
                dist = row.get("distance_km", np.nan)
                # Compute azimuth from station to event (matching old code)
                if not (np.isnan(sta_lat) or np.isnan(sta_lon)
                        or np.isnan(evt_lat) or np.isnan(evt_lon)):
                    azi = haversine_azimuth_deg(
                        sta_lat, sta_lon, evt_lat, evt_lon)
                else:
                    azi = row.get("azimuth_deg", np.nan)
                gt_sta_lookup[key] = (dist, azi)

    result_rows = []
    gt_events_dict = gt_events.set_index("pick_event_id").to_dict("index")
    processed_gt_ids = set()  # track GT event_ids that were matched against predictions

    for event_id, group in tqdm(event_preds.groupby("event_id"), desc="Matching events"):
        if event_id not in gt_events_dict:
            # Predicted event_id with no GT — counted as FP via tp_candidate tracking
            # (no gt row to emit; these predictions produce no tp_candidate row,
            # so they are simply absent which is not perfect but reflects that
            # the event_id doesn't exist in GT at all)
            continue

        processed_gt_ids.add(event_id)
        gt_evt = gt_events_dict[event_id]
        gt_lat = gt_evt["latitude"]
        gt_lon = gt_evt["longitude"]
        gt_depth = gt_evt["depth_km"]
        gt_mag = gt_evt["magnitude"]
        gt_time_index = gt_evt.get("event_time_index", np.nan)
        num_stations = gt_evt.get("num_stations", 0)

        label_x = gt_lon * DEGREE2KM * np.cos(np.radians(gt_lat))
        label_y = gt_lat * DEGREE2KM
        label_z = gt_depth

        n_total = len(group)

        if group.empty:
            result_rows.append({
                "match_type": "unmatched_gt",
                "gt_pick_event_id": event_id,
                "gt_lon": gt_lon, "gt_lat": gt_lat,
                "gt_depth": gt_depth, "gt_magnitude": gt_mag,
                "gt_time_index": gt_time_index,
                "num_gt_stations": num_stations,
                "time_residual_s": np.nan,
                "x_residual_km": np.nan, "y_residual_km": np.nan,
                "z_residual_km": np.nan, "magnitude_residual": np.nan,
                "n_total_centers": 0,
                "n_associated_centers": 0,
                "n_clusters": 0,
                "association_ratio": np.nan,
            })
            continue

        # Build feature matrix for clustering
        hypo = np.column_stack([
            group["event_index"].values,
            group["event_location_x"].values,
            group["event_location_y"].values,
            group["event_location_z"].values,
        ])
        meta = np.column_stack([
            group["event_longitude"].values,
            group["event_latitude"].values,
            group["epicentral_distance"].values,
            group["azimuth"].values,
            group["magnitude"].values,
        ])

        if association_method == "dbscan":
            labels = dbscan.fit_predict(hypo * weights[None, :])
            classes, counts = np.unique(labels, return_counts=True)
            if len(classes) > 1 and classes[0] == -1:
                classes = classes[1:]
                counts = counts[1:]
            if len(hypo) > 5:
                classes = classes[counts >= min_stations]
        elif association_method == "associator":
            if "event_id_pred" in group.columns:
                labels = group["event_id_pred"].values
            else:
                labels = np.zeros(len(group), dtype=int)
            classes, counts = np.unique(labels, return_counts=True)
            if len(hypo) > 5:
                classes = classes[counts >= min_stations]

        n_clusters = len(classes)
        n_associated = int(np.isin(labels, classes).sum())

        if n_clusters == 0:
            result_rows.append({
                "match_type": "unmatched_gt",
                "gt_pick_event_id": event_id,
                "gt_lon": gt_lon, "gt_lat": gt_lat,
                "gt_depth": gt_depth, "gt_magnitude": gt_mag,
                "gt_time_index": gt_time_index,
                "num_gt_stations": num_stations,
                "time_residual_s": np.nan,
                "x_residual_km": np.nan, "y_residual_km": np.nan,
                "z_residual_km": np.nan, "magnitude_residual": np.nan,
                "n_total_centers": n_total,
                "n_associated_centers": n_associated,
                "n_clusters": 0,
                "association_ratio": 0.0,
            })
            continue

        # Find best matching cluster by time
        best_time_diff = np.inf
        best_row = None

        for c in classes:
            cluster_hypo = hypo[labels == c]
            cluster_meta = meta[labels == c]

            idx_sort = np.argsort(cluster_meta[:, 2])  # sort by distance
            cluster_hypo = cluster_hypo[idx_sort]
            cluster_meta = cluster_meta[idx_sort]

            time_diff = (np.median(cluster_hypo[:5, 0]) * feature_scale
                         - gt_time_index) / sample_rate

            if abs(time_diff) < abs(best_time_diff):
                best_time_diff = time_diff
                avg_center = np.median(cluster_meta[:, :2], axis=0)
                pred_x = (avg_center[0]
                          * np.cos(np.radians(avg_center[1])) * DEGREE2KM)
                pred_y = avg_center[1] * DEGREE2KM
                pred_z = np.median(cluster_hypo[:, 3])
                pred_mag = np.median(cluster_meta[:, 4])

                # Per-station distance and azimuth residuals
                stations_arr = group["station_id"].values[labels == c][idx_sort]
                pred_dist = cluster_meta[:, 2]
                pred_azi = cluster_meta[:, 3]

                # Compute per-station residuals against GT
                dist_residuals = []
                azi_residuals = []
                if gt_sta_lookup:
                    for i, sta_id in enumerate(stations_arr):
                        gt_da = gt_sta_lookup.get(
                            (event_id, sta_id), (np.nan, np.nan))
                        gt_dist, gt_azi = gt_da
                        if pd.notna(gt_dist):
                            dist_residuals.append(
                                float(pred_dist[i]) - float(gt_dist))
                        if pd.notna(gt_azi):
                            diff = ((float(pred_azi[i]) - float(gt_azi)
                                     + 180) % 360) - 180
                            azi_residuals.append(diff)

                if best_row is not None:
                    # if there was a previous cluster, emit it as unmatched_pred (FP)
                    best_row["match_type"] = "unmatched_pred"
                    result_rows.append(best_row)
                best_row = {
                    "match_type": "tp_candidate",
                    "gt_pick_event_id": event_id,
                    "gt_lon": gt_lon, "gt_lat": gt_lat,
                    "gt_depth": gt_depth, "gt_magnitude": gt_mag,
                    "gt_time_index": gt_time_index,
                    "num_gt_stations": num_stations,
                    "pred_lon": avg_center[0],
                    "pred_lat": avg_center[1],
                    "pred_depth": pred_z,
                    "pred_magnitude": pred_mag,
                    "time_residual_s": best_time_diff,
                    "x_residual_km": pred_x - label_x,
                    "y_residual_km": pred_y - label_y,
                    "z_residual_km": pred_z - label_z,
                    "magnitude_residual": pred_mag - gt_mag,
                    "num_pred_stations": len(cluster_hypo),
                    "num_pred_clusters": n_clusters,
                    "distance_residuals": json.dumps(dist_residuals),
                    "azimuth_residuals": json.dumps(azi_residuals),
                    "n_total_centers": n_total,
                    "n_associated_centers": n_associated,
                    "n_clusters": n_clusters,
                    "association_ratio": (n_associated / n_total
                                          if n_total > 0 else np.nan),
                }
            else:
                # This cluster is not the best match, emit as unmatched_pred (FP)
                row = {
                    "match_type": "unmatched_pred",
                    "gt_pick_event_id": event_id,
                    "gt_lon": gt_lon, "gt_lat": gt_lat,
                    "gt_depth": gt_depth, "gt_magnitude": gt_mag,
                    "gt_time_index": gt_time_index,
                    "num_gt_stations": num_stations,
                    "pred_lon": np.median(cluster_meta[:, 0]),
                    "pred_lat": np.median(cluster_meta[:, 1]),
                    "pred_depth": np.median(cluster_hypo[:, 3]),
                    "pred_magnitude": np.median(cluster_meta[:, 4]),
                    "time_residual_s": time_diff,
                    "x_residual_km": (np.median(cluster_meta[:, 0]) * np.cos(
                        np.radians(np.median(cluster_meta[:, 1]))) * DEGREE2KM
                                     - label_x),
                    "y_residual_km": (np.median(cluster_meta[:, 1]) * DEGREE2KM
                                     - label_y),
                    "z_residual_km": (np.median(cluster_hypo[:, 3]) - label_z),
                    "magnitude_residual": (np.median(cluster_meta[:, 4]) - gt_mag),
                    "num_pred_stations": len(cluster_hypo),
                    "num_pred_clusters": n_clusters,
                    "distance_residuals": json.dumps(dist_residuals),
                    "azimuth_residuals": json.dumps(azi_residuals),
                    "n_total_centers": n_total,
                    "n_associated_centers": n_associated,
                    "n_clusters": n_clusters,
                    "association_ratio": (n_associated / n_total
                                          if n_total > 0 else np.nan),
                }
                result_rows.append(row)
            

        if best_row is not None:
            result_rows.append(best_row)

    # Add unmatched_gt rows for GT events that had no predictions at all.
    # These are pure FN and are essential for correct recall computation.
    for eid, gt_evt in gt_events_dict.items():
        if eid in processed_gt_ids:
            continue
        result_rows.append({
            "match_type": "unmatched_gt",
            "gt_pick_event_id": eid,
            "gt_lon": gt_evt["longitude"],
            "gt_lat": gt_evt["latitude"],
            "gt_depth": gt_evt["depth_km"],
            "gt_magnitude": gt_evt["magnitude"],
            "gt_time_index": gt_evt.get("event_time_index", np.nan),
            "num_gt_stations": gt_evt.get("num_stations", 0),
            "time_residual_s": np.nan,
            "x_residual_km": np.nan, "y_residual_km": np.nan,
            "z_residual_km": np.nan, "magnitude_residual": np.nan,
            "n_total_centers": 0,
            "n_associated_centers": 0,
            "n_clusters": 0,
            "association_ratio": np.nan,
        })

    if not result_rows:
        return pd.DataFrame()
    return pd.DataFrame(result_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def apply_score_threshold(matched_df, score_threshold):
    """Apply score threshold to matched_df, reclassifying low-score matches.

    - tp_candidate with pred_phase_score < threshold → reclassified to unmatched_gt
    - unmatched_pred with pred_phase_score < threshold → removed (not counted as FP)

    Returns a new DataFrame (copy).
    """
    active = matched_df.copy()
    below = (
        (active["match_type"] == "tp_candidate")
        & (active["pred_phase_score"] < score_threshold)
    )
    active.loc[below, "match_type"] = "unmatched_gt"
    active = active[
        ~(
            (active["match_type"] == "unmatched_pred")
            & (active["pred_phase_score"] < score_threshold)
        )
    ].copy()
    return active


def compute_metrics(matched_df, dt_bar, score_threshold=None):
    """Compute detection metrics from a matched DataFrame.

    Parameters
    ----------
    matched_df : DataFrame
        Output of match_detections with columns: match_type, phase_type,
        residual_s, pred_phase_score, etc.
    dt_bar : float
        Maximum |residual_s| (seconds) for a true positive.
    score_threshold : float or None
        If set, predictions below this score are discarded / reclassified.

    Returns
    -------
    dict : Metrics keyed by ``"<metric>_<phase>"`` (e.g. ``"precision_p"``).
    """
    if matched_df.empty:
        return {}

    if score_threshold is not None:
        active = apply_score_threshold(matched_df, score_threshold)
    else:
        active = matched_df

    results = {}
    phase_types = sorted(active["phase_type"].dropna().unique())

    for phase in phase_types:
        pd_phase = active[active["phase_type"] == phase]
        if pd_phase.empty:
            continue

        tp_mask = (pd_phase["match_type"] == "tp_candidate") & (
            pd_phase["residual_s"].abs() < dt_bar
        )
        tp = int(tp_mask.sum())

        outside_mask = (pd_phase["match_type"] == "tp_candidate") & (
            pd_phase["residual_s"].abs() >= dt_bar
        )
        fp = int((pd_phase["match_type"] == "unmatched_pred").sum()
                 + outside_mask.sum())
        fn = int((pd_phase["match_type"] == "unmatched_gt").sum()
                 + outside_mask.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan
            
        TPR = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        FPR = fp / (fp + tp) if (fp + tp) > 0 else np.nan

        tp_rows = pd_phase[tp_mask]
        res = tp_rows["residual_s"].values

        key = phase.lower()
        results.update({
            f"tp_{key}": tp,
            f"fp_{key}": fp,
            f"fn_{key}": fn,
            f"precision_{key}": precision,
            f"recall_{key}": recall,
            f"f1_{key}": f1,
            f"tpr_{key}": TPR,
            f"fpr_{key}": FPR,
            f"residual_mean_{key}": float(res.mean()) if len(res) > 0 else np.nan,
            f"residual_std_{key}": float(res.std()) if len(res) > 0 else np.nan,
        })

        # Event-center-specific residual stats
        if phase == "event" and len(tp_rows) > 0:
            for col, label in [
                ("travel_time_residual", "travel_time"),
                ("distance_residual", "distance"),
                ("azimuth_residual", "azimuth"),
            ]:
                if col in tp_rows.columns:
                    vals = tp_rows[col].dropna().values
                    results[f"{label}_residual_mean"] = (
                        float(vals.mean()) if len(vals) > 0 else np.nan
                    )
                    results[f"{label}_residual_std"] = (
                        float(vals.std()) if len(vals) > 0 else np.nan
                    )

    return results


def compute_event_cluster_metrics(event_matched_df,
                                   event_dt_bar=DEFAULT_EVENT_DT_BAR):
    """Compute metrics for DBSCAN-clustered event matching.

    Parameters
    ----------
    event_matched_df : DataFrame
        Output of cluster_and_match_events.
    event_dt_bar : float
        Max |time_residual_s| for a true positive.

    Returns
    -------
    dict : Event-level metrics (precision, recall, F1, residual mean/std).
    """
    if event_matched_df.empty:
        return {}

    tp_mask = (
        (event_matched_df["match_type"] == "tp_candidate")
        & (event_matched_df["time_residual_s"].abs() < event_dt_bar)
    )
    tp = int(tp_mask.sum())

    n_gt = int(event_matched_df["match_type"].isin(
        ["tp_candidate", "unmatched_gt"]).sum())
    n_pred = int((event_matched_df["match_type"].isin(["tp_candidate", "unmatched_pred"])).sum())

    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
        else np.nan
    )

    tp_rows = event_matched_df[tp_mask]

    results = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    for col, label in [
        ("time_residual_s", "time"),
        ("x_residual_km", "x"),
        ("y_residual_km", "y"),
        ("z_residual_km", "z"),
        ("magnitude_residual", "magnitude"),
    ]:
        if col in tp_rows.columns:
            vals = tp_rows[col].dropna().values
            results[f"miu_{label}"] = (
                float(vals.mean()) if len(vals) > 0 else np.nan
            )
            results[f"sigma_{label}"] = (
                float(vals.std()) if len(vals) > 0 else np.nan
            )

    # Aggregate per-station distance and azimuth residuals from JSON columns
    for col, label in [
        ("distance_residuals", "distance"),
        ("azimuth_residuals", "azimuth"),
    ]:
        if col in tp_rows.columns:
            all_vals = []
            for s in tp_rows[col].dropna():
                try:
                    arr = json.loads(s) if isinstance(s, str) else s
                    if isinstance(arr, list):
                        all_vals.extend(arr)
                except (json.JSONDecodeError, TypeError):
                    pass
            all_vals = np.array(all_vals, dtype=float)
            results[f"miu_{label}"] = (
                float(all_vals.mean()) if len(all_vals) > 0 else np.nan
            )
            results[f"sigma_{label}"] = (
                float(all_vals.std()) if len(all_vals) > 0 else np.nan
            )

    return results


def compute_metrics_vs_param(matched_df, param, bins, dt_bar,
                              num_pred_per_bin=None, num_gt_per_bin=None):
    """Compute per-bin metrics broken down by a parameter.

    Parameters
    ----------
    matched_df : DataFrame
        Output of match_detections.
    param : str
        Column name to bin by (e.g. ``"snr"``, ``"magnitude"``,
        ``"distance_km"``, ``"azimuth_deg"``).
    bins : list
        Bin edges (may include ``-np.inf`` / ``np.inf``).
    dt_bar : float
        Max |residual_s| for a true positive.
    num_pred_per_bin : dict of {phase_key: array}, optional
        Pre-computed total prediction counts per bin, keyed by lowercase
        phase (``"p"``, ``"s"``, ``"event"``).  If *None*, computed from
        matched_df (may under-count FP for unmatched preds lacking GT
        metadata).
    num_gt_per_bin : dict of {phase_key: array}, optional
        Pre-computed total GT counts per bin.

    Returns
    -------
    dict : ``{phase_key: {"precision": list, "recall": list, "f1": list,
    "tp": list, "fp": list, "fn": list}}``.
    """
    if matched_df.empty:
        return {}

    results = {}
    phase_types = sorted(matched_df["phase_type"].dropna().unique())

    for phase in phase_types:
        pd_phase = matched_df[matched_df["phase_type"] == phase]
        key = phase.lower()

        # GT per bin
        gt_rows = pd_phase[
            pd_phase["match_type"].isin(["tp_candidate", "unmatched_gt"])
        ]
        if num_gt_per_bin is not None and key in num_gt_per_bin:
            gt_bin = np.array(num_gt_per_bin[key], dtype=float)
        else:
            gt_bin = np.histogram(gt_rows[param].dropna().values, bins=bins)[0].astype(float)

        # Pred per bin
        if num_pred_per_bin is not None and key in num_pred_per_bin:
            pred_bin = np.array(num_pred_per_bin[key], dtype=float)
        else:
            pred_rows = pd_phase[
                pd_phase["match_type"].isin(["tp_candidate", "unmatched_pred"])
            ]
            pred_bin = np.histogram(
                pred_rows[param].dropna().values, bins=bins
            )[0].astype(float)

        # TP per bin
        tp_rows = pd_phase[
            (pd_phase["match_type"] == "tp_candidate")
            & (pd_phase["residual_s"].abs() < dt_bar)
        ]
        tp_bin = np.histogram(
            tp_rows[param].dropna().values, bins=bins
        )[0].astype(float)

        fp_bin = pred_bin - tp_bin
        fn_bin = gt_bin - tp_bin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            precision_bin = np.where(
                (tp_bin + fp_bin) > 0,
                tp_bin / (tp_bin + fp_bin), 0.0,
            )
            recall_bin = np.where(
                (tp_bin + fn_bin) > 0,
                tp_bin / (tp_bin + fn_bin), 0.0,
            )
            f1_bin = np.where(
                (precision_bin + recall_bin) > 0,
                2 * precision_bin * recall_bin / (precision_bin + recall_bin),
                0.0,
            )

        results[key] = {
            "precision": precision_bin.tolist(),
            "recall": recall_bin.tolist(),
            "f1": f1_bin.tolist(),
            "tp": tp_bin.tolist(),
            "fp": fp_bin.tolist(),
            "fn": fn_bin.tolist(),
        }

    return results


def sweep_thresholds(matched_df, thresholds, dt_bar):
    """Run a threshold sweep, returning per-threshold metrics.

    Because *matched_df* contains **all** predictions (matched, unmatched_gt,
    and unmatched_pred with their scores), sweeping is just repeated calls to
    :func:`compute_metrics` — no re-matching needed.

    Parameters
    ----------
    matched_df : DataFrame
        Output of match_detections (**not** pre-filtered by score).
    thresholds : list of float
        Score thresholds to sweep (will be sorted).
    dt_bar : float
        Max |residual_s| for TP.

    Returns
    -------
    dict : ``{phase_key: {"precision": ndarray, "recall": ndarray,
    "f1": ndarray}}``.
    """
    if matched_df.empty:
        return {}

    all_stats = []
    for t in sorted(thresholds):
        all_stats.append(compute_metrics(matched_df, dt_bar=dt_bar,
                                         score_threshold=t))

    # Discover phase keys
    phase_keys = set()
    for s in all_stats:
        for k in s:
            if k.startswith("precision_"):
                phase_keys.add(k.split("_", 1)[1])

    results = {}
    for pk in sorted(phase_keys):
        results[pk] = {
            "precision": np.array(
                [s.get(f"precision_{pk}", np.nan) for s in all_stats]),
            "recall": np.array(
                [s.get(f"recall_{pk}", np.nan) for s in all_stats]),
            "f1": np.array(
                [s.get(f"f1_{pk}", np.nan) for s in all_stats]),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section F: Plotting functions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_residual_histogram(matched_df, dt_bar, distribution_bar=None,
                            save_path=None, info_only=False):
    """Plot pick residual histograms and print summary statistics.

    Produces three figure files (when *save_path* is given):
      1. ``phase_picking*`` — residual histograms for P and S
      2. ``phase_score*``   — residual vs score scatter
      3. ``phase_score_hist*`` — score histograms
    """
    if distribution_bar is None:
        distribution_bar = dt_bar

    stats = compute_metrics(matched_df, dt_bar=dt_bar)

    for phase in ["p", "s"]:
        if f"tp_{phase}" not in stats:
            continue
        print(f"  {phase.upper()}: TP={stats[f'tp_{phase}']}, "
              f"FP={stats[f'fp_{phase}']}, FN={stats[f'fn_{phase}']}")
        print(f"  precision={stats[f'precision_{phase}']:.4f}, "
              f"recall={stats[f'recall_{phase}']:.4f}, "
              f"F1={stats[f'f1_{phase}']:.4f}")
        print(f"  residual mean={stats[f'residual_mean_{phase}']:.4f}, "
              f"std={stats[f'residual_std_{phase}']:.4f}")

    if info_only:
        return stats

    tp_mask = (
        (matched_df["match_type"] == "tp_candidate")
        & (matched_df["residual_s"].abs() < dt_bar)
    )
    tp_data = matched_df[tp_mask]

    # --- Residual histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, phase, title in zip(axes, ["P", "S"], ["P phase", "S phase"]):
        res = tp_data[tp_data["phase_type"] == phase]["residual_s"].values
        res_filt = res[np.abs(res) < distribution_bar]
        ax.hist(res_filt, bins=11,
                range=(-distribution_bar, distribution_bar))
        ax.set_title(title)
        ax.set_xlabel("Residual (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Score scatter ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax, phase, color, label in zip(
        axes, ["P", "S"], ["orange", "#43BEFF"], ["TP_P", "TP_S"]
    ):
        sub = tp_data[tp_data["phase_type"] == phase]
        if not sub.empty:
            ax.scatter(sub["residual_s"].values,
                       sub["pred_phase_score"].values,
                       s=10, c=color, label=label)
        ax.set_xlabel("pred-label time residual [s]")
        ax.set_ylabel("phase score")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace("phase_picking", "phase_score"),
                     dpi=300, bbox_inches="tight")
    plt.close()

    # --- Score histogram ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax, phase, color, label in zip(
        axes, ["P", "S"], ["orange", "#43BEFF"], ["TP_P", "TP_S"]
    ):
        sub = tp_data[tp_data["phase_type"] == phase]
        if not sub.empty:
            ax.hist(sub["pred_phase_score"].values, bins=41, range=(0, 1),
                    color=color, alpha=0.7, label=label)
        ax.set_xlabel("phase score")
        ax.set_ylabel("count")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace("phase_picking", "phase_score_hist"),
                     dpi=300, bbox_inches="tight")
    plt.close()

    return stats


def plot_polarity_confusion(matched_df, dt_bar, save_path=None):
    """Plot polarity confusion matrices (P and S).

    Includes ALL tp_candidate rows (regardless of residual) and unmatched_gt
    rows (with pred_polarity='N'), matching old metrics_old.py behavior where
    every matched GT pick counts, including false negatives.
    """
    # Collect tp_candidate rows (all matched, regardless of residual)
    # tp_rows = matched_df[(matched_df["match_type"] == "tp_candidate") & (matched_df["residual_s"].abs() < dt_bar)].copy()
    tp_rows = matched_df[matched_df["match_type"] == "tp_candidate"].copy()
    # Collect unmatched_gt rows — these are FN, assign pred_polarity_label='N'
    fn_rows = matched_df[matched_df["match_type"] == "unmatched_gt"].copy()
    if not fn_rows.empty:
        fn_rows["pred_polarity_label"] = "N"
    data = pd.concat([tp_rows, fn_rows], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    for ax, phase, title in zip(axes, ["P", "S"], ["P phase", "S phase"]):
        sub = data[data["phase_type"] == phase]
        if sub.empty or "gt_phase_polarity" not in sub.columns:
            ax.set_title(f"{title} (no data)")
            continue

        gt_pol = sub["gt_phase_polarity"].astype(str).values
        pred_pol = sub["pred_polarity_label"].astype(str).values

        conf = confusion_matrix(
            gt_pol, pred_pol, labels=["D", "N", "U"]
        )[::-1]

        ax.set_title(title)
        sns.heatmap(
            conf, annot=True, fmt="d", cmap="Blues", linewidths=1.0,
            xticklabels=["D", "N", "U"], yticklabels=["U", "N", "D"],
            cbar=False, ax=ax,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = conf / conf.sum(axis=1, keepdims=True) * 100
            pct = np.nan_to_num(pct)
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                ax.text(
                    j + 0.5, i + 0.6, f"{pct[i, j]:.2f}%",
                    ha="center", va="center", fontsize=8,
                    color="black" if pct[i, j] < 50 else "white",
                )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Manual")
        ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_event_residuals(event_matched_df, event_dt_bar=DEFAULT_EVENT_DT_BAR,
                          save_path=None, info_only=False):
    """Plot clustered-event residual histograms and print summary."""
    stats = compute_event_cluster_metrics(event_matched_df, event_dt_bar)

    if stats:
        print(f"  Events: TP={stats.get('true_positives', 0)}, "
              f"FP={stats.get('false_positives', 0)}, "
              f"FN={stats.get('false_negatives', 0)}")
        print(f"  precision={stats.get('precision', np.nan):.4f}, "
              f"recall={stats.get('recall', np.nan):.4f}, "
              f"F1={stats.get('f1', np.nan):.4f}")
        for label in ["time", "x", "y", "z", "magnitude",
                       "distance", "azimuth"]:
            m = stats.get(f"miu_{label}", np.nan)
            s = stats.get(f"sigma_{label}", np.nan)
            if pd.notna(m):
                print(f"  {label}: mean={m:.4f}, std={s:.4f}")

    if info_only or event_matched_df.empty:
        return stats

    tp_mask = (
        (event_matched_df["match_type"] == "tp_candidate")
        & (event_matched_df["time_residual_s"].abs() < event_dt_bar)
    )
    tp = event_matched_df[tp_mask]

    if tp.empty:
        return stats

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    configs = [
        (ax[0, 0], "time_residual_s", "event time residual", 31, (-3, 3)),
        (ax[0, 1], "x_residual_km", "event x residual", 31, (-30, 30)),
        (ax[0, 2], "y_residual_km", "event y residual", 31, (-30, 30)),
        (ax[0, 3], "z_residual_km", "event z residual", 21, (-10, 10)),
    ]
    if "magnitude_residual" in tp.columns:
        configs.append(
            (ax[1, 0], "magnitude_residual", "magnitude residual",
             21, (-2, 2)))

    for a, col, title, nbins, rng in configs:
        if col in tp.columns:
            vals = tp[col].dropna().values
            if len(vals) > 0:
                a.hist(vals, bins=nbins, range=rng)
        a.set_title(title)

    # Per-station distance and azimuth residual histograms from JSON columns
    for a, col, title, nbins, rng in [
        (ax[1, 1], "distance_residuals", "epicenter distance residual",
         41, (-20, 20)),
        (ax[1, 2], "azimuth_residuals", "azimuth residual",
         71, (-180, 180)),
    ]:
        if col in tp.columns:
            all_vals = []
            for s in tp[col].dropna():
                try:
                    arr = json.loads(s) if isinstance(s, str) else s
                    if isinstance(arr, list):
                        all_vals.extend(arr)
                except (json.JSONDecodeError, TypeError):
                    pass
            if all_vals:
                a.hist(all_vals, bins=nbins, range=rng)
        a.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return stats


def plot_association_stats(event_matched_df, save_path=None, show=True):
    """Plot association ratio and cluster count distributions.

    Parameters
    ----------
    event_matched_df : DataFrame
        Output of cluster_and_match_events (must include columns:
        association_ratio, n_clusters, n_total_centers, n_associated_centers).
    """
    if event_matched_df.empty:
        return

    has_cols = all(c in event_matched_df.columns
                   for c in ["association_ratio", "n_clusters",
                              "n_total_centers"])
    if not has_cols:
        print("  Association stats columns not found, skipping plot.")
        return

    # Only consider rows that have actual predictions (tp_candidate or those
    # with n_total_centers > 0)
    df = event_matched_df[event_matched_df["n_total_centers"] > 0].copy()
    if df.empty:
        print("  No events with predictions for association stats.")
        return

    ratio = df["association_ratio"].dropna().values
    n_clust = df["n_clusters"].dropna().values.astype(int)
    n_total = df["n_total_centers"].dropna().values

    print(f"  Association stats ({len(df)} events with predictions):")
    print(f"    ratio: mean={np.mean(ratio):.4f}, "
          f"median={np.median(ratio):.4f}")
    print(f"    n_clusters: mean={np.mean(n_clust):.2f}, "
          f"median={np.median(n_clust):.1f}")
    print(f"    n_total_centers: mean={np.mean(n_total):.1f}, "
          f"median={np.median(n_total):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Histogram of association_ratio
    axes[0].hist(ratio, bins=20, range=(0, 1), edgecolor="black")
    axes[0].set_xlabel("Association ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Association ratio distribution")

    # 2. Histogram of n_clusters
    max_clust = int(n_clust.max()) if len(n_clust) > 0 else 1
    axes[1].hist(n_clust, bins=range(0, max_clust + 2),
                 edgecolor="black", align="left")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Cluster count distribution")

    # 3. Scatter: n_total_centers vs association_ratio
    axes[2].scatter(n_total, ratio, s=15, alpha=0.5)
    axes[2].set_xlabel("Total event centers")
    axes[2].set_ylabel("Association ratio")
    axes[2].set_title("Centers vs association ratio")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_event_center_residuals(center_matched_df,
                                 dt_bar=DEFAULT_EVENT_CENTER_DT_BAR,
                                 save_path=None, info_only=False):
    """Plot event-center detection residual histograms."""
    stats = compute_metrics(center_matched_df, dt_bar=dt_bar)

    if "tp_event" in stats:
        print(f"  Event centers: TP={stats['tp_event']}, "
              f"FP={stats['fp_event']}, FN={stats['fn_event']}")
        print(f"  precision={stats['precision_event']:.4f}, "
              f"recall={stats['recall_event']:.4f}, "
              f"F1={stats['f1_event']:.4f}")
        print(f"  center residual: "
              f"mean={stats['residual_mean_event']:.4f}, "
              f"std={stats['residual_std_event']:.4f}")
        for label in ["travel_time", "distance", "azimuth"]:
            m = stats.get(f"{label}_residual_mean", np.nan)
            s = stats.get(f"{label}_residual_std", np.nan)
            if pd.notna(m):
                print(f"  {label}: mean={m:.4f}, std={s:.4f}")

    if info_only or center_matched_df.empty:
        return stats

    tp_mask = (
        (center_matched_df["match_type"] == "tp_candidate")
        & (center_matched_df["residual_s"].abs() < dt_bar)
    )
    tp = center_matched_df[tp_mask]
    if tp.empty:
        return stats

    cols = [
        ("residual_s", "Center time residual (s)", 31, (-dt_bar, dt_bar)),
        ("travel_time_residual", "Travel time residual (s)", 31, (-5, 5)),
        ("distance_residual", "Distance residual (km)", 31, (-20, 20)),
        ("azimuth_residual", "Azimuth residual (deg)", 71, (-180, 180)),
    ]
    existing = [(c, t, b, r) for c, t, b, r in cols if c in tp.columns]
    ncols = max(len(existing), 1)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    for ax, (col, title, nbins, rng) in zip(axes, existing):
        vals = tp[col].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=nbins, range=rng)
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return stats


def plot_pr_curve(sweep_results, thresholds=None, save_path=None,
                  fig=None, ax=None, ax_event=None,
                  model="", linestyle="-", alpha=1.0,
                  x_lim=None, y_lim=None, y_lim_event=None,
                  show=True):
    """Plot precision-recall curve from :func:`sweep_thresholds` output."""
    own_fig = fig is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=300)
        if "event" in sweep_results:
            ax_event = ax.twinx()

    x_lim = x_lim or (0.85, 1.0)
    y_lim = y_lim or (0.85, 1.0)
    y_lim_event = y_lim_event or (0.5, 1.0)
    model_label = f" ({model})" if model else ""

    colors = {"p": "C3", "s": "C0", "event": "C2"}
    labels = {"p": "P phase", "s": "S phase", "event": "Event"}

    for phase in ["p", "s"]:
        if phase not in sweep_results:
            continue
        s = sweep_results[phase]
        ax.plot(
            s["recall"], s["precision"],
            label=f"{labels[phase]}{model_label}",
            linewidth=3, c=colors[phase], linestyle=linestyle, alpha=alpha,
        )
        best = np.nanargmax(s["f1"])
        ax.scatter(
            s["recall"][best], s["precision"][best],
            marker="*", s=150, c="orange", edgecolors="C1",
            linewidths=1, zorder=10, alpha=alpha,
        )
        ax.text(s["recall"][best], s["precision"][best],
                f"{s['f1'][best]:.2f}", fontsize=14)

    if "event" in sweep_results and ax_event is not None:
        s = sweep_results["event"]
        ax_event.plot(
            s["recall"], s["precision"],
            label=f"Event{model_label}",
            linewidth=3, c=colors["event"], linestyle=linestyle, alpha=alpha,
        )
        best = np.nanargmax(s["f1"])
        ax_event.scatter(
            s["recall"][best], s["precision"][best],
            marker="*", s=150, c="orange", edgecolors="C1",
            linewidths=1, zorder=10, alpha=alpha,
        )
        ax_event.text(s["recall"][best], s["precision"][best],
                       f"{s['f1'][best]:.2f}", fontsize=14)
        ax_event.set_ylim(y_lim_event)

    handles, labels_list = ax.get_legend_handles_labels()
    if ax_event is not None:
        h2, l2 = ax_event.get_legend_handles_labels()
        handles += h2
        labels_list += l2
    ax.legend(handles, labels_list, loc="lower left", fontsize=12)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title("Precision-Recall Curve", fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and own_fig:
        plt.show()
    if own_fig:
        plt.close()

    return fig, ax, ax_event


def plot_metrics_vs_thresholds(sweep_results, thresholds, phase="p",
                                fig=None, ax=None,
                                model="", y_lim=None, linestyle="-",
                                alpha=1.0, verbose=False, show=True,
                                save_path=None):
    """Plot precision / recall / F1 vs score thresholds for one phase."""
    own_fig = fig is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=300)

    y_lim = y_lim or (0.6, 1.0)
    model_label = f" ({model})" if model else ""
    s = sweep_results[phase.lower()]
    thresholds = np.asarray(thresholds)

    ax.plot(thresholds, s["recall"],
            label=f"recall{model_label}", linewidth=2, c="C3",
            linestyle=linestyle, alpha=alpha)
    ax.plot(thresholds, s["precision"],
            label=f"precision{model_label}", linewidth=2, c="C0",
            linestyle=linestyle, alpha=alpha)
    ax.plot(thresholds, s["f1"],
            label=f"F1{model_label}", linewidth=2, c="k")

    best = np.nanargmax(s["f1"])
    ax.scatter(thresholds[best], s["f1"][best], marker="*", s=200,
               c="orange", edgecolors="C1", linewidths=1,
               zorder=10, alpha=alpha)
    ax.text(thresholds[best], s["f1"][best],
            f"{s['f1'][best]:.2f}", fontsize=14)

    if verbose:
        for i, t in enumerate(thresholds):
            ax.text(t, 0.8, f"{s['recall'][i]:.4f}", fontsize=10)
            ax.text(t, 0.7, f"{s['precision'][i]:.4f}", fontsize=10)
            ax.text(t, 0.75, f"{s['f1'][i]:.4f}", fontsize=10)

    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlabel("Threshold", fontsize=16)
    ax.set_ylim(y_lim)
    ax.set_title(f"{phase.upper()} VS prob thresholds", fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and own_fig:
        plt.show()
    if own_fig:
        plt.close()

    return fig, ax


def plot_metrics_vs_param(per_bin_stats, param, bins,
                           fig=None, axes=None,
                           interval=0.25, ylim=(0.9, 1.0), alpha=1,
                           title=None, save_path=None, show=True):
    """Plot per-bin precision / recall / F1 as grouped bar chart."""
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    bin_labels = [
        f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)
    ]
    width = 0.25

    for ax_idx, (ax, phase_key, phase_label) in enumerate(
        zip(axes, ["p", "s"], ["P phase", "S phase"])
    ):
        if phase_key not in per_bin_stats:
            continue
        s = per_bin_stats[phase_key]
        prec = np.asarray(s["precision"])
        rec = np.asarray(s["recall"])
        f1 = np.asarray(s["f1"])
        x = np.arange(len(bins) - 1) * interval / width

        ax.bar(x - width, prec, width=width, label="Precision", alpha=alpha)
        ax.bar(x, rec, width=width, label="Recall", alpha=alpha)
        ax.bar(x + width, f1, width=width, label="F1", alpha=alpha)

        ax.text(
            0.05, 0.95, phase_label, va="center", ha="center",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.5, edgecolor="black"),
        )
        if ax_idx == 0:
            ax.set_xticks([])
        else:
            ax.set_xlabel(param)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels)
        ax.legend(loc="lower right", fontsize=12)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and own_fig:
        plt.show()
    if own_fig:
        plt.close()

    return fig, axes


def plot_metrics_vs_param_lineplot(per_bin_stats, param, bins,
                                    fig=None, axes=None,
                                    ylim=(0.9, 1.0), alpha=1,
                                    title=None, save_path=None, show=True):
    """Plot per-bin precision / recall / F1 as line chart."""
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    bin_labels = [
        f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)
    ]
    x = np.arange(len(bins) - 1)

    for ax_idx, (ax, phase_key, phase_label) in enumerate(
        zip(axes, ["p", "s"], ["P phase", "S phase"])
    ):
        if phase_key not in per_bin_stats:
            continue
        s = per_bin_stats[phase_key]
        ax.plot(x, s["precision"], label="Precision", marker="o", alpha=alpha)
        ax.plot(x, s["recall"], label="Recall", marker="o", alpha=alpha)
        ax.plot(x, s["f1"], label="F1", marker="o", alpha=alpha)

        ax.text(
            0.08, 0.90, phase_label, va="center", ha="center",
            transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.5, edgecolor="black"),
        )
        if ax_idx == 0:
            ax.set_xticks([])
        else:
            ax.set_xlabel(param)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels)
            ax.set_ylabel("Score")
        ax.set_ylim(*ylim)
        ax.legend(loc="lower right")
        ax.grid(axis="y")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and own_fig:
        plt.show()
    if own_fig:
        plt.close()

    return fig, axes


def plot_f1_vs_params_comparison(per_bin_stats_list, models, param, bins,
                                  fig=None, axes=None,
                                  interval=0.25, ylim=(0.9, 1.0), alpha=1,
                                  title=None, save_path=None, show=True):
    """Plot F1 scores vs parameter bins for multiple models."""
    assert len(per_bin_stats_list) == len(models)

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    bin_labels = [
        f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)
    ]
    width = 0.25
    x = np.arange(len(bins) - 1) * interval
    n = len(models)

    for ax_idx, (ax, phase_key, phase_label) in enumerate(
        zip(axes, ["p", "s"], ["P phase", "S phase"])
    ):
        for i, (model, stats) in enumerate(
            zip(models, per_bin_stats_list)
        ):
            if phase_key not in stats:
                continue
            f1 = np.asarray(stats[phase_key]["f1"])
            offset = (-n / 2 + i + 0.5) * width
            ax.bar(x + offset, f1, width=width, label=model,
                   color=f"C{i}", alpha=alpha)

        ax.text(
            0.08, 0.90, phase_label, va="center", ha="center",
            transform=ax.transAxes, fontsize=16,
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.5, edgecolor="black"),
        )
        if ax_idx == 0:
            ax.set_xticks([])
        else:
            ax.set_xlabel(param)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels)
            ax.legend(loc="lower right", fontsize=12)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and own_fig:
        plt.show()
    if own_fig:
        plt.close()

    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# Section G: CLI main
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    """CLI entry point for seismic phase picking and event detection metrics."""
    args = parse_args(argv)
    os.makedirs(args.save_path, exist_ok=True)

    # ── Step 1: Ensure GT CSV exists ─────────────────────────────────────────
    if args.h5_file and args.extract_only:
        gt_csv_dir = args.gt_csv_dir or os.path.dirname(args.h5_file)
        extract_gt_from_h5(args.h5_file, gt_csv_dir)
        return

    gt_csv_dir = args.gt_csv_dir
    if gt_csv_dir is None and args.h5_file is not None:
        gt_csv_dir = os.path.dirname(args.h5_file)
    if gt_csv_dir is None:
        raise ValueError("Must provide --gt_csv_dir or --h5_file")

    gt_events_path = Path(gt_csv_dir, "gt_events.csv")
    gt_picks_path = Path(gt_csv_dir, "gt_picks.csv")
    if not gt_events_path.exists() or not gt_picks_path.exists():
        if args.h5_file is None:
            raise FileNotFoundError(
                f"GT CSVs not found in {gt_csv_dir}; provide --h5_file to extract")
        print("GT CSVs not found, extracting from H5...")
        extract_gt_from_h5(args.h5_file, gt_csv_dir)

    gt_events, gt_picks = load_gt(gt_csv_dir)
    print(f"Loaded GT: {len(gt_events)} events, {len(gt_picks)} picks")

    # ── Step 2: Load predictions ─────────────────────────────────────────────
    pick_preds, event_preds = load_predictions(
        args.pick_dir, args.event_dir, args.model)
    print(f"Loaded predictions: {len(pick_preds)} picks, {len(event_preds)} events")

    # ── Step 3: Apply filters ────────────────────────────────────────────────
    # Filter events by ID list
    if args.filter_events is not None:
        with open(args.filter_events) as f:
            filter_ids = set(json.load(f))
        if args.include_events:
            gt_events = gt_events[gt_events["pick_event_id"].isin(filter_ids)]
            gt_picks = gt_picks[gt_picks["event_id"].isin(filter_ids)]
        else:
            gt_events = gt_events[~gt_events["pick_event_id"].isin(filter_ids)]
            gt_picks = gt_picks[~gt_picks["event_id"].isin(filter_ids)]
        print(f"After event filter: {len(gt_events)} events, {len(gt_picks)} picks")

    # ── Step 4b: Normalize event predictions for center matching ─────────────
    # Done BEFORE main_event_only so we can apply pick-style filtering to
    # the normalized preds (which use phase_index in sample units).
    if not event_preds.empty:
        event_preds_norm = normalize_event_preds_for_matching(event_preds)
    else:
        event_preds_norm = pd.DataFrame()

    # ── Step 4: Main event only filter ──────────────────────────────────────
    # gt_picks_raw: main-event-only picks (all stations, including non-manual),
    # used for per-station GT distance/azimuth lookup in cluster_and_match_events.
    gt_picks_raw = gt_picks[
            (gt_picks["pick_event_id"] == gt_picks["event_id"])
        ].copy()
    if args.main_event_only:
        # Pick predictions and event center matching both use MANUAL stations'
        # labels for ALL events at each station (old pick-level behavior:
        # `for e_id in set(e_ids)` — not restricted to main event_id).
        gt_manual_all = gt_picks[gt_picks["is_manual"] == True]

        # Filter pick preds: manual stations, all events
        if not pick_preds.empty:
            pick_preds = filter_predictions_main_event(
                pick_preds, gt_manual_all, is_event_pred=False,
                sample_rate=SAMPLE_RATE, feature_scale=FEATURE_SCALE)

        # Filter event center preds (normalized, phase_index in sample units):
        # same logic as pick preds — manual stations, all events.
        if not event_preds_norm.empty:
            event_preds_norm = filter_predictions_main_event(
                event_preds_norm, gt_manual_all, is_event_pred=False,
                sample_rate=SAMPLE_RATE, feature_scale=FEATURE_SCALE)

        # Filter raw event preds for clustering: ALL stations' labels but
        # MAIN EVENT only (is_event_pred=True filters to pick_event_id ==
        # event_id inside the function), matching old event-level behavior:
        # `for e_id in [event_id]`.
        # keep_unmatched=True: old code keeps stations with no main-event GT.
        if not event_preds.empty:
            event_preds = filter_predictions_main_event(
                event_preds, gt_picks, is_event_pred=True,
                sample_rate=SAMPLE_RATE, feature_scale=FEATURE_SCALE,
                keep_unmatched=False)

        # Restrict gt_picks to manual main-event only for metrics.
        gt_picks = gt_picks[
            (gt_picks["pick_event_id"] == gt_picks["event_id"])
            & (gt_picks["is_manual"] == True)
        ].copy()
        print(f"After main_event_only: {len(gt_picks)} manual GT picks, "
              f"{len(pick_preds)} pred picks, {len(event_preds)} pred events")
    else:
        gt_picks = gt_picks[gt_picks["is_manual"] == True].copy()
        gt_pick_traces = set(gt_picks.apply(lambda r: f"{r['event_id']}/{r['station_id']}", axis=1).to_list())
        pick_preds = pick_preds[pick_preds.apply(lambda r: f"{r['event_id']}/{r['station_id']}", axis=1).isin(gt_pick_traces)].copy()
        event_preds_norm = event_preds_norm[event_preds_norm.apply(lambda r: f"{r['event_id']}/{r['station_id']}", axis=1).isin(gt_pick_traces)].copy()

    # ── Step 5: Matching ─────────────────────────────────────────────────────
    # Convert max_residual_samples to seconds for the matching window
    max_residual_s = args.max_residual_samples / SAMPLE_RATE

    # Pick matching (P and S)
    print("\nMatching picks...")
    pick_matched = match_detections(
        pick_preds, gt_picks,
        phase_types=["P", "S"],
        max_residual_s=max_residual_s,
        match_by=args.match_by,
        sample_rate=SAMPLE_RATE,
        polarity_threshold=args.polarity_threshold,
    )
    print(f"  Pick matched: {len(pick_matched)} rows")

    # Event center matching
    print("Matching event centers...")
    center_matched = match_detections(
        event_preds_norm, gt_picks,
        phase_types=["event"],
        max_residual_s=max_residual_s,
        match_by=args.match_by,
        sample_rate=SAMPLE_RATE,
        polarity_threshold=args.polarity_threshold,
    )
    print(f"  Center matched: {len(center_matched)} rows")

    # Event clustering + matching
    print("Clustering and matching events...")
    event_matched = cluster_and_match_events(
        event_preds[event_preds["event_score"] >= args.score_threshold].copy(), gt_events,
        association_method=args.association_method,
        dbscan_epsilon=args.dbscan_epsilon,
        min_stations=args.min_stations,
        time_space_weight=args.time_space_weight,
        event_dt_bar=args.event_dt_bar,
        sample_rate=SAMPLE_RATE,
        feature_scale=FEATURE_SCALE,
        gt_picks=gt_picks_raw,
    )
    print(f"  Event matched: {len(event_matched)} rows")

    # ── Step 6: Apply score threshold and compute metrics ────────────────────
    suffix = args.suffix

    print(f"\n{'='*60}")
    print(f"Metrics at score_threshold={args.score_threshold}")
    print(f"{'='*60}")

    # Pick metrics
    if not pick_matched.empty:
        pick_active = apply_score_threshold(pick_matched, args.score_threshold)
        print("\n--- Pick Detection ---")
        pick_stats = plot_residual_histogram(
            pick_active, dt_bar=args.pick_dt_bar,
            distribution_bar=args.distribution_bar,
            save_path=os.path.join(
                args.save_path, f"phase_picking{suffix}.png"),
        )

        # Polarity confusion
        plot_polarity_confusion(
            pick_active, dt_bar=args.pick_dt_bar,
            save_path=os.path.join(
                args.save_path, f"polarity_confusion{suffix}.png"),
        )
    else:
        pick_stats = {}
        print("\n--- Pick Detection: No data ---")

    # Event center metrics
    if not center_matched.empty:
        center_active = apply_score_threshold(
            center_matched, args.score_threshold)
        print("\n--- Event Center Detection ---")
        center_stats = plot_event_center_residuals(
            center_active, dt_bar=args.event_center_dt_bar,
            save_path=os.path.join(
                args.save_path, f"event_center{suffix}.png"),
        )
    else:
        center_stats = {}
        print("\n--- Event Center Detection: No data ---")

    # Clustered event metrics
    if not event_matched.empty:
        print("\n--- Clustered Event Detection ---")
        event_stats = plot_event_residuals(
            event_matched, event_dt_bar=args.event_dt_bar,
            save_path=os.path.join(
                args.save_path, f"event_residuals{suffix}.png"),
        )
        # Association statistics
        print("\n--- Association Statistics ---")
        plot_association_stats(
            event_matched,
            save_path=os.path.join(
                args.save_path, f"association_stats{suffix}.png"),
            show=False,
        )
    else:
        event_stats = {}
        print("\n--- Clustered Event Detection: No data ---")

    # Station-count subset metrics
    if args.station_threshold > 0 and not gt_events.empty:
        high_sta_events = gt_events[
            gt_events["num_stations"] >= args.station_threshold
        ]["pick_event_id"].values
        if not pick_matched.empty:
            pick_high = pick_matched[
                pick_matched["gt_pick_event_id"].isin(high_sta_events)
                | pick_matched["match_type"].eq("unmatched_pred")
            ]
            if not pick_high.empty:
                pick_high_active = apply_score_threshold(
                    pick_high, args.score_threshold)
                print(f"\n--- Pick Detection "
                      f"(stations >= {args.station_threshold}) ---")
                plot_residual_histogram(
                    pick_high_active, dt_bar=args.pick_dt_bar,
                    distribution_bar=args.distribution_bar,
                    save_path=os.path.join(
                        args.save_path,
                        f"phase_picking_sta{args.station_threshold}"
                        f"{suffix}.png"),
                )

    # ── Step 7: Save intermediate results ────────────────────────────────────
    if not pick_matched.empty:
        pick_matched.to_csv(
            os.path.join(args.save_path, f"pick_matched{suffix}.csv"),
            index=False)
    if not center_matched.empty:
        center_matched.to_csv(
            os.path.join(args.save_path, f"center_matched{suffix}.csv"),
            index=False)
    if not event_matched.empty:
        event_matched.to_csv(
            os.path.join(args.save_path, f"event_matched{suffix}.csv"),
            index=False)

    # Save summary stats as JSON
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "pick_stats": {k: (float(v) if isinstance(v, (int, float, np.floating))
                           else v)
                       for k, v in pick_stats.items()},
        "center_stats": {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else v)
                         for k, v in center_stats.items()},
        "event_stats": {k: (float(v) if isinstance(v, (int, float, np.floating))
                            else v)
                        for k, v in event_stats.items()},
    }
    with open(os.path.join(args.save_path, f"summary{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {args.save_path}/summary{suffix}.json")

    # ── Step 8: Threshold sweep for PR curves ────────────────────────────────
    if args.sweep_thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold sweep: {args.thresholds}")
        print(f"{'='*60}")

        sweep_results = {}

        # Pick sweep
        if not pick_matched.empty:
            pick_sweep = sweep_thresholds(
                pick_matched, args.thresholds, dt_bar=args.pick_dt_bar)
            sweep_results.update(pick_sweep)

        # Event center sweep
        if not center_matched.empty:
            center_sweep = sweep_thresholds(
                center_matched, args.thresholds,
                dt_bar=args.event_center_dt_bar)
            if "event" in center_sweep:
                sweep_results["event"] = center_sweep["event"]

        if sweep_results:
            # PR curve
            plot_pr_curve(
                sweep_results, thresholds=args.thresholds,
                save_path=os.path.join(
                    args.save_path, f"pr_curve{suffix}.png"),
                show=False,
            )

            # Metrics vs thresholds (per phase)
            for phase in ["p", "s"]:
                if phase in sweep_results:
                    plot_metrics_vs_thresholds(
                        sweep_results, args.thresholds, phase=phase,
                        save_path=os.path.join(
                            args.save_path,
                            f"metrics_vs_threshold_{phase}{suffix}.png"),
                        show=False,
                    )

            # Save sweep results
            sweep_data = {}
            for k, v in sweep_results.items():
                sweep_data[k] = {
                    m: v[m].tolist() if hasattr(v[m], "tolist") else v[m]
                    for m in v
                }
            sweep_data["thresholds"] = args.thresholds
            with open(os.path.join(args.save_path,
                                   f"sweep{suffix}.json"), "w") as f:
                json.dump(sweep_data, f, indent=2)
            print(f"Saved sweep results to {args.save_path}/sweep{suffix}.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
