"""Run PhaseNet (unet2018) on gs://quakeflow_das/ridgecrest_north.

UNet2018 uses (1, K) kernels that only process along the time axis, so the
spatial DAS dimension (nx) is treated independently — equivalent to running
1D PhaseNet on each DAS channel in parallel.  The single DAS component is
repeated 3× to match the pretrained 3-component (ENZ) input weights.
"""
import json
import os
import sys

sys.path.insert(0, ".")

import fsspec
import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data

matplotlib.use("agg")

import matplotlib.pyplot as plt

import eqnet
from eqnet.utils import detect_peaks, extract_picks
from eqnet.utils.detect_peaks_1d import detect_peaks as detect_peaks_1d

EVENT_ID = "ci38595978"   # largest phasenet picks file
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIGURE_DIR = "tests/figures/quakeflow_das"
PHASES = ["P", "S", "N"]
MIN_PROB = 0.3
BUCKET = "gs://quakeflow_das/ridgecrest_north/data"
PHASENET_PICKS = "gs://quakeflow_das/ridgecrest_north/phasenet/picks"
VP_VS = 1.73        # Vp/Vs ratio used for P-S origin time estimation
MAX_PS_DT = 20.0    # maximum P-S separation (s) to form a valid pair

os.makedirs(FIGURE_DIR, exist_ok=True)


def get_storage_options():
    creds = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(creds):
        with open(creds) as f:
            return {"token": json.load(f)}
    return {}


class DASDataset(torch.utils.data.IterableDataset):
    """Simple IterableDataset for quakeflow_das h5 files.

    Each h5 file contains one seismic event recorded on a DAS array.
    Data layout: dataset["data"] of shape (nx, nt), with attributes
    dt_s, dx_m, begin_time, event_time_index.

    Yields one dict per file with keys:
        data            (1, nx, nt) float32 tensor, demeaned + median-removed
        file_name       str  — event id (h5 stem)
        begin_time      str  — ISO timestamp of first sample
        event_time_index int — sample index of the event origin time
        dt_s            float
        dx_m            float
        nx              int  — original number of channels
        nt              int  — original number of time samples
    """

    def __init__(self, data_path: str):
        super().__init__()
        storage_opts = get_storage_options()
        self.storage_opts = storage_opts
        fs = fsspec.filesystem("gcs", **storage_opts)
        path_clean = data_path.removeprefix("gs://")
        if path_clean.endswith(".h5"):
            self.files = [f"gs://{path_clean}"]
        else:
            self.files = sorted(f"gs://{p}" for p in fs.glob(f"{path_clean}/*.h5"))
        print(f"DASDataset: {len(self.files)} files")

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        fs = fsspec.filesystem("gcs", **self.storage_opts)
        for path in files:
            try:
                with fs.open(path, "rb") as f:
                    with h5py.File(f, "r") as fp:
                        ds = fp["data"]
                        data = ds[()].astype(np.float32)  # (nx, nt)
                        attrs = dict(ds.attrs)

                dt_s = float(attrs.get("dt_s", 0.01))
                dx_m = float(attrs.get("dx_m", 8.0))
                begin_time = str(attrs.get("begin_time", ""))
                event_time_index = int(attrs.get("event_time_index", -1))

            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            nx, nt = data.shape

            # Preprocessing: demean along time, remove common-mode noise
            data -= data.mean(axis=-1, keepdims=True)
            data -= np.median(data, axis=0, keepdims=True)

            # (nx, nt) -> (1, nx, nt)
            data = torch.from_numpy(data[np.newaxis])

            stem = path.split("/")[-1].replace(".h5", "")
            yield {
                "data": data,
                "file_name": stem,
                "begin_time": begin_time,
                "event_time_index": event_time_index,
                "dt_s": dt_s,
                "dx_m": dx_m,
                "nx": nx,
                "nt": nt,
            }


def collate(batch):
    """Stack a list of samples into a batch dict."""
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals  # strings
    return out


def sliding_window_predict(model, data, nx_win=3000, overlap=0.5, device="cpu"):
    """Sliding window inference along the nx (spatial) axis with score averaging.

    Slides a window of width nx_win with the given overlap ratio, runs the
    model on each window, and averages predictions in overlapping regions.
    The model always receives tensors of exactly nx_win channels; the last
    window is zero-padded if the array is shorter than nx_win.

    Args:
        model  : PhaseNet model in eval mode
        data   : (B, 1, nx, nt) float tensor
        nx_win : spatial window size the model was trained with
        overlap: fraction of window to overlap between consecutive windows
        device : torch device string

    Returns:
        scores : (B, 3, nx, nt) averaged softmax probabilities
    """
    B, C, nx, nt = data.shape
    stride = max(1, int(nx_win * (1 - overlap)))

    # Build start positions: uniform stride, plus a final window flush with nx
    starts = list(range(0, max(1, nx - nx_win + 1), stride))
    if not starts:
        starts = [0]
    if starts[-1] + nx_win < nx:          # ensure full coverage
        starts.append(nx - nx_win)

    scores_sum = torch.zeros(B, 3, nx, nt)
    counts     = torch.zeros(B, 1, nx, nt)

    with torch.inference_mode():
        for s in starts:
            end = min(s + nx_win, nx)
            patch = data[:, :, s:end, :]

            # Zero-pad to nx_win so model always sees the trained input size
            if patch.shape[2] < nx_win:
                patch = torch.nn.functional.pad(patch, (0, 0, 0, nx_win - patch.shape[2]))

            # Build 3-channel input: DAS in channel 0, zeros in channels 1 & 2.
            # Normalize by std across all 3 channels together.  Because channels
            # 1 & 2 are zero, the overall std is reduced by ~1/sqrt(3) vs a
            # full 3-channel signal, so channel 0 is amplified by sqrt(3) — the
            # correct compensation for having only one active component.
            # This is more principled than repeating (repeat sums three
            # independent kernel sets onto the same signal).
            batch = torch.zeros(B, 3, nx_win, nt, dtype=patch.dtype)
            batch[:, 0:1] = patch
            std = batch.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            batch = (batch / std).to(device)
            logits = model({"data": batch})["phase"]           # (B, 3, nx_win, nt)
            score  = torch.softmax(logits, dim=1).cpu()

            n_valid = end - s
            scores_sum[:, :, s:end, :] += score[:, :, :n_valid, :]
            counts    [:, :, s:end, :] += 1

    return scores_sum / counts.clamp(min=1)


def ps_origin_times(picks_df, ch_col, dt_s, dx_m,
                    max_ps_dt=MAX_PS_DT, vp_vs=VP_VS):
    """Estimate origin times from all valid P-S pick pairs per channel.

    Returns a DataFrame with one row per valid pair:
        channel, ch_km, tp (s), ts (s), t0 (s)
    """
    empty = pd.DataFrame(columns=["channel", "ch_km", "tp", "ts", "t0"])
    if picks_df is None or picks_df.empty:
        return empty

    p_df = picks_df[picks_df["phase_type"] == "P"]
    s_df = picks_df[picks_df["phase_type"] == "S"]
    common_chs = sorted(set(p_df[ch_col].unique()) & set(s_df[ch_col].unique()))
    if not common_chs:
        return empty

    batches = []
    for ch in common_chs:
        tp_vals = p_df.loc[p_df[ch_col] == ch, "phase_index"].values * dt_s
        ts_vals = s_df.loc[s_df[ch_col] == ch, "phase_index"].values * dt_s
        tp_grid, ts_grid = np.meshgrid(tp_vals, ts_vals, indexing="ij")  # (N_P, N_S)
        dt_ps = ts_grid - tp_grid
        valid = (dt_ps > 0) & (dt_ps < max_ps_dt)
        if not valid.any():
            continue
        tp_v = tp_grid[valid]
        ts_v = ts_grid[valid]
        t0_v = tp_v - dt_ps[valid] / (vp_vs - 1)
        batches.append(pd.DataFrame({
            "channel": int(ch),
            "ch_km": ch * dx_m / 1000,
            "tp": tp_v,
            "ts": ts_v,
            "t0": t0_v,
        }))

    return pd.concat(batches, ignore_index=True) if batches else empty


def detect_origin_peaks(t0_s, t_max, bin_size=1.0, mpd=5, mph=100):
    """Histogram origin-time estimates and detect peaks.

    Args:
        t0_s     : (N,) array of estimated origin times (s)
        t_max    : right edge of histogram (s)
        bin_size : bin width in seconds (default 1.0)
        mpd      : minimum peak distance in bins (default 5)
        mph      : minimum peak height — count (default 100)

    Returns:
        bin_centers : (M,) bin centre times (s)
        counts      : (M,) histogram counts
        peak_times  : (K,) detected origin times (s)
    """
    edges = np.arange(0, t_max + bin_size, bin_size)
    counts, _ = np.histogram(np.asarray(t0_s), bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() == 0:
        return bin_centers, counts, np.array([])
    peak_inds, _ = detect_peaks_1d(counts, mpd=mpd, mph=mph)
    peak_times = bin_centers[peak_inds] if len(peak_inds) else np.array([])
    return bin_centers, counts, peak_times


def filter_origin_times(t0_df, peak_times, window=3.0):
    """For each detected peak, keep P-S pairs within 1 MAD of the median.

    The origin-time residuals follow a Laplacian distribution, so we use
    robust statistics instead of mean ± std:
      - center : median of t0 values near the peak
      - scale  : MAD = median(|t0 - center|)  — Laplacian analog of σ

    Steps per peak:
      1. Select pairs where |t0 - t_peak| < window (3 s)
      2. Compute center = median, scale = MAD of those t0 values
      3. Keep pairs where |t0 - center| < scale  (1 MAD)

    Returns the union of survivors across all peaks.
    """
    if t0_df.empty or len(peak_times) == 0:
        return t0_df.iloc[0:0]

    t0_vals = t0_df["t0"].values
    mask = np.zeros(len(t0_df), dtype=bool)
    for t_peak in peak_times:
        near = np.abs(t0_vals - t_peak) < window
        if near.sum() < 2:
            continue
        t0_near = t0_vals[near]
        center = float(np.median(t0_near))
        mad    = float(np.median(np.abs(t0_near - center)))
        scale  = max(mad, 0.1)          # floor at 0.1 s
        mask |= np.abs(t0_vals - center) < scale

    return t0_df[mask].reset_index(drop=True)


def plot_das_picks(data, our_picks, ref_df, dt_s, dx_m, figure_dir, event_id):
    """3×2 grid layout:
         col 0 (ours)              col 1 (reference)
      row 0  raw picks             raw picks
      row 1  origin-filtered picks filtered picks
      row 2  origin-time histogram origin-time histogram
    """
    wave = data[0, 0].numpy()          # (nx, nt)
    nx_w, nt_w = wave.shape
    t_max = nt_w * dt_s
    x_max = nx_w * dx_m / 1000        # km

    std = wave.std() or 1.0
    wave_norm = np.clip(wave / (2 * std), -1, 1)

    ref_ch_col = None
    if ref_df is not None and not ref_df.empty:
        for candidate in ("channel_index", "station_id", "station_index"):
            if candidate in ref_df.columns:
                ref_ch_col = candidate
                break
    has_ref = ref_ch_col is not None

    fig, axes = plt.subplots(
        3, 2, figsize=(20, 18), sharex=True,
        gridspec_kw={"height_ratios": [5, 5, 2]},
    )

    def draw_bg(ax):
        ax.imshow(wave_norm, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                  alpha=0.3, extent=[0, t_max, x_max, 0], interpolation="nearest")
        ax.set_ylabel("Distance (km)")
        ax.set_xlim(0, t_max)
        ax.set_ylim(x_max, 0)

    def draw_vlines(ax, times, color="limegreen"):
        for t in times:
            ax.axvline(t, color=color, lw=1.5, ls="--", alpha=0.9, zorder=6)

    def scatter_raw(ax, picks_list):
        for ph, color in {"P": "blue", "S": "red", "N": "gray"}.items():
            ch_v = [int(p["station_id"]) * dx_m / 1000
                    for p in picks_list if p["phase_type"] == ph]
            t_v  = [p["phase_index"] * dt_s
                    for p in picks_list if p["phase_type"] == ph]
            if ch_v:
                ax.scatter(t_v, ch_v, s=4, color=color, alpha=0.8,
                           marker="o", linewidths=0, label=ph)

    def scatter_filtered(ax, df):
        if df.empty:
            return
        # deduplicate P and S picks (multiple pairs may share one pick)
        p_uniq = df.drop_duplicates(subset=["channel", "tp"])
        s_uniq = df.drop_duplicates(subset=["channel", "ts"])
        ax.scatter(p_uniq["tp"], p_uniq["ch_km"], s=4, color="blue",
                   alpha=0.8, marker="o", linewidths=0, label="P")
        ax.scatter(s_uniq["ts"], s_uniq["ch_km"], s=4, color="red",
                   alpha=0.8, marker="o", linewidths=0, label="S")
        ax.scatter(df["t0"], df["ch_km"], s=6, color="limegreen",
                   alpha=0.5, marker="D", linewidths=0, label="t₀")

    def draw_hist(ax, bins, counts, peak_times, color, label):
        ax.bar(bins, counts, width=0.9, color=color, alpha=0.8, label=label)
        for t in peak_times:
            ax.axvline(t, color=color, lw=1.5, ls="--")
        ax.set_ylabel("P-S pairs")
        ax.set_xlabel("Estimated origin time (s)")

    # ------------------------------------------------------------------
    # Our picks — compute origin times and filter
    # ------------------------------------------------------------------
    our_list = our_picks[0] if our_picks else []
    our_df   = pd.DataFrame(our_list)
    t0_df_our = pd.DataFrame(columns=["channel", "ch_km", "tp", "ts", "t0"])
    if not our_df.empty:
        our_df["station_id"] = our_df["station_id"].astype(int)
        t0_df_our = ps_origin_times(our_df, "station_id", dt_s, dx_m)

    bins_our, cnt_our, peak_times_our = detect_origin_peaks(
        t0_df_our["t0"].values if not t0_df_our.empty else np.array([]), t_max)
    t0_filt_our = filter_origin_times(t0_df_our, peak_times_our)

    draw_bg(axes[0, 0])
    scatter_raw(axes[0, 0], our_list)
    draw_vlines(axes[0, 0], peak_times_our)
    axes[0, 0].set_title(
        f"{event_id} — our picks  (n={len(our_list)},  "
        f"origins: {', '.join(f'{t:.1f}s' for t in peak_times_our)})"
    )
    axes[0, 0].legend(loc="upper right", fontsize=9, markerscale=2)

    draw_bg(axes[1, 0])
    scatter_filtered(axes[1, 0], t0_filt_our)
    draw_vlines(axes[1, 0], peak_times_our)
    axes[1, 0].set_title(
        f"our filtered picks  (pairs={len(t0_filt_our)},  "
        f"σ-trimmed per event)"
    )
    axes[1, 0].legend(loc="upper right", fontsize=9, markerscale=2)

    draw_hist(axes[2, 0], bins_our, cnt_our, peak_times_our, "steelblue", "ours")
    axes[2, 0].set_title("Origin time histogram (ours)")

    # ------------------------------------------------------------------
    # Reference picks — compute origin times and filter
    # ------------------------------------------------------------------
    if has_ref:
        t0_df_ref = ps_origin_times(ref_df, ref_ch_col, dt_s, dx_m)
        bins_ref, cnt_ref, peak_times_ref = detect_origin_peaks(
            t0_df_ref["t0"].values if not t0_df_ref.empty else np.array([]), t_max)
        t0_filt_ref = filter_origin_times(t0_df_ref, peak_times_ref)

        draw_bg(axes[0, 1])
        for ph, color in {"P": "blue", "S": "red"}.items():
            sub = ref_df[ref_df["phase_type"] == ph]
            if not sub.empty:
                axes[0, 1].scatter(sub["phase_index"].values * dt_s,
                                   sub[ref_ch_col].values * dx_m / 1000,
                                   s=4, color=color, alpha=0.8,
                                   marker="o", linewidths=0, label=ph)
        draw_vlines(axes[0, 1], peak_times_ref)
        axes[0, 1].set_title(
            f"reference picks  (n={len(ref_df)},  "
            f"origins: {', '.join(f'{t:.1f}s' for t in peak_times_ref)})"
        )
        axes[0, 1].legend(loc="upper right", fontsize=9, markerscale=2)

        draw_bg(axes[1, 1])
        scatter_filtered(axes[1, 1], t0_filt_ref)
        draw_vlines(axes[1, 1], peak_times_ref)
        axes[1, 1].set_title(
            f"reference filtered picks  (pairs={len(t0_filt_ref)},  "
            f"σ-trimmed per event)"
        )
        axes[1, 1].legend(loc="upper right", fontsize=9, markerscale=2)

        draw_hist(axes[2, 1], bins_ref, cnt_ref, peak_times_ref, "darkorange", "reference")
        axes[2, 1].set_title("Origin time histogram (reference)")
    else:
        for row in range(3):
            axes[row, 1].set_visible(False)

    fig.suptitle(event_id, fontsize=13, y=1.005)
    fig.tight_layout()
    fname = os.path.join(figure_dir, f"{event_id}_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Detected origins — ours:      {peak_times_our.tolist()}")
    if has_ref:
        print(f"  Detected origins — reference: {peak_times_ref.tolist()}")


# ------------------------------------------------------------------
# Load model  (pretrained phasenet2018: n_channel=3, n_class=3)
# ------------------------------------------------------------------
model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-2018/phasenet2018.pth"
model = eqnet.models.phasenet.build_model(backbone="unet2018")
ckpt = torch.hub.load_state_dict_from_url(
    model_url, model_dir="./model_phasenet", progress=True, check_hash=True, map_location="cpu"
)
model.load_state_dict(ckpt["model"])
model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}")

# ------------------------------------------------------------------
# DataLoader — single event
# ------------------------------------------------------------------
dataset = DASDataset(f"{BUCKET}/{EVENT_ID}.h5")
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate
)

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
meta = next(iter(loader))

nx = meta["nx"][0].item()
nt = meta["nt"][0].item()
print(f"  data shape: {tuple(meta['data'].shape)}  nx={nx}  nt={nt}")

# ------------------------------------------------------------------
# Load reference phasenet picks from GCS
# ------------------------------------------------------------------
fs = fsspec.filesystem("gcs", **get_storage_options())
with fs.open(f"quakeflow_das/ridgecrest_north/phasenet/picks/{EVENT_ID}.csv") as f:
    ref_df = pd.read_csv(f)
ref_picks = [ref_df.to_dict("records")]
all_phases = sorted(ref_df["phase_type"].unique())

# ------------------------------------------------------------------
# Inference (10% overlap along nx axis)
# ------------------------------------------------------------------
dt = meta["dt_s"][0].item()

scores = sliding_window_predict(model, meta["data"], nx_win=3000, overlap=0.1, device=DEVICE)

topk_scores, topk_inds = detect_peaks(scores, vmin=MIN_PROB, kernel=21, dt=dt)
picks = extract_picks(
    topk_inds,
    topk_scores,
    file_name=meta["file_name"],
    dt=meta["dt_s"],
    vmin=MIN_PROB,
    phases=PHASES,
)

# ------------------------------------------------------------------
# Compare pick counts by phase type
# ------------------------------------------------------------------
our_df = pd.DataFrame(picks[0]) if picks[0] else pd.DataFrame(columns=["phase_type"])

print(f"\nPick comparison for {EVENT_ID}:")
print(f"  {'Phase':<8} {'Ours':>8} {'Reference':>10}")
print(f"  {'-'*28}")
for phase in all_phases:
    n_ours = (our_df["phase_type"] == phase).sum() if not our_df.empty else 0
    n_ref  = (ref_df["phase_type"]  == phase).sum()
    print(f"  {phase:<8} {n_ours:>8} {n_ref:>10}")
print(f"  {'-'*28}")
print(f"  {'Total':<8} {len(our_df):>8} {len(ref_df):>10}")

# ------------------------------------------------------------------
# Plot: our picks vs. reference overlaid on DAS waveform
# ------------------------------------------------------------------
plot_das_picks(
    meta["data"].float(),
    picks,
    ref_df,
    dt_s=meta["dt_s"][0].item(),
    dx_m=meta["dx_m"][0].item(),
    figure_dir=FIGURE_DIR,
    event_id=EVENT_ID,
)

print(f"\nDone. Figures saved to {FIGURE_DIR}/")
