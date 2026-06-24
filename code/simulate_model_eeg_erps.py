#!/usr/bin/env python3
"""Generate simulated BioSemi64 MNE ERPs from basal-ganglia model output."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

DEFAULT_MODEL_PREFIX = Path(
    "/Users/mq20185996/projects/cat_learn_automaticity_models/output/"
    "model_spiking_cat_90vs180_vectorized_90"
)

SFREQ = 1000.0
TMIN = -1.0
STIM_SAMPLE = 1000
BIOSEMI_MONTAGE = "biosemi64"

MODEL_UNITS = {
    0: ("DMS_A", "DMS", "A"),
    1: ("DMS_B", "DMS", "B"),
    2: ("PM_A", "PM", "A"),
    3: ("PM_B", "PM", "B"),
    4: ("DLS_A", "DLS", "A"),
    5: ("DLS_B", "DLS", "B"),
    6: ("M1_A", "M1", "A"),
    7: ("M1_B", "M1", "B"),
}

# A is the left-hand button, so its dominant motor loop is right hemisphere.
RESPONSE_DOMINANT_HEMI = {"A": "R", "B": "L"}

LATERALITY = {
    "DMS": 0.55,
    "DLS": 0.75,
    "PM": 0.85,
    "M1": 0.95,
}

# MNI coordinates in millimetres. These are intentionally coarse template ROIs:
# caudate/associative striatum, posterior putamen/sensorimotor striatum, dorsal
# premotor hand-region, and M1 hand knob.
ROI_MNI_R = {
    "DMS": (12.0, 10.0, 10.0),
    "DLS": (28.0, -4.0, 4.0),
    "PM": (28.0, -8.0, 58.0),
    "M1": (38.0, -24.0, 58.0),
}

ROI_SOURCE_SCALE = {
    "DMS": 0.15,
    "DLS": 0.25,
    "PM": 0.70,
    "M1": 1.00,
}

OUTPUT_STEM = "simulated_model_biosemi64"


@dataclass(frozen=True)
class SourceSpec:
    source_index: int
    model_index: int
    model_unit: str
    roi: str
    response_pathway: str
    hemi: str
    mni_x: float
    mni_y: float
    mni_z: float
    laterality_gain: float
    roi_scale: float
    total_gain: float
    dominant: bool


def hemi_mni(roi: str, hemi: str) -> tuple[float, float, float]:
    x, y, z = ROI_MNI_R[roi]
    if hemi == "L":
        x = -x
    return x, y, z


def build_source_specs() -> list[SourceSpec]:
    specs = []
    for model_index, (model_unit, roi, pathway) in MODEL_UNITS.items():
        dominant_hemi = RESPONSE_DOMINANT_HEMI[pathway]
        dominant_gain = LATERALITY[roi]
        weak_gain = 1.0 - dominant_gain
        for hemi in ("L", "R"):
            is_dominant = hemi == dominant_hemi
            laterality_gain = dominant_gain if is_dominant else weak_gain
            roi_scale = ROI_SOURCE_SCALE[roi]
            specs.append(
                SourceSpec(
                    source_index=len(specs),
                    model_index=model_index,
                    model_unit=model_unit,
                    roi=roi,
                    response_pathway=pathway,
                    hemi=hemi,
                    mni_x=hemi_mni(roi, hemi)[0],
                    mni_y=hemi_mni(roi, hemi)[1],
                    mni_z=hemi_mni(roi, hemi)[2],
                    laterality_gain=laterality_gain,
                    roi_scale=roi_scale,
                    total_gain=laterality_gain * roi_scale,
                    dominant=is_dominant,
                )
            )
    return specs


def robust_zscore(data: np.ndarray) -> np.ndarray:
    median = np.nanmedian(data, axis=(1, 2, 3), keepdims=True)
    q25 = np.nanpercentile(data, 25, axis=(1, 2, 3), keepdims=True)
    q75 = np.nanpercentile(data, 75, axis=(1, 2, 3), keepdims=True)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(data, axis=(1, 2, 3), keepdims=True)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (data - median) / scale


def biosemi_info_and_positions() -> tuple[mne.Info, list[str], np.ndarray]:
    montage = mne.channels.make_standard_montage(BIOSEMI_MONTAGE)
    ch_names = list(montage.ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    info.set_montage(montage)
    positions = np.array([montage.get_positions()["ch_pos"][ch] for ch in ch_names])
    return info, ch_names, positions


def dipole_orientation(mni_m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mni_m)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return mni_m / norm


def build_leadfield(sensor_pos_m: np.ndarray, sources: list[SourceSpec]) -> np.ndarray:
    leadfield = np.zeros((sensor_pos_m.shape[0], len(sources)), dtype=float)
    for source in sources:
        src_m = np.array([source.mni_x, source.mni_y, source.mni_z], dtype=float) / 1000.0
        q = dipole_orientation(src_m)
        displacement = sensor_pos_m - src_m[None, :]
        distance = np.linalg.norm(displacement, axis=1)
        distance = np.maximum(distance, 0.01)
        potential = displacement @ q / (distance**3)
        potential -= potential.mean()
        rms = np.sqrt(np.mean(potential**2))
        if rms > 0:
            potential /= rms
        leadfield[:, source.source_index] = potential * source.total_gain
    return leadfield


def build_source_timeseries(g: np.ndarray, sources: list[SourceSpec]) -> np.ndarray:
    z = robust_zscore(g.astype(float, copy=False))
    n_sim, n_trials, n_times = z.shape[1], z.shape[2], z.shape[3]
    source_ts = np.zeros((n_sim, n_trials, len(sources), n_times), dtype=np.float32)
    for source in sources:
        source_ts[:, :, source.source_index, :] = z[source.model_index]
    return source_ts


def load_model_inputs(model_prefix: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    g_path = model_prefix.with_name(model_prefix.name + "_g.npy")
    cat_path = model_prefix.with_name(model_prefix.name + "_cat.npy")
    ds_path = model_prefix.with_name(model_prefix.name + "_ds.csv")
    if not g_path.exists():
        raise FileNotFoundError(f"Missing model signal file: {g_path}")
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing model category file: {cat_path}")
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing model trial table: {ds_path}")
    return np.load(g_path), np.load(cat_path), pd.read_csv(ds_path)


def make_epochs(
    eeg_v: np.ndarray,
    cat: np.ndarray,
    ds: pd.DataFrame,
    info: mne.Info,
) -> mne.EpochsArray:
    labels = cat.reshape(-1)
    n_sim, n_trials = cat.shape
    data = eeg_v.reshape(n_sim * n_trials, eeg_v.shape[2], eeg_v.shape[3])
    valid = np.isin(labels, [1, 2])
    data = data[valid]
    labels = labels[valid].astype(int)

    sim_index = np.repeat(np.arange(n_sim), n_trials)[valid]
    trial_index = np.tile(np.arange(n_trials), n_sim)[valid]
    ds_aligned = ds.iloc[trial_index].reset_index(drop=True)
    metadata = pd.DataFrame(
        {
            "simulation": sim_index,
            "model_trial": trial_index,
            "cat_num": labels,
            "cat": np.where(labels == 1, "A", "B"),
            "phase": ds_aligned["phase"].astype(str).to_numpy(),
            "x": ds_aligned["x"].to_numpy(),
            "y": ds_aligned["y"].to_numpy(),
        }
    )

    events = np.column_stack(
        [
            np.arange(len(labels), dtype=int) * int(eeg_v.shape[-1]),
            np.zeros(len(labels), dtype=int),
            labels,
        ]
    )
    event_id = {"Stim/A": 1, "Stim/B": 2}
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=event_id,
        tmin=TMIN,
        metadata=metadata,
        verbose="ERROR",
    )
    return epochs


def save_erp_figures(evoked_a, evoked_b, evoked_diff, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = mne.viz.plot_compare_evokeds(
        {"A": evoked_a, "B": evoked_b},
        picks="eeg",
        combine="gfp",
        show=False,
        title="Simulated BioSemi64 ERP: A vs B",
    )
    fig[0].savefig(figures_dir / f"{OUTPUT_STEM}_erp_a_vs_b_gfp.png", dpi=150)
    plt.close(fig[0])

    fig = evoked_diff.plot(
        spatial_colors=True,
        show=False,
        titles="Simulated BioSemi64 ERP: A - B",
    )
    fig.savefig(figures_dir / f"{OUTPUT_STEM}_erp_a_minus_b_butterfly.png", dpi=150)
    plt.close(fig)

    fig = evoked_diff.plot_topomap(
        times=[-0.2, 0.0, 0.2, 0.5, 1.0],
        ch_type="eeg",
        show=False,
        time_unit="s",
    )
    fig.savefig(figures_dir / f"{OUTPUT_STEM}_erp_a_minus_b_topomap.png", dpi=150)
    plt.close(fig)


def save_outputs(
    model_prefix: Path,
    epochs: mne.EpochsArray,
    source_ts: np.ndarray,
    leadfield: np.ndarray,
    sources: list[SourceSpec],
    ch_names: list[str],
    output_dir: Path,
    figures_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    evoked_a = epochs["Stim/A"].average()
    evoked_b = epochs["Stim/B"].average()
    evoked_diff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
    evoked_diff.comment = "Stim/A - Stim/B"

    paths = {
        "epochs": output_dir / f"{OUTPUT_STEM}_epochs-epo.fif",
        "evoked_a": output_dir / f"{OUTPUT_STEM}_evoked-a-ave.fif",
        "evoked_b": output_dir / f"{OUTPUT_STEM}_evoked-b-ave.fif",
        "evoked_diff": output_dir / f"{OUTPUT_STEM}_evoked-a_minus_b-ave.fif",
        "source_timeseries": output_dir / f"{OUTPUT_STEM}_source_timeseries.npy",
        "leadfield": output_dir / f"{OUTPUT_STEM}_leadfield.npy",
        "sources": output_dir / f"{OUTPUT_STEM}_sources.csv",
        "channels": output_dir / f"{OUTPUT_STEM}_channels.csv",
        "info": output_dir / f"{OUTPUT_STEM}_info.json",
    }

    epochs.save(paths["epochs"], overwrite=True)
    evoked_a.save(paths["evoked_a"], overwrite=True)
    evoked_b.save(paths["evoked_b"], overwrite=True)
    evoked_diff.save(paths["evoked_diff"], overwrite=True)
    np.save(paths["source_timeseries"], source_ts.astype(np.float32))
    np.save(paths["leadfield"], leadfield.astype(np.float32))
    pd.DataFrame([source.__dict__ for source in sources]).to_csv(paths["sources"], index=False)
    pd.DataFrame(
        {"channel_index": np.arange(len(ch_names), dtype=int), "channel_name": ch_names}
    ).to_csv(paths["channels"], index=False)

    info_payload = {
        "input_model_prefix": str(model_prefix),
        "signal": "g",
        "montage": BIOSEMI_MONTAGE,
        "sfreq_hz": SFREQ,
        "tmin_sec": TMIN,
        "stimulus_sample": STIM_SAMPLE,
        "data_units": "volts",
        "n_epochs": len(epochs),
        "n_channels": len(ch_names),
        "n_times": len(epochs.times),
        "category_counts": epochs.metadata["cat"].value_counts().to_dict(),
        "laterality": LATERALITY,
        "roi_source_scale": ROI_SOURCE_SCALE,
        "mni_coordinates_right": ROI_MNI_R,
        "notes": (
            "No subject MRI is used. MNI ROI dipoles are projected to the BioSemi64 "
            "standard montage with a crude homogeneous dipole field and average reference."
        ),
    }
    paths["info"].write_text(json.dumps(info_payload, indent=2))
    save_erp_figures(evoked_a, evoked_b, evoked_diff, figures_dir)
    return {key: str(path) for key, path in paths.items()}


def simulate_model_eeg_erps(
    model_prefix: Path = DEFAULT_MODEL_PREFIX,
    output_dir: Path = OUTPUT_DIR,
    figures_dir: Path = FIGURES_DIR,
    eeg_scale_uv: float = 2.0,
) -> dict[str, str]:
    g, cat, ds = load_model_inputs(model_prefix)
    if g.shape[0] != len(MODEL_UNITS):
        raise ValueError(f"Expected {len(MODEL_UNITS)} model units, got g.shape={g.shape}")
    if cat.shape != g.shape[1:3]:
        raise ValueError(f"Expected cat shape {g.shape[1:3]}, got {cat.shape}")

    info, ch_names, sensor_pos_m = biosemi_info_and_positions()
    sources = build_source_specs()
    source_ts = build_source_timeseries(g, sources)
    leadfield = build_leadfield(sensor_pos_m, sources)

    eeg = np.einsum("cs,ntsl->ntcl", leadfield, source_ts, optimize=True)
    eeg -= eeg.mean(axis=2, keepdims=True)
    eeg_v = (eeg * eeg_scale_uv * 1e-6).astype(np.float32)

    epochs = make_epochs(eeg_v, cat, ds, info)
    return save_outputs(model_prefix, epochs, source_ts, leadfield, sources, ch_names, output_dir, figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", type=Path, default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--eeg-scale-uv", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = simulate_model_eeg_erps(
        model_prefix=args.model_prefix,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        eeg_scale_uv=args.eeg_scale_uv,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
