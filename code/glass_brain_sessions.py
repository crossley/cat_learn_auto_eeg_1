#!/usr/bin/env python3
"""
Four slowly rotating glass brains — one per training session.

ROI highlight strategy:
  roi_type = True        → pial surface patch (cortical clusters)
  roi_type = str         → Harvard-Oxford subcortical atlas mesh (e.g. "Right Thalamus")
  roi_type = False       → semi-transparent sphere fallback

Available HO subcortical labels:
  Left/Right Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Accumbens
  Brain-Stem, Left/Right Cerebral White Matter, Left/Right Cerebral Cortex

Sessions: 2, 4, 10, 20
Output:   figures/glass_brain_sessions.mp4
"""

from pathlib import Path
import numpy as np
import mne
import vedo

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_DIR / "figures" / "glass_brain_wald_whole_brain.mp4"

# ── display ───────────────────────────────────────────────────────────
BRAIN_COLOR  = (210, 215, 235)
BRAIN_ALPHA  = 0.12
BG           = (5, 5, 15)
FPS          = 30
SECONDS      = 12
N_FRAMES     = FPS * SECONDS

# ── spin ──────────────────────────────────────────────────────────────
DEG_PER_SECOND = 16.0
DEG_PER_FRAME  = DEG_PER_SECOND / FPS

# ── ROI peak-marker spheres (all ROIs) ────────────────────────────────
ROI_COLOR  = (255, 165, 30)
ROI_ALPHA  = 0.90
ROI_R_MIN  = 4.0
ROI_R_MAX  = 18.0

# ── cortical surface patches ──────────────────────────────────────────
PATCH_ALPHA  = 0.75
PATCH_R_MIN  = 12.0   # mm from peak, smallest cluster
PATCH_R_MAX  = 40.0   # mm from peak, largest cluster

# ── subcortical atlas meshes / sphere fallback ────────────────────────
VOL_COLOR       = (255, 120, 40)
VOL_ALPHA       = 0.40
ATLAS_THRESH    = 25          # % probability threshold for HO atlas

# ── layout ────────────────────────────────────────────────────────────
X_OFFSET = 220
Z_OFFSET = 90

# ── session data: (label, cluster_size, (x,y,z) MNI peak, roi_type) ──
#   roi_type: True = cortical  |  str = HO-sub label  |  False = sphere
SESSION_DATA = {
    2: [
        ("Precuneus",          19154, ( -8, -74,  40), True),
        ("R. frontal pole",     9958, ( 38,  64,  -2), True),
        ("L. insula",           5170, (-36,  20,  -2), True),
        ("Paracingulate",       2058, (  4,  28,  38), True),
        ("Post. cingulate",      560, (  2, -32,  20), True),
    ],
    4: [
        ("R. supramarginal",   27008, ( 46, -32,  46), True),
        ("R. frontal opercul", 13571, ( 42,  18,  -2), True),
    ],
    10: [
        ("R. cuneal cortex",   22839, ( 14, -74,  34), True),
        ("R. paracingulate",   20327, ( 12,  30,  28), True),
        ("L. M1",                863, (-24, -10,  62), True),
        ("R. mid. frontal",      783, ( 26,   4,  52), True),
    ],
    20: [
        ("R. occ. fusiform",   57455, ( 28, -74, -18), True),
    ],
}

GRID = [
    ( 2, -X_OFFSET,  Z_OFFSET),
    ( 4, +X_OFFSET,  Z_OFFSET),
    (10, -X_OFFSET, -Z_OFFSET),
    (20, +X_OFFSET, -Z_OFFSET),
]

LABEL_POS = [
    ("Session 2",   0.05, 0.82),
    ("Session 4",   0.55, 0.82),
    ("Session 10",  0.05, 0.14),
    ("Session 20",  0.55, 0.14),
]

_all_clusters = [row[1] for rows in SESSION_DATA.values() for row in rows]
_CBRT_LO = np.cbrt(min(_all_clusters))
_CBRT_HI = np.cbrt(max(_all_clusters))


def _scale(cluster, lo, hi):
    t = (np.cbrt(cluster) - _CBRT_LO) / (_CBRT_HI - _CBRT_LO)
    return lo + (hi - lo) * t


# ── Harvard-Oxford subcortical atlas ─────────────────────────────────
# labels[0] = "Background" has no volume; vol i → labels[i+1]
_atlas_img    = None
_atlas_labels = None
_atlas_data   = None
_struct_cache = {}


def _ensure_atlas():
    global _atlas_img, _atlas_labels, _atlas_data
    if _atlas_img is not None:
        return True
    try:
        from nilearn import datasets as _nl
        atlas = _nl.fetch_atlas_harvard_oxford('sub-prob-1mm', verbose=0)
        _atlas_img    = atlas.maps
        _atlas_labels = atlas.labels
        _atlas_data   = _atlas_img.get_fdata()
        return True
    except Exception as e:
        print(f"Warning: HO atlas unavailable ({e}). Subcortical ROIs → sphere fallback.")
        _atlas_img = False   # sentinel: tried and failed
        return False


def _get_structure_mesh(label_name):
    """Return (verts_mni, faces) for a named HO subcortical structure."""
    if label_name in _struct_cache:
        return _struct_cache[label_name]

    try:
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        import nibabel as nib
    except ImportError as e:
        print(f"Warning: missing dependency ({e}). Using sphere fallback.")
        _struct_cache[label_name] = None
        return None

    if not _ensure_atlas():
        _struct_cache[label_name] = None
        return None

    if label_name not in _atlas_labels:
        print(f"Warning: '{label_name}' not in HO subcortical atlas. Using sphere fallback.")
        _struct_cache[label_name] = None
        return None

    vi = _atlas_labels.index(label_name) - 1   # skip Background entry
    prob = _atlas_data[..., vi].astype(float)
    smoothed = gaussian_filter(prob, sigma=1.5)

    try:
        verts_vox, faces, _, _ = marching_cubes(smoothed, level=ATLAS_THRESH)
    except ValueError:
        print(f"Warning: marching cubes failed for '{label_name}'. Using sphere fallback.")
        _struct_cache[label_name] = None
        return None

    verts_mni = nib.affines.apply_affine(_atlas_img.affine, verts_vox)
    result = (verts_mni, faces)
    _struct_cache[label_name] = result
    return result


# ── surface patch extraction ──────────────────────────────────────────

def extract_surface_patch(coords, faces, peak, radius):
    """Triangulated pial patch whose face centroids fall within radius of peak."""
    centroids = coords[faces].mean(axis=1)
    face_mask = np.linalg.norm(centroids - peak, axis=1) < radius
    if not np.any(face_mask):
        return None
    sub_faces = faces[face_mask]
    old_idx = np.unique(sub_faces)
    remap = np.zeros(len(coords), dtype=int)
    remap[old_idx] = np.arange(len(old_idx))
    return vedo.Mesh([coords[old_idx], remap[sub_faces]])


# ── brain builder ─────────────────────────────────────────────────────

def create_brain(surf_dir, rois, dx, dz):
    raw_coords_list, raw_faces_list = [], []
    meshes = []
    offset = 0
    for hemi in ("lh", "rh"):
        coords, faces = mne.read_surface(str(surf_dir / f"{hemi}.pial"))
        coords = coords.copy()
        coords[:, 0] += dx
        coords[:, 2] += dz
        raw_coords_list.append(coords)
        raw_faces_list.append(faces + offset)
        offset += len(coords)
        meshes.append(
            vedo.Mesh([coords, faces])
            .c(BRAIN_COLOR).alpha(BRAIN_ALPHA).lighting("glossy")
        )

    all_coords = np.concatenate(raw_coords_list, axis=0)
    all_faces  = np.concatenate(raw_faces_list,  axis=0)
    center = tuple(all_coords.mean(axis=0).tolist())

    spheres, highlights = [], []
    for _name, cluster, (x, y, z), roi_type in rois:
        peak = np.array([x + dx, y, z + dz])
        r_hi = _scale(cluster, PATCH_R_MIN, PATCH_R_MAX)

        # peak marker for every ROI
        spheres.append(
            vedo.Sphere(pos=peak.tolist(), r=_scale(cluster, ROI_R_MIN, ROI_R_MAX))
            .c(ROI_COLOR).alpha(ROI_ALPHA).lighting("glossy")
        )

        if roi_type is True:
            # ── cortical: pial surface patch ──────────────────────────
            patch = extract_surface_patch(all_coords, all_faces, peak, r_hi)
            if patch is not None:
                highlights.append(
                    patch.c(ROI_COLOR).alpha(PATCH_ALPHA).lighting("glossy")
                )

        elif isinstance(roi_type, str):
            # ── subcortical: Harvard-Oxford atlas mesh ─────────────────
            struct = _get_structure_mesh(roi_type)
            if struct is not None:
                verts_mni, faces_arr = struct
                verts_off = verts_mni.copy()
                verts_off[:, 0] += dx
                verts_off[:, 2] += dz
                highlights.append(
                    vedo.Mesh([verts_off, faces_arr])
                    .c(VOL_COLOR).alpha(VOL_ALPHA).lighting("glossy")
                )
            else:
                highlights.append(
                    vedo.Sphere(pos=peak.tolist(), r=r_hi)
                    .c(VOL_COLOR).alpha(VOL_ALPHA).lighting("glossy")
                )

        else:
            # ── False: explicit sphere fallback ────────────────────────
            highlights.append(
                vedo.Sphere(pos=peak.tolist(), r=r_hi)
                .c(VOL_COLOR).alpha(VOL_ALPHA).lighting("glossy")
            )

    return meshes, spheres, highlights, center


# ── main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fsaverage_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    surf_dir = fsaverage_dir / "surf"

    brains = []
    all_objs = []
    for session, dx, dz in GRID:
        meshes, spheres, highlights, center = create_brain(
            surf_dir, SESSION_DATA[session], dx, dz
        )
        spinning = meshes + spheres + highlights
        brains.append({"spinning": spinning, "center": center})
        all_objs.extend(spinning)

    plt = vedo.Plotter(offscreen=True, size=(1920, 1080), bg=BG)
    plt.show(*all_objs, interactive=False)

    plt.camera.SetPosition(0, -700, 80)
    plt.camera.SetFocalPoint(0, -15, 0)
    plt.camera.SetViewUp(0, 0, 1)
    plt.camera.SetViewAngle(55)
    plt.render()

    for text, px, py in LABEL_POS:
        plt.add(vedo.Text2D(text, pos=(px, py), s=1.3, c="white"))

    video = vedo.Video(str(OUTPUT), fps=FPS)

    for frame_i in range(N_FRAMES):
        for b in brains:
            for obj in b["spinning"]:
                obj.rotate_z(DEG_PER_FRAME, around=b["center"])
        plt.render()
        video.add_frame()

    video.close()
    plt.close()
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
