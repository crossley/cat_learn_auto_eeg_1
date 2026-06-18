#!/usr/bin/env python3
"""
Two rotating glass brains for talk slides.

Brain 1 (left):  visual-associative loop only
                 visual cortex → tail of caudate → GPi → VA thalamus → pre-SMA

Brain 2 (right): same loop + sensorimotor extension
                 ... pre-SMA → posterior putamen → GPi → VL thalamus → M1

Install:  pip install vedo imageio[ffmpeg]
Output:   figures/glass_brain_rotation.mp4
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import splprep, splev
import mne
import vedo

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_DIR / "figures" / "glass_brain_rotation.mp4"

# ── display ───────────────────────────────────────────────────────────
BRAIN_COLOR  = (210, 215, 235)
BRAIN_ALPHA  = 0.12
BG           = (5, 5, 15)
FPS          = 30
SECONDS      = 12
N_FRAMES     = FPS * SECONDS

# ── globe spin ────────────────────────────────────────────────────────
DEG_PER_SECOND = 24.0          # one full rotation every 15 s
DEG_PER_FRAME  = DEG_PER_SECOND / FPS

# ── tracts ────────────────────────────────────────────────────────────
TRACT_COLOR       = (255, 190, 60)
TRACT_RADIUS      = 1.5
TRACT_ALPHA       = 0.80
TRACT_ALPHA_START = 0.03

# ── bolus ─────────────────────────────────────────────────────────────
BOLUS_COLOR  = (255, 255, 255)
BOLUS_RADIUS = 6.0
BASE_DIM     = 0.30
FLASH_TAU    = 0.18
SWEEP_PERIOD = 5.5
TRACT_TRAVEL = 0.45

# ── layout ────────────────────────────────────────────────────────────
BRAIN_OFFSET = 210    # mm — x-offset of each brain from scene centre

# ── Circuit 1 ROIs (warm: red → yellow-green) ─────────────────────────
C1_ROIS = [
    ("Visual cortex",   [(22, -72,  -8)], (255,  70,  30), 6),
    ("Tail of caudate", [(28, -28,   0)], (255, 140,   0), 6),
    ("GPi",             [(20,  -5,  -4)], (255, 200,   0), 5),
    ("VA thalamus",     [( 9,  -4,  10)], (255, 230,  50), 5),
    ("Pre-SMA",         [( 6,   5,  52)], (200, 255,  50), 6),
]

# ── Circuit 2 extra ROIs (cool: teal → blue) ──────────────────────────
C2_ROIS = [
    ("Post. putamen",   [(28, -12,   5)], (  0, 210, 180), 6),
    ("VL thalamus",     [(14, -12,   8)], (  0, 170, 255), 5),
    ("M1",              [(38, -22,  58)], ( 80, 120, 255), 6),
]

# ── Tract waypoints (right hemisphere; left mirrored automatically) ────
C1_TRACTS = [
    [(22, -72,  -8), (25, -55,  -5), (28, -28,   0)],  # visual → tail caudate
    [(28, -28,   0), (24, -16,  -2), (20,  -5,  -4)],  # tail caudate → GPi
    [(20,  -5,  -4), (14,  -4,   3), ( 9,  -4,  10)],  # GPi → VA thalamus
    [( 9,  -4,  10), ( 7,   0,  32), ( 6,   5,  52)],  # VA thalamus → pre-SMA
]

C2_TRACTS = [
    [( 6,   5,  52), (18,  -2,  30), (28, -12,   5)],  # pre-SMA → post. putamen
    [(28, -12,   5), (24,  -8,  -1), (20,  -5,  -4)],  # post. putamen → GPi
    [(20,  -5,  -4), (17,  -8,   1), (14, -12,   8)],  # GPi → VL thalamus
    [(14, -12,   8), (22, -16,  30), (32, -20,  48), (38, -22, 58)],
]

T = TRACT_TRAVEL
B1_ARRIVALS = {i: [i * T] for i in range(len(C1_ROIS))}
B2_ARRIVALS = {
    0: [0 * T],
    1: [1 * T],
    2: [2 * T, 6 * T],
    3: [3 * T],
    4: [4 * T],
    5: [5 * T],
    6: [7 * T],
    7: [8 * T],
}


# ── helpers ───────────────────────────────────────────────────────────

def spline_pts(waypoints, n_pts=120):
    pts = np.array(waypoints, dtype=float)
    k = min(3, len(pts) - 1)
    tck, _ = splprep(pts.T, s=0, k=k)
    return np.column_stack(splev(np.linspace(0, 1, n_pts), tck))


def make_tract(waypoints):
    smooth = spline_pts(waypoints)
    tube = (
        vedo.Tube(smooth.tolist(), r=TRACT_RADIUS)
        .c(TRACT_COLOR).alpha(0).lighting("glossy")
    )
    return tube, smooth


def create_brain(surf_dir, rois, tracts, dx):
    """Build all vedo objects for one brain centred at x = dx.
    Returns (meshes, roi_groups, tract_objs, pts_r, pts_l,
             bolus_r, bolus_l, center)."""

    raw_coords = []
    meshes = []
    for hemi in ("lh", "rh"):
        coords, faces = mne.read_surface(str(surf_dir / f"{hemi}.pial"))
        coords = coords.copy()
        coords[:, 0] += dx
        raw_coords.append(coords)
        meshes.append(
            vedo.Mesh([coords, faces])
            .c(BRAIN_COLOR).alpha(BRAIN_ALPHA).lighting("glossy")
        )
    center = tuple(np.concatenate(raw_coords).mean(axis=0).tolist())

    roi_groups = []
    for _name, coord_list, color, radius in rois:
        group = []
        for (x, y, z) in coord_list:
            for sx in (x, -x):
                group.append(
                    vedo.Sphere(pos=(sx + dx, y, z), r=radius)
                    .c(color).alpha(0.95).lighting("glossy")
                )
        roi_groups.append((color, group))

    tract_objs, pts_r, pts_l = [], [], []
    for wpts in tracts:
        r_wpts = [(x + dx,  y, z) for x, y, z in wpts]
        l_wpts = [(-x + dx, y, z) for x, y, z in wpts]
        tube_r, pr = make_tract(r_wpts)
        tube_l, pl = make_tract(l_wpts)
        tract_objs.extend([tube_r, tube_l])
        pts_r.append(pr)
        pts_l.append(pl)

    bolus_r = (
        vedo.Sphere(pos=(dx, 0, 0), r=BOLUS_RADIUS)
        .c(BOLUS_COLOR).alpha(0).lighting("glossy")
    )
    bolus_l = (
        vedo.Sphere(pos=(dx, 0, 0), r=BOLUS_RADIUS)
        .c(BOLUS_COLOR).alpha(0).lighting("glossy")
    )

    return meshes, roi_groups, tract_objs, pts_r, pts_l, bolus_r, bolus_l, center


def roi_flash(sweep_t, arrivals):
    return max(
        (np.exp(-(sweep_t - arr) / FLASH_TAU) for arr in arrivals if sweep_t >= arr),
        default=0.0,
    )


def update_bolus(sweep_t, bolus_r, bolus_l, pts_r, pts_l, cum_angle, center):
    cx, cy, cz = center
    rad = np.deg2rad(cum_angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    for ti, (pr, pl) in enumerate(zip(pts_r, pts_l)):
        onset = ti * TRACT_TRAVEL
        if onset <= sweep_t < onset + TRACT_TRAVEL:
            frac = (sweep_t - onset) / TRACT_TRAVEL
            idx = int(np.clip(frac, 0.0, 1.0) * (len(pr) - 1))
            for sphere, pt in [(bolus_r, pr[idx]), (bolus_l, pl[idx])]:
                dx, dy = pt[0] - cx, pt[1] - cy
                rx = cos_a * dx - sin_a * dy + cx
                ry = sin_a * dx + cos_a * dy + cy
                sphere.pos(rx, ry, pt[2])
                sphere.alpha(0.95)
            return
    bolus_r.alpha(0.0)
    bolus_l.alpha(0.0)


# ── main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fsaverage_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    surf_dir = fsaverage_dir / "surf"

    b1_meshes, b1_rois, b1_tracts, b1_pr, b1_pl, b1_br, b1_bl, b1_ctr = create_brain(
        surf_dir, C1_ROIS, C1_TRACTS, dx=-BRAIN_OFFSET
    )
    b2_meshes, b2_rois, b2_tracts, b2_pr, b2_pl, b2_br, b2_bl, b2_ctr = create_brain(
        surf_dir, C1_ROIS + C2_ROIS, C1_TRACTS + C2_TRACTS, dx=+BRAIN_OFFSET
    )
    n_c1_tract_objs = len(C1_TRACTS) * 2

    b1_spinning = (
        b1_meshes
        + [s for _, g in b1_rois for s in g]
        + b1_tracts
    )
    b2_spinning = (
        b2_meshes
        + [s for _, g in b2_rois for s in g]
        + b2_tracts
    )
    b1_all = b1_spinning + [b1_br, b1_bl]
    b2_all = b2_spinning + [b2_br, b2_bl]

    all_objs = b1_all + b2_all
    plt = vedo.Plotter(offscreen=True, size=(1920, 1080), bg=BG)
    plt.show(*all_objs, interactive=False)

    # Fixed camera — no movement during render loop
    plt.camera.SetPosition(0, -560, 80)
    plt.camera.SetFocalPoint(0, -15, 20)
    plt.camera.SetViewUp(0, 0, 1)
    plt.camera.SetViewAngle(48)
    plt.render()

    cum_angle = 0.0
    video = vedo.Video(str(OUTPUT), fps=FPS)

    for frame_i in range(N_FRAMES):
        t = frame_i / FPS
        sweep_t = t % SWEEP_PERIOD
        learn_frac = frame_i / max(N_FRAMES - 1, 1)
        cum_angle = frame_i * DEG_PER_FRAME

        # globe spin — each brain rotates around its own centre
        for obj in b1_spinning:
            obj.rotate_z(DEG_PER_FRAME, around=b1_ctr)
        for obj in b2_spinning:
            obj.rotate_z(DEG_PER_FRAME, around=b2_ctr)

        # tract transparency
        alpha_now = TRACT_ALPHA_START + (TRACT_ALPHA - TRACT_ALPHA_START) * learn_frac
        for tract in b1_tracts:
            tract.alpha(alpha_now)
        for tract in b2_tracts[:n_c1_tract_objs]:
            tract.alpha(TRACT_ALPHA)
        for tract in b2_tracts[n_c1_tract_objs:]:
            tract.alpha(alpha_now)

        # bolus
        update_bolus(sweep_t, b1_br, b1_bl, b1_pr, b1_pl, cum_angle, b1_ctr)
        update_bolus(sweep_t, b2_br, b2_bl, b2_pr, b2_pl, cum_angle, b2_ctr)

        # ROI flash
        for roi_i, (base_color, grp) in enumerate(b1_rois):
            f = roi_flash(sweep_t, B1_ARRIVALS.get(roi_i, [roi_i * T]))
            b = BASE_DIM + (1.0 - BASE_DIM) * f
            lit = tuple(min(255, int(c * b)) for c in base_color)
            for s in grp:
                s.c(lit)

        for roi_i, (base_color, grp) in enumerate(b2_rois):
            f = roi_flash(sweep_t, B2_ARRIVALS.get(roi_i, [roi_i * T]))
            b = BASE_DIM + (1.0 - BASE_DIM) * f
            lit = tuple(min(255, int(c * b)) for c in base_color)
            for s in grp:
                s.c(lit)

        plt.render()
        video.add_frame()

    video.close()
    plt.close()
    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
