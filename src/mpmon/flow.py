"""In-line flow-through probe: water is pumped through a transparent tube past a strobed camera.

Instead of filtering a sample and imaging the filter (chapter 16.3.2), the camera images particles
while they flow. That introduces problems the filter method does not have, and this module
simulates each of them:

* **Laminar (Poiseuille) flow**: particles near the axis move twice as fast as the mean, particles
  near the wall barely move. Fibres align with the flow.
* **Depth of field**: a microscope objective is only sharp in a thin slab. In a round 1 mm tube
  most of the water lies outside it, so small particles blur away.
* **Motion blur**: without a short strobe pulse a moving particle becomes a streak.
* **Frame gaps and double counting**: a particle is seen in several frames when slow and in none
  when it crosses the field between two frames. Tracking links detections so each particle counts
  once. The imaged volume saturates at field volume x frame rate.
* **Air bubbles**: round, bright, the classic false positive.
* **Window fouling**: biofilm dims the image and particles stick to the wall. Stuck particles do not
  move, so the tracker can reject them, and a wiper restores the window.
* **Sampled volume**: concentration = particles / imaged volume. A spike of reference beads
  measures the recovery and corrects it.

All optical numbers (depth of field, blur growth, strobe) are assumptions for a ~4 um/px
telecentric objective, not a characterised instrument.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi

from .ml import Dataset, ImageResult, Model
from .render import PLASTICS, UM_PER_PX, Condition, Particle, render, sample_particles
from .vision import FEATURES, features, match_ground_truth, segment

PX_PER_MM = 1000 / UM_PER_PX
MARGIN_PX = 60  # render particles this far outside the frame so they enter smoothly
MIN_AREA_PX = 16  # ~18 um equivalent diameter: below the reporting limit


@dataclass(frozen=True)
class Geometry:
    name: str
    width_mm: float  # across the flow, in the image plane
    depth_mm: float  # along the optical axis
    round: bool

    @property
    def area_mm2(self) -> float:
        return math.pi / 4 * self.width_mm ** 2 if self.round else self.width_mm * self.depth_mm

    def sample_cross_section(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform positions in the cross-section: y (across, mm), z (depth, mm), v / v_mean."""
        if self.round:
            R = self.width_mm / 2
            rho, phi = R * np.sqrt(rng.random(n)), rng.uniform(0, 2 * np.pi, n)
            y, z = rho * np.cos(phi), rho * np.sin(phi)
            return y, z, 2 * (1 - (rho / R) ** 2)
        y = rng.uniform(-self.width_mm / 2, self.width_mm / 2, n)
        z = rng.uniform(-self.depth_mm / 2, self.depth_mm / 2, n)
        return y, z, 1.5 * (1 - (2 * z / self.depth_mm) ** 2)  # wide-slot approximation

    @property
    def v_max_factor(self) -> float:
        return 2.0 if self.round else 1.5


ROUND_TUBE = Geometry("round tube 1 mm ID", 1.0, 1.0, True)
FLAT_CELL = Geometry("flat-walled cell 1 x 0.3 mm", 1.0, 0.3, False)


@dataclass(frozen=True)
class Optics:
    fps: float = 20.0
    exposure_s: float = 20e-6  # LED strobe pulse; continuous light = the camera exposure (e.g. 5 ms)
    dof_mm: float = 0.3
    blur_px_per_mm: float = 25.0  # defocus sigma growth outside the depth of field (assumption)
    frame_h: int = 256  # across the flow: 1.02 mm
    frame_w: int = 512  # along the flow: 2.05 mm

    @property
    def field_mm(self) -> float:
        return self.frame_w / PX_PER_MM


@dataclass(frozen=True)
class Water:
    plastic_per_ml: float
    natural_per_ml: float
    bubbles_per_ml: float = 0.0
    size_range_um: tuple[float, float] = (20, 300)
    turbidity: float = 0.0
    weathering: float = 0.3


@dataclass(frozen=True)
class Window:
    """State of the optical window: transmission and particles stuck to the wall."""
    transmission: float = 1.0
    stuck: tuple[Particle, ...] = ()


@dataclass(frozen=True)
class Probe:
    geometry: Geometry = FLAT_CELL
    optics: Optics = Optics()
    flow_ml_min: float = 0.25

    @property
    def v_mean_mm_s(self) -> float:
        return self.flow_ml_min * 1000 / 60 / self.geometry.area_mm2

    @property
    def max_imaging_ml_min(self) -> float:
        """Flow at which the field is refreshed every frame: area x field length x fps."""
        return self.geometry.area_mm2 * self.optics.field_mm * self.optics.fps * 60 / 1000

    def imaged_fraction(self, rng: np.random.Generator | None = None, n: int = 200_000) -> dict:
        """Theory: share of the particle flux that appears in >= 1 frame, and that is also in focus."""
        rng = rng or np.random.default_rng(0)
        _, z, vf = self.geometry.sample_cross_section(rng, n)
        w = vf / vf.sum()  # flux weighting
        step = vf * self.v_mean_mm_s / self.optics.fps
        seen = np.minimum(1.0, self.optics.field_mm / np.maximum(step, 1e-12))
        focus = np.abs(z) <= self.optics.dof_mm / 2
        return {"seen": float(w @ seen), "seen_in_focus": float(w @ (seen * focus))}


@dataclass
class Flowing:
    particle: Particle
    x0_mm: float
    v_mm_s: float
    y_mm: float
    z_mm: float


@dataclass
class Detection:
    frame: int
    x: float
    y: float
    area: float
    row: np.ndarray
    true_id: int  # pid + 1 of the particle it mostly covers, 0 = none
    material: str = ""


@dataclass
class WindowRun:
    probe: Probe
    particles: list[Particle]
    flowing: list[Flowing]
    detections: list[Detection]
    n_frames: int
    crossing_ids: set[int]  # pid+1 of particles crossing the field midline during the window
    backgrounds: list[float] = field(default_factory=list)
    images: dict[int, np.ndarray] = field(default_factory=dict)  # frames kept for figures

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.probe.optics.fps

    @property
    def volume_ml(self) -> float:
        return self.probe.flow_ml_min * self.duration_s / 60

    def by_id(self) -> dict[int, Particle]:
        return {p.pid + 1: p for p in self.particles}


def make_flow(probe: Probe, water: Water, n_frames: int, rng: np.random.Generator,
              spike: tuple[str, str, float, float] | None = None, pid0: int = 0) -> list[Flowing]:
    """Particles uniformly distributed in the slug of water that passes the field during the window.

    ``spike``: (material, shape, size_um, per_ml) reference particles added to the water.
    """
    g, o = probe.geometry, probe.optics
    T = n_frames / o.fps
    v_max = g.v_max_factor * probe.v_mean_mm_s
    margin = MARGIN_PX / PX_PER_MM
    x_lo, x_hi = -v_max * T - margin, o.field_mm + margin
    vol_ml = (x_hi - x_lo) * g.area_mm2 / 1000
    n_pl = int(rng.poisson(water.plastic_per_ml * vol_ml))
    n_nat = int(rng.poisson(water.natural_per_ml * vol_ml))
    n_bub = int(rng.poisson(water.bubbles_per_ml * vol_ml))
    parts = sample_particles(rng, n_pl, n_nat, water.size_range_um)
    parts += [Particle(0, "bubble", "bubble", float(np.exp(rng.uniform(np.log(30), np.log(250)))), 0, 0, 0)
              for _ in range(n_bub)]
    if spike:
        mat, shp, size, per_ml = spike
        parts += [Particle(0, mat, shp, size, 0, 0, 0) for _ in range(int(rng.poisson(per_ml * vol_ml)))]
    y, z, vf = g.sample_cross_section(rng, len(parts))
    out = []
    for i, p in enumerate(parts):
        angle = rng.normal(0, 0.15) if p.shape == "fiber" else p.angle  # shear aligns fibres
        p = replace(p, pid=pid0 + i, angle=angle, shape_seed=int(rng.integers(2 ** 31)))
        out.append(Flowing(p, float(rng.uniform(x_lo, x_hi)), float(vf[i] * probe.v_mean_mm_s),
                           float(y[i]), float(z[i])))
    return out


def stuck_particles(rng: np.random.Generator, n: int, probe: Probe, pid0: int = 100_000) -> list[Particle]:
    """Biofilm flocs, plant fibres and microplastics (40 %) stuck on the wall facing the camera."""
    g, o = probe.geometry, probe.optics
    z_wall = g.width_mm / 2 if g.round else g.depth_mm / 2
    defocus = o.blur_px_per_mm * max(0.0, z_wall - o.dof_mm / 2)
    out = []
    for i in range(n):
        fibre = rng.random() < 0.3
        material = ("PET" if fibre else "PE") if rng.random() < 0.4 else "organic"
        out.append(Particle(pid0 + i, material, "fiber" if fibre else "fragment",
                            float(rng.uniform(25, 120)) * (2 if fibre else 1),
                            float(rng.uniform(0, o.frame_w)), float(rng.uniform(0.15, 0.85) * o.frame_h),
                            float(rng.uniform(0, np.pi)), defocus, 0.0, int(rng.integers(2 ** 31))))
    return out


def run_window(probe: Probe, water: Water, n_frames: int, rng: np.random.Generator, model: Model | None = None,
               window: Window = Window(), spike=None, keep: tuple[int, ...] = ()) -> WindowRun:
    """Render, segment and (optionally) classify every frame of one acquisition window."""
    o, g = probe.optics, probe.geometry
    flowing = make_flow(probe, water, n_frames, rng, spike)
    cond = Condition("flow", turbidity=water.turbidity, weathering=water.weathering,
                     exposure=window.transmission, blur_px=0.8)
    mid = o.field_mm / 2
    T = n_frames / o.fps
    crossing = {f.particle.pid + 1 for f in flowing if f.x0_mm < mid <= f.x0_mm + f.v_mm_s * T}
    dets, bgs, kept = [], [], {}
    lo, hi = -MARGIN_PX, o.frame_w + MARGIN_PX
    for k in range(n_frames):
        t = k / o.fps
        visible = list(window.stuck)
        for f in flowing:
            x_px = (f.x0_mm + f.v_mm_s * t) * PX_PER_MM
            if lo < x_px < hi:
                visible.append(replace(
                    f.particle, cx=x_px, cy=o.frame_h / 2 + f.y_mm * PX_PER_MM,
                    defocus_px=o.blur_px_per_mm * max(0.0, abs(f.z_mm) - o.dof_mm / 2),
                    motion_px=f.v_mm_s * o.exposure_s * PX_PER_MM))
        img, gt = render(visible, cond, rng, shape=(o.frame_h, o.frame_w))
        if k in keep:
            kept[k] = img
        lab, bg = segment(img)
        X, boxes = features(img, lab, bg)
        match = match_ground_truth(lab, gt)
        n = int(lab.max())
        bgs.append(float(np.median(bg)))
        if n:
            cms = ndi.center_of_mass(np.ones_like(lab), lab, np.arange(1, n + 1))
            areas = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
            for i in range(n):
                if boxes[i][1].start == 0 or boxes[i][1].stop == o.frame_w:
                    continue  # cut by the frame edge: size and centroid are wrong, wait for a full view
                dets.append(Detection(k, float(cms[i][1]), float(cms[i][0]), float(areas[i]), X[i], int(match[i])))
    if model is not None:
        classify(dets, model)
    parts = [f.particle for f in flowing] + list(window.stuck)
    return WindowRun(probe, parts, flowing, dets, n_frames, crossing, bgs, kept)


def classify(dets: list[Detection], model: Model) -> None:
    """One batched forest call per window (per-frame calls are ~50x slower)."""
    if dets:
        for d, m in zip(dets, model.predict(np.vstack([d.row for d in dets]))[0]):
            d.material = str(m)


def as_dataset(runs: list[WindowRun]) -> Dataset:
    """Every frame's segments as training examples for the existing random-forest Model."""
    ds = Dataset()
    for run in runs:
        if run.detections:
            X = np.vstack([d.row for d in run.detections])
            ds.images.append(ImageResult(run.particles, X, np.array([d.true_id for d in run.detections])))
    return ds


def train_flow_model(seed: int = 5, n_frames: int = 120, bubbles: bool = True) -> Model:
    """Classifier trained on flow frames: blurred, streaked, aligned fibres and (optionally) bubbles."""
    rng = np.random.default_rng(seed)
    runs = []
    for geom in (FLAT_CELL, ROUND_TUBE):
        for turb in (0.0, 0.5):
            probe = Probe(geom, flow_ml_min=0.25)
            water = Water(1500, 1500, 600 if bubbles else 0, turbidity=turb, weathering=0.3)
            runs.append(run_window(probe, water, n_frames, rng))
    return Model().fit(as_dataset(runs))


# ---------------------------------------------------------------- tracking

@dataclass
class Track:
    dets: list[Detection]
    vx: float | None = None  # px per frame

    @property
    def last(self) -> Detection:
        return self.dets[-1]

    @property
    def countable(self) -> bool:
        """Moving and at least once as large as the 20 um lower size limit (drops fringe fragments)."""
        return not self.stationary and max(d.area for d in self.dets) >= MIN_AREA_PX

    @property
    def stationary(self) -> bool:
        if len(self.dets) < 3:
            return False
        xs = np.array([d.x for d in self.dets])
        fr = np.array([d.frame for d in self.dets])
        return abs(np.median(np.diff(xs) / np.diff(fr))) < 1.0

    def material(self) -> str:
        vals, counts = np.unique([d.material for d in self.dets], return_counts=True)
        return str(vals[counts.argmax()])

    def true_id(self) -> int:
        ids = [d.true_id for d in self.dets if d.true_id]
        if not ids:
            return 0
        vals, counts = np.unique(ids, return_counts=True)
        return int(vals[counts.argmax()])


def track(dets: list[Detection], probe: Probe, max_gap: int = 1) -> list[Track]:
    """Link detections frame to frame. Particles move only downstream (+x) and keep their lane (y).

    A new track accepts any forward step up to the fastest streamline; once a track has a velocity
    it must follow its prediction. Greedy assignment by cost.
    """
    vmax = probe.geometry.v_max_factor * probe.v_mean_mm_s / probe.optics.fps * PX_PER_MM
    frames: dict[int, list[Detection]] = {}
    for d in dets:
        frames.setdefault(d.frame, []).append(d)
    tracks: list[Track] = []
    active: list[Track] = []
    for k in sorted(frames):
        cand = []
        for ti, t in enumerate(active):
            gap = k - t.last.frame
            for di, d in enumerate(frames[k]):
                dy = abs(d.y - t.last.y)
                ratio = d.area / max(t.last.area, 1)
                if dy > 6 or not 0.33 < ratio < 3:
                    continue
                dx = d.x - t.last.x
                if t.vx is None:
                    if not -3 <= dx <= vmax * gap + 5:
                        continue
                    cost = dy + 0.01 * abs(dx)
                else:
                    err = abs(dx - t.vx * gap)
                    if err > 6 + 0.25 * abs(t.vx) * gap:
                        continue
                    cost = dy + err
                cand.append((cost, ti, di))
        used_t, used_d = set(), set()
        for cost, ti, di in sorted(cand):
            if ti in used_t or di in used_d:
                continue
            t, d = active[ti], frames[k][di]
            vx = (d.x - t.last.x) / (k - t.last.frame)
            t.vx = vx if t.vx is None else 0.5 * (t.vx + vx)
            t.dets.append(d)
            used_t.add(ti)
            used_d.add(di)
        for di, d in enumerate(frames[k]):
            if di not in used_d:
                t = Track([d])
                tracks.append(t)
                active.append(t)
        active = [t for t in active if k - t.last.frame <= max_gap]
    return tracks


# ---------------------------------------------------------------- what the probe itself can measure

def summarise(dets: list[Detection], probe: Probe, n_frames: int, backgrounds: list[float],
              ref_background: float | None = None) -> dict:
    """Counts the edge device can report without ground truth: tracks per class, sizes, health."""
    tracks = track(dets, probe)
    moving = [t for t in tracks if t.countable]
    diam = FEATURES.index("eq_diam_um")
    counts: dict[str, int] = {}
    sizes = {"20-50": 0, "50-100": 0, "100-300": 0, ">300": 0}
    for t in moving:
        m = t.material()
        counts[m] = counts.get(m, 0) + 1
        if m in PLASTICS:
            d = float(np.median([x.row[diam] for x in t.dets]))
            key = "20-50" if d < 50 else "50-100" if d < 100 else "100-300" if d < 300 else ">300"
            sizes[key] += 1
    bg = float(np.median(backgrounds)) if backgrounds else float("nan")
    return {
        "plastic": sum(v for k, v in counts.items() if k in PLASTICS),
        "by_class": counts, "plastic_by_size_um": sizes,
        "stuck_objects": int(sum(t.stationary for t in tracks)),
        "bubbles": counts.get("bubble", 0),
        "background": bg,
        "transmission": bg / ref_background if ref_background else float("nan"),
        "duration_s": n_frames / probe.optics.fps,
    }


# ---------------------------------------------------------------- evaluation against ground truth

def count_window(run: WindowRun, stationary_filter: bool = True) -> dict:
    """Tracked counts, naive per-frame counts and the truth for one window."""
    tracks = track(run.detections, run.probe)
    by_id = run.by_id()
    moving = [t for t in tracks if (t.countable if stationary_filter
                                    else max(d.area for d in t.dets) >= MIN_AREA_PX)]
    plastic_tracks = [t for t in moving if t.material() in PLASTICS]
    true_plastic = {i for i in run.crossing_ids if by_id[i].is_plastic}
    found = {t.true_id() for t in plastic_tracks} & true_plastic
    per_poly_ok = [t.material() == by_id[t.true_id()].material for t in plastic_tracks
                   if t.true_id() in true_plastic]
    frame_ok = [d.material == by_id[d.true_id].material for d in run.detections
                if d.true_id and by_id[d.true_id].is_plastic and d.material in PLASTICS]
    bubbles_as_plastic = sum(1 for t in plastic_tracks if t.true_id() and by_id[t.true_id()].material == "bubble")
    return {
        "true_plastic": len(true_plastic),
        "tracked_plastic": len(plastic_tracks),
        "naive_frame_sum": sum(1 for d in run.detections if d.material in PLASTICS),
        "recall": len(found) / max(len(true_plastic), 1),
        "count_ratio": len(plastic_tracks) / max(len(true_plastic), 1),
        "polymer_acc_track": float(np.mean(per_poly_ok)) if per_poly_ok else float("nan"),
        "polymer_acc_frame": float(np.mean(frame_ok)) if frame_ok else float("nan"),
        "stationary_tracks": int(sum(t.stationary for t in tracks)),
        "bubble_tracks": sum(1 for t in moving if t.material() == "bubble"),
        "bubbles_as_plastic": bubbles_as_plastic,
        "tracks": len(moving),
        "true_ids_found": found,
    }


def recall_by_size(runs: list[WindowRun], counts: list[dict],
                   bins=((20, 50), (50, 100), (100, 300), (300, 10_000))) -> dict:
    hit = {b: [0, 0] for b in bins}
    for run, c in zip(runs, counts):
        by_id = run.by_id()
        for i in run.crossing_ids:
            p = by_id[i]
            if not p.is_plastic:
                continue
            b = next(b for b in bins if b[0] <= p.size_um < b[1])
            hit[b][1] += 1
            hit[b][0] += i in c["true_ids_found"]
    return {f"{lo}-{hi if hi < 10_000 else ''} um": (v[0] / v[1] if v[1] else float("nan")) for (lo, hi), v in hit.items()}


def poisson_ci(n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95 % interval for a Poisson count (Wilson-Hilferty style, good for n >= 0)."""
    lo = 0.0 if n == 0 else n * (1 - 1 / (9 * n) - z / (3 * math.sqrt(n))) ** 3
    m = n + 1
    hi = m * (1 - 1 / (9 * m) + z / (3 * math.sqrt(m))) ** 3
    return lo, hi


def counting_time_h(conc_per_l: float, eff_ml_min: float, rel_half_width: float = 0.2) -> float:
    """Hours to count enough particles for a +/- ``rel_half_width`` 95 % interval (N = (1.96/r)^2)."""
    n_needed = (1.96 / rel_half_width) ** 2
    return n_needed / (conc_per_l / 1000 * eff_ml_min * 60)
