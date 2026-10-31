"""Synthetic 4-band microscope images of filtered water samples, with pixel-exact ground truth.

Each image is a 2 mm x 2 mm filter field (512 px at 4 um/px). Particles are microplastics
(PE, PP, PS, PET as beads, fragments, fibres and films) mixed with natural look-alikes (organic
matter, mineral grains). The 4 bands stand in for a multispectral / stained-fluorescence camera.
The spectral signatures are illustrative assumptions, not measured polymer spectra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy import ndimage as ndi

UM_PER_PX = 4.0
SIZE = 512
BANDS = 4

PLASTICS = ("PE", "PP", "PS", "PET")
NATURAL = ("organic", "mineral")
ARTEFACTS = ("bubble",)  # air bubbles in the in-line flow probe (flow.py)
MATERIALS = PLASTICS + NATURAL + ARTEFACTS
SHAPES = ("bead", "fragment", "fiber", "film", "bubble")

SIGNATURE = {  # relative response per band (assumption); PE and PP deliberately similar
    "PE": (0.90, 0.80, 0.30, 0.20), "PP": (0.80, 0.90, 0.40, 0.20),
    "PS": (0.50, 0.70, 0.90, 0.30), "PET": (0.30, 0.50, 0.80, 0.90),
    "organic": (0.60, 0.40, 0.30, 0.50), "mineral": (0.25, 0.25, 0.25, 0.25),
    "bubble": (0.65, 0.65, 0.65, 0.65),  # scatters all bands equally; drawn as a bright ring
}
BRIGHTNESS = {"PE": 1.0, "PP": 1.0, "PS": 0.9, "PET": 0.9, "organic": 0.6, "mineral": 0.35, "bubble": 1.1}
TEXTURE_SD = {"organic": 0.30, "mineral": 0.15, "bubble": 0.02}  # plastics default 0.05

PLASTIC_SHAPE_P = {"fragment": 0.45, "fiber": 0.30, "bead": 0.10, "film": 0.15}
MATERIAL_GIVEN_SHAPE = {
    "fiber": {"PET": 0.6, "PP": 0.2, "PE": 0.2},
    "bead": {"PE": 0.7, "PS": 0.3},
    "film": {"PE": 0.8, "PP": 0.2},
    "fragment": {"PE": 0.35, "PP": 0.30, "PS": 0.25, "PET": 0.10},
}


@dataclass(frozen=True)
class Particle:
    pid: int
    material: str
    shape: str
    size_um: float  # max dimension (fibre length)
    cx: float
    cy: float
    angle: float
    defocus_px: float = 0.0  # Gaussian sigma from lying outside the depth of field
    motion_px: float = 0.0  # streak length along x (flow direction) during the exposure
    shape_seed: int | None = None  # fixed outline across video frames; None = draw from render's rng

    @property
    def is_plastic(self) -> bool:
        return self.material in PLASTICS


@dataclass(frozen=True)
class Condition:
    name: str = "clean"
    turbidity: float = 0.0  # 0..1: haze, background level, sub-visible specks
    weathering: float = 0.0  # 0..1: plastic spectra drift toward organic, dimmer
    exposure: float = 1.0
    blur_px: float = 0.8


CLEAN = Condition()
TURBID = Condition("turbid", turbidity=1.0)
WEATHERED = Condition("weathered", weathering=0.6)


def _pick(rng: np.random.Generator, probs: dict[str, float]) -> str:
    keys = list(probs)
    return keys[rng.choice(len(keys), p=np.array([probs[k] for k in keys]) / sum(probs.values()))]


def sample_particles(rng: np.random.Generator, n_plastic: int, n_natural: int,
                     size_range_um: tuple[float, float] = (20, 500)) -> list[Particle]:
    lo, hi = np.log(size_range_um[0]), np.log(size_range_um[1])
    out = []
    for i in range(n_plastic + n_natural):
        if i < n_plastic:
            shape = _pick(rng, PLASTIC_SHAPE_P)
            material = _pick(rng, MATERIAL_GIVEN_SHAPE[shape])
        else:
            material = "organic" if rng.random() < 0.6 else "mineral"
            shape = ("fiber" if rng.random() < 0.35 else "fragment") if material == "organic" else "fragment"
        size = float(np.exp(rng.uniform(lo, hi))) * (2.0 if shape == "fiber" else 1.0)
        out.append(Particle(i, material, shape, size, rng.uniform(0, SIZE), rng.uniform(0, SIZE),
                            rng.uniform(0, np.pi)))
    return out


def _shape_alpha(p: Particle, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Alpha mask of one particle and the image coordinates of its top-left corner."""
    if p.shape_seed is not None:
        rng = np.random.default_rng(p.shape_seed)
    r = p.size_um / UM_PER_PX / 2
    half = int(np.ceil(r * 1.1)) + 3 + int(np.ceil(3 * p.defocus_px + p.motion_px / 2))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    c, s = np.cos(p.angle), np.sin(p.angle)
    u, v = c * xx + s * yy, -s * xx + c * yy  # particle frame

    if p.shape == "bubble":
        rho = np.hypot(u, v)
        ring = max(1.5, 0.15 * r)
        alpha = np.clip(1 - np.abs(rho - r) / ring, 0, 1) + 0.15 * (rho < r)
    elif p.shape == "bead":
        b = r * rng.uniform(0.85, 1.0)
        alpha = ((u / r) ** 2 + (v / b) ** 2 <= 1).astype(float)
    elif p.shape == "fiber":
        width = rng.uniform(1.5, 3.0)
        n = 6
        steps = np.cumsum(rng.normal(0, 0.35, n - 1))
        seg = 2 * r / (n - 1)
        pts = [np.zeros(2)]
        for a in steps:
            pts.append(pts[-1] + seg * np.array([np.cos(a), np.sin(a)]))
        pts = np.array(pts) - np.mean(pts, axis=0)
        dist = np.full(u.shape, np.inf)
        for a, b2 in zip(pts[:-1], pts[1:]):
            d = b2 - a
            t = np.clip(((u - a[0]) * d[0] + (v - a[1]) * d[1]) / (d @ d), 0, 1)
            dist = np.minimum(dist, np.hypot(u - a[0] - t * d[0], v - a[1] - t * d[1]))
        alpha = np.clip(width / 2 + 0.5 - dist, 0, 1)
    else:  # fragment, film, mineral grain: random polygon
        k = rng.integers(4, 7) if p.material == "mineral" else rng.integers(6, 11)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rad = r * rng.uniform(0.6, 1.0, k)
        stretch = rng.uniform(1.0, 2.0)
        verts = np.c_[rad * np.cos(ang), rad * np.sin(ang) / stretch]
        inside = Path(verts).contains_points(np.c_[u.ravel(), v.ravel()]).reshape(u.shape)
        alpha = inside.astype(float) * (0.55 if p.shape == "film" else 1.0)
    return alpha, int(round(p.cy)) - half, int(round(p.cx)) - half


def _blur(alpha: np.ndarray, p: Particle) -> np.ndarray:
    if p.motion_px >= 1:
        alpha = ndi.uniform_filter1d(alpha, size=int(round(p.motion_px)) + 1, axis=1, mode="constant")
    if p.defocus_px > 0:
        alpha = ndi.gaussian_filter(alpha, p.defocus_px, mode="constant")
    return alpha


def render(particles: list[Particle], cond: Condition, rng: np.random.Generator,
           shape: tuple[int, int] = (SIZE, SIZE)) -> tuple[np.ndarray, np.ndarray]:
    """Return (image [BANDS, H, W] float32, ground-truth id map [H, W], 0 = none, pid+1)."""
    H, W = shape
    t = cond.turbidity
    img = np.full((BANDS, H, W), 0.05 + 0.10 * t, dtype=np.float32)
    if t > 0:
        haze = ndi.gaussian_filter(rng.normal(size=(H, W)), 20)
        img += (haze / (np.abs(haze).max() + 1e-9) * 0.08 * t).astype(np.float32)
        for _ in range(int(400 * t * H * W / SIZE ** 2)):  # sub-visible suspended specks
            y, x = rng.integers(0, H), rng.integers(0, W)
            img[:, y:y + 2, x:x + 2] += 0.25 * np.array(SIGNATURE["organic"], dtype=np.float32)[:, None, None]
    gt = np.zeros((H, W), dtype=np.int32)
    organic = np.array(SIGNATURE["organic"])
    for p in particles:
        alpha, y0, x0 = _shape_alpha(p, rng)
        sharp = alpha
        if p.defocus_px > 0 or p.motion_px >= 1:
            alpha = _blur(alpha, p)
        sig = np.array(SIGNATURE[p.material])
        bright = BRIGHTNESS[p.material]
        if p.is_plastic and cond.weathering:
            sig = (1 - cond.weathering) * sig + cond.weathering * organic
            bright *= 1 - 0.4 * cond.weathering
        tex = 1 + rng.normal(0, TEXTURE_SD.get(p.material, 0.05), alpha.shape)
        ys, xs = slice(max(y0, 0), min(y0 + alpha.shape[0], H)), slice(max(x0, 0), min(x0 + alpha.shape[1], W))
        if ys.stop <= ys.start or xs.stop <= xs.start:
            continue
        a = alpha[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
        tx = tex[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
        value = (bright * sig)[:, None, None] * tx[None]
        img[:, ys, xs] = img[:, ys, xs] * (1 - a) + a * value
        footprint = a > 0.5
        if sharp is not alpha:  # blurred: sharp outline or the visible halo
            footprint = (sharp[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0] > 0.5) | (a > 0.25)
        gt[ys, xs][footprint] = p.pid + 1
    for b in range(BANDS):
        img[b] = ndi.gaussian_filter(img[b], cond.blur_px)
    img = img * cond.exposure + rng.normal(0, 0.012 + 0.03 * t, img.shape).astype(np.float32)
    return img.astype(np.float32), gt
