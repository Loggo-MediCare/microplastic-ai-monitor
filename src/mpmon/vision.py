"""Computer vision (chapter 16.3.2): background removal, segmentation, feature extraction."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

from .render import BANDS, UM_PER_PX

FEATURES = ("area_um2", "eq_diam_um", "length_um", "width_um", "aspect", "circularity", "solidity",
            "mean_int", "tex_cv", "r0", "r1", "r2", "r3")
SPECTRAL = ("r0", "r1", "r2", "r3")


def background(intensity: np.ndarray, step: int = 8, win: int = 9) -> np.ndarray:
    """Slowly varying background (haze, illumination): median on a coarse grid, then upsampled."""
    small = ndi.median_filter(intensity[::step, ::step], size=win)
    bg = ndi.zoom(small, step, order=1)
    return bg[: intensity.shape[0], : intensity.shape[1]]


def segment(img: np.ndarray, k: float = 5.0, min_area_px: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Label map of candidate particles and the background estimate.

    Threshold = k robust standard deviations (MAD) above the local background, so it adapts to
    exposure and haze. Components smaller than ``min_area_px`` (about 16 um diameter) are dropped.
    """
    inten = img.mean(axis=0)
    bg = background(inten)
    resid = inten - bg
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    mask = ndi.binary_fill_holes(resid > k * max(mad, 1e-6))
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_area_px
        lab = np.where(keep[lab], lab, 0)
        lab, _ = ndi.label(lab > 0, structure=np.ones((3, 3)))
    return lab, bg


def _solidity(mask: np.ndarray, area: float) -> float:
    edge = mask & ~ndi.binary_erosion(mask)
    pts = np.argwhere(edge).astype(float)
    if len(pts) < 3:
        return 1.0
    corners = np.concatenate([pts + d for d in ((0, 0), (0, 1), (1, 0), (1, 1))])
    try:
        return float(min(1.0, area / ConvexHull(corners).volume))
    except QhullError:
        return 1.0


def features(img: np.ndarray, lab: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    """Feature matrix [n_segments, len(FEATURES)] and each segment's bounding box."""
    rows, boxes = [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = float(m.sum())
        ys, xs = np.nonzero(m)
        cov = np.cov(np.vstack([ys, xs])) if area > 1 else np.zeros((2, 2))
        l1, l2 = sorted(np.linalg.eigvalsh(cov + np.eye(2) / 12), reverse=True)  # +1/12: pixel extent
        length, width = 4 * np.sqrt(l1), 4 * np.sqrt(l2)
        perim = float((m & ~ndi.binary_erosion(m)).sum())
        bands = np.array([(img[b][sl][m] - bg[sl][m]).mean() for b in range(BANDS)])
        bands = np.clip(bands, 1e-4, None)
        inten = img[:, sl[0], sl[1]].mean(axis=0)[m]
        rows.append([
            area * UM_PER_PX ** 2,
            2 * np.sqrt(area / np.pi) * UM_PER_PX,
            length * UM_PER_PX,
            width * UM_PER_PX,
            length / max(width, 1e-6),
            min(1.0, 4 * np.pi * area / max(perim, 1) ** 2),
            _solidity(m, area),
            float(bands.mean()),
            float(inten.std() / max(abs(inten.mean()), 1e-6)),
            *(bands / bands.sum()),
        ])
        boxes.append(sl)
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES)), boxes


def match_ground_truth(lab: np.ndarray, gt: np.ndarray, min_frac: float = 0.3) -> np.ndarray:
    """For each segment, the ground-truth particle id it mostly covers (0 = spurious segment)."""
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        ids = gt[sl][m]
        ids = ids[ids > 0]
        if len(ids) == 0:
            out.append(0)
            continue
        vals, counts = np.unique(ids, return_counts=True)
        j = counts.argmax()
        out.append(int(vals[j]) if counts[j] >= min_frac * m.sum() else 0)
    return np.array(out, dtype=int)
