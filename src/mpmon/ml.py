"""Machine learning (chapter 16.3.1): material/shape classification and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

from .render import PLASTICS, Condition, Particle, render, sample_particles
from .vision import FEATURES, SPECTRAL, features, match_ground_truth, segment

NONE = "none"  # spurious segment (noise, speck, fragment of background)
SIZE_BINS_UM = ((0, 50), (50, 100), (100, 300), (300, 10_000))


@dataclass
class ImageResult:
    particles: list[Particle]
    X: np.ndarray
    match: np.ndarray  # gt pid+1 per segment, 0 = none

    def labels(self) -> tuple[np.ndarray, np.ndarray]:
        by_id = {p.pid + 1: p for p in self.particles}
        mat = np.array([by_id[m].material if m else NONE for m in self.match], dtype=object)
        shp = np.array([by_id[m].shape if m else NONE for m in self.match], dtype=object)
        return mat, shp


@dataclass
class Dataset:
    images: list[ImageResult] = field(default_factory=list)

    @property
    def X(self) -> np.ndarray:
        return np.vstack([im.X for im in self.images]) if self.images else np.zeros((0, len(FEATURES)))

    def y(self) -> tuple[np.ndarray, np.ndarray]:
        pairs = [im.labels() for im in self.images]
        return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def analyse_image(particles: list[Particle], cond: Condition, rng: np.random.Generator) -> ImageResult:
    img, gt = render(particles, cond, rng)
    lab, bg = segment(img)
    X, _ = features(img, lab, bg)
    return ImageResult(particles, X, match_ground_truth(lab, gt))


def make_dataset(n_images: int, cond: Condition, seed: int, n_plastic: tuple[int, int] = (15, 40),
                 n_natural: tuple[int, int] = (10, 30)) -> Dataset:
    rng = np.random.default_rng(seed)
    ds = Dataset()
    for _ in range(n_images):
        parts = sample_particles(rng, int(rng.integers(*n_plastic)), int(rng.integers(*n_natural)))
        ds.images.append(analyse_image(parts, cond, rng))
    return ds


class Model:
    """Random forest over segment features. ``spectral=False`` = greyscale camera (morphology only)."""

    def __init__(self, spectral: bool = True, seed: int = 0):
        self.cols = [i for i, f in enumerate(FEATURES) if spectral or f not in SPECTRAL]
        self.material = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
        self.shape = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)

    def fit(self, ds: Dataset) -> "Model":
        X = ds.X[:, self.cols]
        mat, shp = ds.y()
        self.material.fit(X, mat)
        plastic = np.isin(mat, PLASTICS)
        self.shape.fit(X[plastic], shp[plastic])
        return self

    def fit_semi_supervised(self, ds: Dataset, label_fraction: float, seed: int = 0) -> "Model":
        """Self-training (16.3.1 semi-supervised): only ``label_fraction`` of segments carry labels."""
        X = ds.X[:, self.cols]
        mat, _ = ds.y()
        classes = np.array(sorted(set(mat)))
        y = np.searchsorted(classes, mat)
        rng = np.random.default_rng(seed)
        y_partial = np.where(rng.random(len(y)) < label_fraction, y, -1)
        # n_jobs=1: inside SelfTrainingClassifier a parallel forest floods joblib warnings
        base = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
        st = SelfTrainingClassifier(base, threshold=0.8).fit(X, y_partial)
        self.material = _Decode(st, classes)
        _, shp = ds.y()
        labelled_plastic = (y_partial >= 0) & np.isin(mat, PLASTICS)  # shape model: labelled segments only
        self.shape.fit(X[labelled_plastic], shp[labelled_plastic])
        return self

    def calibrate(self, ds: Dataset) -> "Model":
        """Fit true plastic count = a * predicted count per image (least squares through the origin).

        This corrects the systematic undercount caused by touching or overlapping particles merging
        into one segment, which classification cannot fix. A free intercept was tried and
        over-corrected clean low-concentration samples.
        """
        pred = np.array([self.count_plastics(im.X, raw=True) for im in ds.images], dtype=float)
        true = np.array([sum(p.is_plastic for p in im.particles) for im in ds.images], dtype=float)
        self.count_fit = (float(pred @ true / max(pred @ pred, 1e-9)), 0.0)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.array([], dtype=object), np.array([], dtype=object)
        Xs = X[:, self.cols]
        mat = self.material.predict(Xs)
        shp = np.where(np.isin(mat, PLASTICS), self.shape.predict(Xs), NONE) if hasattr(self.shape, "classes_") \
            else np.full(len(mat), NONE, dtype=object)
        return mat, shp

    count_fit: tuple[float, float] | None = None

    def count_plastics(self, X: np.ndarray, raw: bool = False) -> float:
        n = int(np.isin(self.predict(X)[0], PLASTICS).sum())
        if raw or self.count_fit is None:
            return n
        a, b = self.count_fit
        return max(0.0, a * n + b)


class _Decode:
    def __init__(self, est, classes):
        self.est, self.classes = est, classes

    def predict(self, X):
        return self.classes[self.est.predict(X)]


def evaluate(model: Model, ds: Dataset) -> dict:
    """Detection, classification and counting metrics against ground truth."""
    mat_true, shp_true = ds.y()
    mat_pred, shp_pred = model.predict(ds.X)
    plastic_true = np.isin(mat_true, PLASTICS)
    plastic_pred = np.isin(mat_pred, PLASTICS)
    tp = int((plastic_true & plastic_pred).sum())
    fp = int((~plastic_true & plastic_pred).sum())
    fn = int((plastic_true & ~plastic_pred).sum())
    tn = int((~plastic_true & ~plastic_pred).sum())

    # Detection recall of real plastic particles by size: found (segmented AND called plastic) at least once.
    found_by_bin = {b: [0, 0] for b in SIZE_BINS_UM}
    count_err, offset, segmented = [], 0, 0
    for im in ds.images:
        n = len(im.match)
        called = plastic_pred[offset: offset + n]
        found_ids = set(im.match[called & (im.match > 0)].tolist())
        seg_ids = set(im.match[im.match > 0].tolist())
        offset += n
        plastics = [p for p in im.particles if p.is_plastic]
        segmented += sum((p.pid + 1) in seg_ids for p in plastics)
        for p in plastics:
            b = next(b for b in SIZE_BINS_UM if b[0] <= p.size_um < b[1])
            found_by_bin[b][1] += 1
            found_by_bin[b][0] += (p.pid + 1) in found_ids
        if plastics:
            count_err.append(abs(int(called.sum()) - len(plastics)) / len(plastics))

    both = plastic_true & plastic_pred
    n_plastic_particles = sum(v[1] for v in found_by_bin.values())
    return {
        "segments": len(mat_true),
        "accuracy_material": float((mat_true == mat_pred).mean()) if len(mat_true) else float("nan"),
        "plastic_precision": tp / max(tp + fp, 1),
        "plastic_recall_segments": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "polymer_accuracy": float((mat_true[both] == mat_pred[both]).mean()) if both.any() else float("nan"),
        "shape_accuracy": float((shp_true[both] == shp_pred[both]).mean()) if both.any() else float("nan"),
        "segmentation_recall": segmented / max(n_plastic_particles, 1),  # got its own segment at all
        "particle_recall": sum(v[0] for v in found_by_bin.values()) / max(n_plastic_particles, 1),
        "recall_by_size": {f"{lo}-{hi if hi < 10_000 else ''}um": v[0] / max(v[1], 1)
                           for (lo, hi), v in found_by_bin.items()},
        "median_count_error": float(np.median(count_err)) if count_err else float("nan"),
    }
