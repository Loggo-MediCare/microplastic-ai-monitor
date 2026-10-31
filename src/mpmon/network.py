"""Monitoring network (chapter 16.3.3, 16.3.4, 16.4.4).

Stations sample weekly. Every sample is imaged and counted by the AI pipeline at the edge, and the
result is stored in SQLite with an alarm if it exceeds the action level. Every 4 weeks each station
runs a reference-material check: 50 PE beads (25 x 30 um, 25 x 150 um) must be recovered at
80-120 %, the chapter's "monthly verification with standard reference material". Station results
after a failed QC are flagged. Network analytics: Mann-Kendall trend (Kendall tau vs time),
hotspots and a naive linear forecast.

Microplastics in water have no regulatory limit. ``ACTION_LEVEL_PER_L`` is a site-defined
assumption.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

from .ml import Model, analyse_image
from .render import SIZE, Condition, Particle, sample_particles

VOLUME_L = 10.0  # water filtered per sample
FIELD_FRACTION = 0.1  # share of the filter imaged
COUNT_PER_CONC = VOLUME_L * FIELD_FRACTION  # expected particles in the image per (particle/L)
ACTION_LEVEL_PER_L = 50.0
QC_EVERY_WEEKS = 4
QC_RECOVERY_OK = (0.8, 1.2)


@dataclass(frozen=True)
class Station:
    name: str
    base_per_l: float
    trend_per_week: float = 0.0  # fractional growth per week
    seasonal_amp: float = 0.0
    event: tuple[int, int, float] | None = None  # (start_week, end_week, multiplier)
    drift_from_week: int | None = None  # camera degrades: blur + exposure loss
    turbidity: float = 0.0
    weathering: float = 0.3

    def true_conc(self, week: int) -> float:
        c = self.base_per_l * (1 + self.trend_per_week * week) * (1 + self.seasonal_amp * math.sin(2 * math.pi * week / 26))
        if self.event and self.event[0] <= week <= self.event[1]:
            c *= self.event[2]
        return c

    def condition(self, week: int) -> Condition:
        """Camera drift = progressive defocus and lamp ageing after ``drift_from_week``."""
        k = max(0, week - self.drift_from_week + 1) if self.drift_from_week is not None else 0
        return Condition(self.name, turbidity=self.turbidity, weathering=self.weathering,
                         exposure=max(0.3, 1 - 0.06 * k), blur_px=0.8 + 0.45 * k)


DEFAULT_STATIONS = (
    Station("river_upstream", 12),
    Station("river_below_wwtp", 30, event=(10, 12, 2.5), turbidity=0.6),  # treatment-plant bypass
    Station("beach", 22, seasonal_amp=0.35, weathering=0.6),
    Station("harbour", 25, trend_per_week=0.04, turbidity=0.4),
    Station("lake", 15),
    Station("estuary_drifting_camera", 28, drift_from_week=14, turbidity=0.3),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (station TEXT, week INTEGER, true_per_l REAL, true_count INTEGER,
    est_count INTEGER, est_per_l REAL, alarm INTEGER, qc_ok INTEGER, PRIMARY KEY (station, week));
CREATE TABLE IF NOT EXISTS qc (station TEXT, week INTEGER, recovery REAL, ok INTEGER, PRIMARY KEY (station, week));
"""


def reference_material(rng: np.random.Generator, kind: str = "beads") -> list[Particle]:
    """50 reference particles on a grid so they do not touch.

    ``beads``: 25 x 30 um + 25 x 150 um PE beads (the usual spherical standard).
    ``beads+fibers``: 25 x 30 um PE beads + 25 x 120 um PET fibres, whose thin profile reacts to
    defocus the way real environmental fibres do.
    """
    out, step = [], SIZE / 9
    cells = [(i, j) for i in range(1, 9) for j in range(1, 9)]  # 64 cells, ~57 px apart
    for k, idx in enumerate(rng.choice(len(cells), 50, replace=False)):
        i, j = cells[idx]
        if k < 25:
            spec = ("PE", "bead", 30.0)
        else:
            spec = ("PE", "bead", 150.0) if kind == "beads" else ("PET", "fiber", 120.0)
        out.append(Particle(k, *spec, j * step, i * step, rng.uniform(0, np.pi)))
    return out


def qc_recovery(model: Model, cond: Condition, rng: np.random.Generator, kind: str = "beads") -> float:
    ref = analyse_image(reference_material(rng, kind),
                        Condition("reference", exposure=cond.exposure, blur_px=cond.blur_px), rng)
    return model.count_plastics(ref.X, raw=True) / 50  # separated particles: no merge correction


def run_network(model: Model, weeks: int = 26, stations=DEFAULT_STATIONS, seed: int = 11,
                db_path: str = ":memory:", qc_kind: str = "beads") -> sqlite3.Connection:
    db = sqlite3.connect(db_path)
    db.executescript("DROP TABLE IF EXISTS samples; DROP TABLE IF EXISTS qc;")
    db.executescript(SCHEMA)
    rng = np.random.default_rng(seed)
    for st in stations:
        qc_ok = True
        for week in range(weeks):
            cond = st.condition(week)
            if week % QC_EVERY_WEEKS == 0:
                rec = qc_recovery(model, cond, rng, qc_kind)
                qc_ok = QC_RECOVERY_OK[0] <= rec <= QC_RECOVERY_OK[1]
                db.execute("INSERT OR REPLACE INTO qc VALUES (?,?,?,?)", (st.name, week, rec, int(qc_ok)))
            true_c = st.true_conc(week)
            n_pl = int(rng.poisson(true_c * COUNT_PER_CONC))
            n_nat = int(rng.poisson(0.6 * true_c * COUNT_PER_CONC + 8))
            res = analyse_image(sample_particles(rng, n_pl, n_nat), cond, rng)
            est = model.count_plastics(res.X)
            est_c = est / COUNT_PER_CONC
            db.execute("INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?)",
                       (st.name, week, true_c, n_pl, est, est_c, int(est_c > ACTION_LEVEL_PER_L), int(qc_ok)))
    db.commit()
    return db


def analyse(db: sqlite3.Connection) -> dict:
    out = {}
    names = [r[0] for r in db.execute("SELECT DISTINCT station FROM samples ORDER BY rowid")]
    means = {}
    for name in names:
        rows = db.execute("SELECT week, true_per_l, est_per_l, alarm, qc_ok FROM samples WHERE station=? ORDER BY week",
                          (name,)).fetchall()
        w, true_c, est_c, alarm, qc = (np.array(c, dtype=float) for c in zip(*rows))
        tau, p = kendalltau(w, est_c)
        slope, icpt = np.polyfit(w, est_c, 1)
        true_exceed = true_c > ACTION_LEVEL_PER_L
        means[name] = est_c.mean()
        qc_rows = db.execute("SELECT week, recovery, ok FROM qc WHERE station=? ORDER BY week", (name,)).fetchall()
        first_fail = next((wk for wk, _, ok in qc_rows if not ok), None)
        out[name] = {
            "mean_true": float(true_c.mean()), "mean_est": float(est_c.mean()),
            "bias": float(np.median(est_c / np.maximum(true_c, 1e-9))),
            "trend": ("up" if tau > 0 else "down") if p < 0.05 else "none",
            "kendall_tau": float(tau), "p_value": float(p), "slope_per_week": float(slope),
            "forecast_4w": float(icpt + slope * (w[-1] + 4)),
            "alarms": int(alarm.sum()), "true_exceedances": int(true_exceed.sum()),
            "alarms_correct": int((alarm.astype(bool) & true_exceed).sum()),
            "qc_recoveries": [round(r, 2) for _, r, _ in qc_rows], "first_qc_fail_week": first_fail,
            "flagged_samples": int((qc == 0).sum()),
        }
    med = float(np.median(list(means.values())))
    for name in names:
        out[name]["hotspot"] = means[name] > 1.5 * med
    return out
