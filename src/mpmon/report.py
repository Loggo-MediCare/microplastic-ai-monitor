"""Experiments and the markdown report (docs/results.md)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .ml import Dataset, Model, analyse_image, evaluate, make_dataset
from .network import ACTION_LEVEL_PER_L, QC_RECOVERY_OK, Station, analyse, qc_recovery, run_network
from .render import CLEAN, TURBID, WEATHERED, sample_particles

CONDITIONS = (CLEAN, TURBID, WEATHERED)


def training_set(n_per_condition: int, seed: int) -> Dataset:
    return Dataset([im for i, c in enumerate(CONDITIONS)
                    for im in make_dataset(n_per_condition, c, seed + i).images])


def run_experiments(n_train: int = 25, n_test: int = 15, seed: int = 1) -> dict:
    tests = {c.name: make_dataset(n_test, c, 1000 + i) for i, c in enumerate(CONDITIONS)}
    clean_only = Model().fit(make_dataset(3 * n_train, CLEAN, seed))
    mixed_ds = training_set(n_train, seed + 10)
    mixed = Model().fit(mixed_ds)
    mixed.calibrate(training_set(8, seed + 20))  # separate calibration images
    grey = Model(spectral=False).fit(mixed_ds)
    semi = Model().fit_semi_supervised(mixed_ds, label_fraction=0.05)
    few = Model().fit(Dataset(mixed_ds.images[:: 20]))  # ~5 % of images, fully labelled
    sparse = make_dataset(n_test, CLEAN, 2000, n_plastic=(4, 8), n_natural=(2, 6))

    t0 = time.perf_counter()
    ds = make_dataset(10, TURBID, 3000)
    mixed.predict(ds.X)
    per_image_s = (time.perf_counter() - t0) / 10

    def count_err(ds: Dataset, raw: bool) -> float:
        errs = [abs(mixed.count_plastics(im.X, raw=raw) - t) / t for im in ds.images
                if (t := sum(p.is_plastic for p in im.particles))]
        return float(np.median(errs))

    return {
        "calibration": {"fit (true = a*pred + b)": mixed.count_fit,
                        **{name: (count_err(d, True), count_err(d, False)) for name, d in tests.items()}},
        "models": {
            "trained on clean only": {k: evaluate(clean_only, v) for k, v in tests.items()},
            "trained on clean+turbid+weathered": {k: evaluate(mixed, v) for k, v in tests.items()},
            "greyscale (no spectral bands)": {k: evaluate(grey, v) for k, v in tests.items()},
            "self-training, 5 % labels": {k: evaluate(semi, v) for k, v in tests.items()},
            "supervised, 5 % of images": {k: evaluate(few, v) for k, v in tests.items()},
        },
        "density": {"dense (25-80 particles/field)": evaluate(mixed, tests["clean"]),
                    "sparse (6-14 particles/field)": evaluate(mixed, sparse)},
        "seconds_per_image": per_image_s,
        "model": mixed,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def build_report(out_dir: Path, seed: int = 1, weeks: int = 26) -> str:
    ex = run_experiments(seed=seed)
    db = run_network(ex["model"], weeks=weeks, db_path=str(out_dir / "network.sqlite3"))
    net = analyse(db)
    _plot_network(db, out_dir / "network.png")

    L = ["# Results", "",
         "> Synthetic data. Every number below measures this pipeline on simulated filter images whose physics "
         "and spectra are assumptions (see README). These are not field measurements.", ""]
    L += ["## 1. Detection and identification", "",
          "| Model | Test water | Material acc. | Plastic precision | False-positive rate | Polymer ID acc. | "
          "Shape acc. | Particles found | Median count error |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mname, per in ex["models"].items():
        for cname, e in per.items():
            L.append(f"| {mname} | {cname} | {_pct(e['accuracy_material'])} | {_pct(e['plastic_precision'])} | "
                     f"{_pct(e['false_positive_rate'])} | {_pct(e['polymer_accuracy'])} | {_pct(e['shape_accuracy'])} | "
                     f"{_pct(e['particle_recall'])} | {_pct(e['median_count_error'])} |")
    L += ["", "## 2. Where particles are lost: segmentation, not classification", "",
          "| Filter loading | Got own segment | Found as plastic | <50 um | 50-100 um | 100-300 um | >300 um |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, e in ex["density"].items():
        r = list(e["recall_by_size"].values())
        L.append(f"| {name} | {_pct(e['segmentation_recall'])} | {_pct(e['particle_recall'])} | "
                 + " | ".join(_pct(x) for x in r) + " |")
    a, b = ex["calibration"]["fit (true = a*pred + b)"]
    L += ["", f"Count calibration fitted on separate images: true = {a:.2f} x predicted {b:+.1f}. "
              "Median per-image count error, raw → calibrated: "
              + "; ".join(f"{k} {_pct(v[0])} → {_pct(v[1])}" for k, v in ex["calibration"].items()
                          if not k.startswith("fit")) + "."]
    L += ["", f"Throughput: {ex['seconds_per_image']:.2f} s per 2 mm field (render excluded: segmentation + "
              f"features + classification on one CPU).", ""]
    L += ["## 3. Monitoring network (26 weekly samples per station)", "",
          f"Action level {ACTION_LEVEL_PER_L:g} particles/L (site-defined; there is no regulatory limit).", "",
          "| Station | Mean true /L | Mean est. /L | Median est/true | Trend (Kendall p) | Hotspot | Alarms (correct / true exceedances) "
          "| QC recoveries | First QC fail |", "|---|---:|---:|---:|---|---|---|---|---|"]
    for name, s in net.items():
        L.append(f"| {name} | {s['mean_true']:.1f} | {s['mean_est']:.1f} | {s['bias']:.2f} | "
                 f"{s['trend']} ({s['p_value']:.3f}) | {'yes' if s['hotspot'] else ''} | "
                 f"{s['alarms']} ({s['alarms_correct']}/{s['true_exceedances']}) | "
                 f"{', '.join(f'{r:.2f}' for r in s['qc_recoveries'])} | {s['first_qc_fail_week'] if s['first_qc_fail_week'] is not None else ''} |")
    L += ["", "![network](network.png)", ""]

    L += ["## 4. QC design: does the monthly reference check notice a drifting camera?", "",
          "Drift step k = weeks since defocus began (blur 0.8 + 0.45k px, exposure −6 %/week). The real-sample "
          "column is plastics found ÷ true plastics on a typical environmental sample.", "",
          "| k | blur px | bead reference recovery | bead+fibre reference recovery | real sample: found/true |",
          "|---:|---:|---:|---:|---:|"]
    rng = np.random.default_rng(seed + 50)
    model = ex["model"]
    drifting = Station("probe", 30, drift_from_week=0, turbidity=0.3)
    first = {"beads": None, "beads+fibers": None, "sample": None}
    for k in range(0, 13, 2):
        cond = drifting.condition(k)
        rb = qc_recovery(model, cond, rng, "beads")
        rf = qc_recovery(model, cond, rng, "beads+fibers")
        samples = [analyse_image(sample_particles(rng, 30, 18), cond, rng) for _ in range(3)]
        real = np.mean([model.count_plastics(s.X, raw=True) / 30 for s in samples])
        for key, v in (("beads", rb), ("beads+fibers", rf), ("sample", real)):
            if first[key] is None and not (QC_RECOVERY_OK[0] <= v <= QC_RECOVERY_OK[1]):
                first[key] = k
        L.append(f"| {k} | {cond.blur_px:.2f} | {_pct(rb)} | {_pct(rf)} | {_pct(real)} |")
    L += ["", f"First step outside 80-120 %: bead reference k={first['beads']}, bead+fibre reference "
              f"k={first['beads+fibers']}, real samples k={first['sample']}.", ""]
    return "\n".join(L)


def _plot_network(db, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0] for r in db.execute("SELECT DISTINCT station FROM samples ORDER BY rowid")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        w, t, e, qc = map(np.array, zip(*db.execute(
            "SELECT week, true_per_l, est_per_l, qc_ok FROM samples WHERE station=? ORDER BY week", (name,))))
        ax.plot(w, t, color="#888", lw=1.5, label="true")
        ax.plot(w, e, color="#1f6feb", lw=1.5, marker="o", ms=3, label="AI estimate")
        ax.scatter(w[qc == 0], e[qc == 0], color="#d1242f", zorder=3, s=18, label="after failed QC")
        ax.axhline(ACTION_LEVEL_PER_L, color="#d1242f", lw=0.8, ls="--")
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("week")
    for ax in axes[:, 0]:
        ax.set_ylabel("particles / L")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
