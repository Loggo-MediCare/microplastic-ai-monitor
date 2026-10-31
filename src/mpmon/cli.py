"""Command line: ``python -m mpmon {report,demo,flow}``."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mpmon", description="AI microplastic monitoring prototype")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="run all experiments + network, write docs/results.md")
    r.add_argument("--out", type=Path, default=Path("docs"))
    r.add_argument("--seed", type=int, default=1)
    d = sub.add_parser("demo", help="render one sample, segment and save an annotated picture")
    d.add_argument("--out", type=Path, default=Path("docs/demo.png"))
    d.add_argument("--condition", choices=["clean", "turbid", "weathered"], default="turbid")
    f = sub.add_parser("flow", help="in-line transparent-tube probe + IoT node experiments, write docs/flow-results.md")
    f.add_argument("--out", type=Path, default=Path("docs"))
    f.add_argument("--quick", action="store_true", help="shorter windows (~2 min)")
    a = p.parse_args(argv)

    if a.cmd == "report":
        from .report import build_report
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "results.md").write_text(build_report(a.out, a.seed), encoding="utf-8")
        print(f"wrote {a.out / 'results.md'}, {a.out / 'network.png'}, {a.out / 'network.sqlite3'}")
    elif a.cmd == "flow":
        from .flow_report import write
        r = write(a.out, a.quick)
        print(f"wrote {a.out / 'flow-results.md'}, flow.png, flow-demo.png ({r['runtime_s']:.0f} s)")
    else:
        _demo(a.out, a.condition)
    return 0


def _demo(out: Path, cond_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    from . import render as R
    from .ml import make_dataset, Model
    from .report import training_set
    from .vision import features, segment

    cond = {"clean": R.CLEAN, "turbid": R.TURBID, "weathered": R.WEATHERED}[cond_name]
    model = Model().fit(training_set(10, 5))
    rng = np.random.default_rng(42)
    parts = R.sample_particles(rng, 25, 15)
    img, gt = R.render(parts, cond, rng)
    lab, bg = segment(img)
    X, boxes = features(img, lab, bg)
    mat, _ = model.predict(X)
    rgb = np.clip(img[[0, 1, 2]].transpose(1, 2, 0) / np.percentile(img, 99.5), 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"synthetic {cond_name} filter field (bands 0-2 as RGB)")
    axes[1].imshow(rgb)
    for (ys, xs), m in zip(boxes, mat):
        color = "#ff3b30" if m in R.PLASTICS else "#34c759"
        axes[1].add_patch(plt.Rectangle((xs.start, ys.start), xs.stop - xs.start, ys.stop - ys.start,
                                        fill=False, ec=color, lw=1))
        axes[1].text(xs.start, ys.start - 2, m, color=color, fontsize=6)
    n_true = sum(p.is_plastic for p in parts)
    axes[1].set_title(f"AI: {int(np.isin(mat, R.PLASTICS).sum())} plastic (true {n_true}); red = plastic, green = natural/none")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
