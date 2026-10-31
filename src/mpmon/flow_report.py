"""Experiments for the in-line flow probe; writes docs/flow-results.md, flow.png, flow-demo.png."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .flow import (FLAT_CELL, ROUND_TUBE, Optics, Probe, Water, Window, classify, count_window, counting_time_h,
                   recall_by_size, run_window, stuck_particles, track, train_flow_model)
from .probe import Broker, FlowSensorSim, NodeConfig, ProbeNode, Scenario, WiperSim, simulate_deployment
from .render import PLASTICS

IN_TUBE_PER_ML = 1500.0  # plastic, and as many natural particles: ~1 of each per frame in the flat cell
SPIKE_PER_ML = 2000.0  # reference beads: low enough that beads rarely touch (coincidence), ~80 per window


def _water_for(probe: Probe | None = None, n_frames: int = 0, **kw) -> Water:
    """The same water everywhere (already pre-concentrated upstream of the probe)."""
    return Water(IN_TUBE_PER_ML, IN_TUBE_PER_ML, **kw)


def _sweep_frames(q: float) -> int:
    """More frames at low flow so every point sees a similar number of particles."""
    return int(min(240, max(80, 30 / q)))


def _agg(counts: list[dict]) -> dict:
    true = sum(c["true_plastic"] for c in counts)
    found = sum(len(c["true_ids_found"]) for c in counts)
    tracked = sum(c["tracked_plastic"] for c in counts)
    return {"true": true, "tracked": tracked, "recall": found / max(true, 1), "count_ratio": tracked / max(true, 1),
            "naive_ratio": sum(c["naive_frame_sum"] for c in counts) / max(true, 1),
            "poly_track": float(np.nanmean([c["polymer_acc_track"] for c in counts])),
            "poly_frame": float(np.nanmean([c["polymer_acc_frame"] for c in counts]))}


def run_experiments(quick: bool = False, seed: int = 3) -> dict:
    t0 = time.time()
    s = 0.35 if quick else 1.0
    nf = lambda n: max(30, int(n * s))  # noqa: E731
    rng = np.random.default_rng(seed)
    model = train_flow_model(seed, n_frames=nf(120))
    res: dict = {"quick": quick}

    # 1. round tube vs flat-walled cell at the same flow
    geo = {}
    for g in (ROUND_TUBE, FLAT_CELL):
        probe = Probe(g, flow_ml_min=0.25)
        runs = [run_window(probe, _water_for(probe, nf(200)), nf(200), rng, model, keep=(nf(200) // 2,))
                for _ in range(2)]
        counts = [count_window(r) for r in runs]
        geo[g.name] = {**_agg(counts), "by_size": recall_by_size(runs, counts),
                       "theory": probe.imaged_fraction(), "v_mean": probe.v_mean_mm_s, "_runs": runs}
    res["geometry"] = geo

    # 2. flow-rate sweep in the flat cell (strobe), plus 60 fps and continuous light
    sweep = []
    cases = [(q, Optics()) for q in (0.1, 0.25, 0.5, 1.0, 2.0)]
    cases += [(1.0, Optics(fps=60)), (2.0, Optics(fps=60)), (0.25, Optics(exposure_s=5e-3)), (0.5, Optics(exposure_s=5e-3))]
    for q, opt in cases:
        probe = Probe(FLAT_CELL, opt, q)
        n = nf(_sweep_frames(q))
        c = count_window(run_window(probe, _water_for(probe, n), n, rng, model))
        a = _agg([c])
        sweep.append({"flow": q, "fps": opt.fps, "exposure_ms": opt.exposure_s * 1000, "v_mean": probe.v_mean_mm_s,
                      "theory_seen": probe.imaged_fraction()["seen"], "recall": a["recall"],
                      "count_ratio": a["count_ratio"], "naive_ratio": a["naive_ratio"],
                      "streak_px": probe.v_mean_mm_s * 1.5 * opt.exposure_s * 250,
                      "imaged_ml_h": q * 60 * a["recall"], "max_imaging_ml_min": probe.max_imaging_ml_min})
    res["sweep"] = sweep

    # 3. air bubbles: classifier with and without a bubble class
    probe = Probe(FLAT_CELL, flow_ml_min=0.25)
    w = _water_for()
    run = run_window(probe, replace(w, bubbles_per_ml=w.plastic_per_ml), nf(150), rng)
    no_bubble_model = train_flow_model(seed + 1, n_frames=nf(120), bubbles=False)
    bub = {}
    for name, m in (("with bubble class", model), ("without bubble class", no_bubble_model)):
        classify(run.detections, m)
        c = count_window(run)
        bub[name] = {"count_ratio": c["count_ratio"], "bubbles_as_plastic": c["bubbles_as_plastic"],
                     "bubble_tracks": c["bubble_tracks"], "true_plastic": c["true_plastic"]}
    res["bubbles"] = bub

    # 4. window fouling without a wiper
    foul = []
    probe = Probe(FLAT_CELL, flow_ml_min=0.25)
    clean_bg = float(np.median(run_window(probe, Water(0, 0), 5, rng).backgrounds))
    for h in (0, 24, 48, 72):
        T = float(np.exp(-h / 80))
        win = Window(T, tuple(stuck_particles(rng, int(0.6 * h), probe)))
        r = run_window(probe, _water_for(), nf(150), rng, model, win)
        c_on, c_off = count_window(r, True), count_window(r, False)
        foul.append({"hours": h, "transmission": T, "transmission_est": float(np.median(r.backgrounds)) / clean_bg,
                     "stuck": len(win.stuck), "recall": c_on["recall"], "count_ratio": c_on["count_ratio"],
                     "count_ratio_no_filter": c_off["count_ratio"], "stationary_tracks": c_on["stationary_tracks"]})
    res["fouling"] = foul

    # 5. spike calibration: recovery of reference beads, by bead size and geometry
    spikes = {}
    for g in (ROUND_TUBE, FLAT_CELL):
        probe = Probe(g, flow_ml_min=0.25)
        n, per_ml = nf(200), SPIKE_PER_ML
        for size in (30, 100, 250):
            r = run_window(probe, Water(0, 0), n, rng, model, spike=("PE", "bead", float(size), per_ml))
            c = count_window(r)
            expected = per_ml * r.volume_ml
            spikes[f"{g.name} | {size} um"] = {"expected": expected, "passed": c["true_plastic"],
                                              "counted": c["tracked_plastic"],
                                              "recovery": c["tracked_plastic"] / expected,  # what a lab can compute
                                              "efficiency": c["tracked_plastic"] / max(c["true_plastic"], 1)}
    res["spikes"] = spikes
    # apply the 100 um recovery to experiment 1's real samples
    conc = {}
    for g in (ROUND_TUBE, FLAT_CELL):
        rec = spikes[f"{g.name} | 100 um"]["recovery"]
        runs = geo[g.name]["_runs"]
        true_c = float(np.mean([_true_conc(r) for r in runs]))
        tracked = sum(count_window(r)["tracked_plastic"] for r in runs)
        vol = sum(r.volume_ml for r in runs)
        conc[g.name] = {"true_per_ml": true_c, "naive_per_ml": tracked / vol, "calibrated_per_ml": tracked / vol / rec,
                        "recovery_100um": rec}
    res["concentration"] = conc

    # 6. counting time at environmental concentrations (Poisson statistics)
    best = max((x for x in sweep if x["exposure_ms"] < 1 and x["fps"] == 20 and x["recall"] >= 0.8),
               key=lambda x: x["imaged_ml_h"])  # operating point: most water while still finding >= 80 %
    eff = best["imaged_ml_h"] / 60
    res["counting"] = {"eff_ml_min": eff, "flow": best["flow"], "rows": [
        {"per_l": c, "hours_direct": counting_time_h(c, eff), "hours_preconc_1000x": counting_time_h(c * 1000, eff)}
        for c in (1, 10, 100, 1_000, 10_000)]}

    # 7. IoT node: 12 windows with a pollution spike, a bubble burst, fouling and a pump fault
    broker = Broker()
    probe = Probe(FLAT_CELL, flow_ml_min=0.25)
    n = nf(60)
    base = _water_for()
    factor = 1000.0
    node = ProbeNode("river_below_wwtp", probe, model, broker, FlowSensorSim(0.25), WiperSim(lambda: None),
                     NodeConfig(action_level_per_l=base.plastic_per_ml * 1000 / factor * 2,
                                recovery=spikes[f"{FLAT_CELL.name} | 100 um"]["recovery"],
                                concentration_factor=factor))
    scen = Scenario(base, fouling_per_window=0.88, stuck_per_window=2,
                    events={3: "spike", 4: "spike", 6: "bubbles", 10: "pump_fault"})
    payloads = simulate_deployment(node, scen, 12, n, rng)
    res["iot"] = {"payloads": payloads, "wipes": node.wiper.count,
                  "topics": sorted({t for t, _ in broker.messages}),
                  "n_messages": len(broker.messages), "action_level": node.config.action_level_per_l,
                  "commands_tested": _command_roundtrip(node, broker)}
    res["runtime_s"] = time.time() - t0
    return res


def _true_conc(run) -> float:
    by = run.by_id()
    n = sum(1 for i in run.crossing_ids if by[i].is_plastic)
    return n / run.volume_ml


def _command_roundtrip(node: ProbeNode, broker: Broker) -> dict:
    before = (node.wiper.count, node.flow_sensor.setpoint_ml_min, node.config.recovery)
    broker.publish(node.topic("cmd"), {"cmd": "set_flow", "ml_min": 0.3})
    broker.publish(node.topic("cmd"), {"cmd": "calibrate", "recovery": 0.9})
    broker.publish(node.topic("cmd"), {"cmd": "wipe"})
    after = (node.wiper.count, node.flow_sensor.setpoint_ml_min, node.config.recovery)
    return {"before": before, "after": after}


# ---------------------------------------------------------------- output

def _p(x: float) -> str:
    return "–" if x != x else f"{100 * x:.0f} %"


def write(out: Path, quick: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    r = run_experiments(quick)
    _plot(r, out / "flow.png")
    _demo(r, out / "flow-demo.png")
    L = ["# In-line flow probe: results", "",
         f"> SIMULATION ONLY. Synthetic particles, assumed optics (depth of field 0.3 mm, 4 µm/px, 20 fps). "
         f"Generated by `python -m mpmon flow{' --quick' if quick else ''}` in {r['runtime_s']:.0f} s.", ""]

    L += ["## 1. Round tube vs flat-walled flow cell (0.25 mL/min, 20 µs strobe)", "",
          "| | round tube 1 mm | flat cell 1 × 0.3 mm |", "|---|---:|---:|"]
    g1, g2 = r["geometry"][ROUND_TUBE.name], r["geometry"][FLAT_CELL.name]
    rows = [("mean velocity (mm/s)", lambda g: f"{g['v_mean']:.1f}"),
            ("theory: flux in focus", lambda g: _p(g["theory"]["seen_in_focus"])),
            ("plastic particles through the field", lambda g: str(g["true"])),
            ("found (recall)", lambda g: _p(g["recall"])),
            ("counted / true (tracking)", lambda g: f"{g['count_ratio']:.2f}"),
            ("counted / true (sum over frames, no tracking)", lambda g: f"{g['naive_ratio']:.2f}"),
            ("polymer ID, single frame", lambda g: _p(g["poly_frame"])),
            ("polymer ID, majority vote along track", lambda g: _p(g["poly_track"]))]
    rows += [(f"recall {k}", (lambda k: lambda g: _p(g["by_size"][k]))(k)) for k in g1["by_size"]]
    for name, f in rows:
        L.append(f"| {name} | {f(g1)} | {f(g2)} |")

    L += ["", "## 2. Flow-rate sweep (flat cell)", "",
          "| flow mL/min | fps | light | mean v mm/s | streak px | theory seen | recall | counted/true | no tracking | imaged mL/h |",
          "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in r["sweep"]:
        light = "strobe 20 µs" if x["exposure_ms"] < 1 else f"continuous {x['exposure_ms']:.0f} ms"
        L.append(f"| {x['flow']} | {x['fps']:.0f} | {light} | {x['v_mean']:.0f} | {x['streak_px']:.1f} | "
                 f"{_p(x['theory_seen'])} | {_p(x['recall'])} | {x['count_ratio']:.2f} | {x['naive_ratio']:.1f} | "
                 f"{x['imaged_ml_h']:.1f} |")
    L += ["", f"Field refreshed every frame up to {r['sweep'][0]['max_imaging_ml_min']:.2f} mL/min at 20 fps "
          f"(area × field length × fps); above that, particles cross between frames."]

    L += ["", "## 3. Air bubbles (as many bubbles as plastic particles)", "",
          "| classifier | counted/true plastic | bubbles called plastic | bubble tracks |", "|---|---:|---:|---:|"]
    for k, v in r["bubbles"].items():
        L.append(f"| {k} | {v['count_ratio']:.2f} | {v['bubbles_as_plastic']} | {v['bubble_tracks']} |")

    L += ["", "## 4. Window fouling without a wiper", "",
          "| hours | transmission (true / estimated from background) | stuck objects | recall | counted/true | without stationary filter |",
          "|---:|---|---:|---:|---:|---:|"]
    for x in r["fouling"]:
        L.append(f"| {x['hours']} | {x['transmission']:.2f} / {x['transmission_est']:.2f} | {x['stuck']} | "
                 f"{_p(x['recall'])} | {x['count_ratio']:.2f} | {x['count_ratio_no_filter']:.2f} |")

    L += ["", "## 5. Spike calibration (PE reference beads)", "",
          "| geometry, bead size | expected from dose | actually passed | counted | recovery vs dose | vs beads passed |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, v in r["spikes"].items():
        L.append(f"| {k} | {v['expected']:.0f} | {v['passed']} | {v['counted']} | {_p(v['recovery'])} | "
                 f"{_p(v['efficiency'])} |")
    L += ["", "| geometry | true (/mL in tube) | naive estimate | corrected with 100 µm recovery |", "|---|---:|---:|---:|"]
    for k, v in r["concentration"].items():
        L.append(f"| {k} | {v['true_per_ml']:.0f} | {v['naive_per_ml']:.0f} | {v['calibrated_per_ml']:.0f} |")

    c = r["counting"]
    L += ["", f"## 6. Counting time for ±20 % (95 %) at {c['flow']} mL/min, imaged volume {c['eff_ml_min'] * 60:.0f} mL/h", "",
          "| particles / L in the river | direct | with 1000× pre-concentration |", "|---:|---:|---:|"]
    for x in c["rows"]:
        L.append(f"| {x['per_l']:,} | {_hours(x['hours_direct'])} | {_hours(x['hours_preconc_1000x'])} |")

    io = r["iot"]
    L += ["", "## 7. IoT node (12 windows)", "",
          f"Action level {io['action_level']:.0f} /L (alarm only if the lower 95 % bound exceeds it). "
          f"Topics: {', '.join(io['topics'])}. {io['n_messages']} messages, {io['wipes']} automatic wipes.", "",
          "| # | event | flow mL/min | plastic | per L (95 % CI) | true per L | transmission | stuck | bubbles | alarms |",
          "|---:|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for p in io["payloads"]:
        if p["status"] != "ok":
            L.append(f"| {p['seq']} | {p['_event']} | {p['flow_ml_min']} | – | – | – | – | – | – | {', '.join(p['alarms'])} |")
            continue
        h = p["health"]
        L.append(f"| {p['seq']} | {p['_event']} | {p['flow_ml_min']} | {p['plastic']} | {p['per_l']:.0f} "
                 f"({p['per_l_ci95'][0]:.0f}–{p['per_l_ci95'][1]:.0f}) | {p['_true_per_l']:.0f} | {h['transmission']:.2f} | "
                 f"{h['stuck_objects']} | {h['bubbles']} | {', '.join(p['alarms'])} |")
    sample = next(p for p in io["payloads"] if p["status"] == "ok")
    L += ["", "Example telemetry message:", "", "```json",
          json.dumps({k: v for k, v in sample.items() if not k.startswith("_")}, indent=1, ensure_ascii=False), "```", "",
          f"Downlink commands (set_flow, calibrate, wipe): before {io['commands_tested']['before']}, "
          f"after {io['commands_tested']['after']} (wipe count, flow setpoint, recovery)."]
    (out / "flow-results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    slim = {k: v for k, v in r.items() if k != "geometry"}
    slim["geometry"] = {k: {kk: vv for kk, vv in v.items() if kk != "_runs"} for k, v in r["geometry"].items()}
    (out / "flow-results.json").write_text(json.dumps(slim, indent=1, default=_json), encoding="utf-8")
    return r


def _json(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    if h < 24 * 365:
        return f"{h / 24:.0f} days"
    return f"{h / 24 / 365:.0f} years"


def _plot(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#1f6feb", "#d4760a", "#6e7781"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    sw = [x for x in r["sweep"] if x["exposure_ms"] < 1]
    for fps, col in ((20, blue), (60, orange)):
        xs = [x for x in sw if x["fps"] == fps]
        ax[0].plot([x["flow"] for x in xs], [x["imaged_ml_h"] for x in xs], "o-", color=col, lw=2, ms=6, label=f"{fps} fps")
    q = np.linspace(0.05, 2.1, 50)
    for fps, col in ((20, blue), (60, orange)):
        cap = Probe(FLAT_CELL, Optics(fps=fps)).max_imaging_ml_min * 60
        ax[0].plot(q, np.minimum(q * 60, cap), "--", color=col, lw=1)
    ax[0].set(xlabel="pumped flow (mL/min)", ylabel="water actually imaged (mL/h)",
              title="Imaged volume saturates at field volume × fps")
    ax[0].legend(frameon=False)

    g1, g2 = r["geometry"][ROUND_TUBE.name], r["geometry"][FLAT_CELL.name]
    keys = list(g1["by_size"])
    x = np.arange(len(keys))
    ax[1].bar(x - 0.2, [g1["by_size"][k] for k in keys], 0.38, color=grey, label="round tube 1 mm")
    ax[1].bar(x + 0.2, [g2["by_size"][k] for k in keys], 0.38, color=blue, label="flat cell 1 × 0.3 mm")
    ax[1].set(xticks=x, xticklabels=keys, ylim=(0, 1.3), yticks=np.linspace(0, 1, 6), ylabel="plastic particles found",
              title="Depth of field: small particles vanish in a round tube")
    ax[1].legend(frameon=False, loc="upper center", ncol=2)

    f = r["fouling"]
    hrs = [x["hours"] for x in f]
    ax[2].plot(hrs, [x["count_ratio"] for x in f], "o-", color=blue, lw=2, label="tracking + stationary filter")
    ax[2].plot(hrs, [x["count_ratio_no_filter"] for x in f], "o-", color=orange, lw=2, label="no stationary filter")
    ax[2].plot(hrs, [x["transmission_est"] for x in f], "--", color=grey, lw=1, label="window transmission (estimated)")
    ax[2].axhline(1, color="#999", lw=0.6)
    ax[2].set(xlabel="hours since last wipe", ylabel="counted / true", title="Fouling without a wiper")
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _demo(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, g in zip(axes, (ROUND_TUBE, FLAT_CELL)):
        run = r["geometry"][g.name]["_runs"][0]
        k, img = next(iter(run.images.items()))
        rgb = np.clip(img[[0, 1, 2]].transpose(1, 2, 0) / 0.8, 0, 1) ** 0.6  # fixed scale, gamma for dim particles
        ax.imshow(rgb)
        for t in track(run.detections, run.probe):
            near = [d for d in t.dets if abs(d.frame - k) <= 3]
            if not near or t.stationary:
                continue
            col = "#ff453a" if t.material() in PLASTICS else "#32d74b" if t.material() != "bubble" else "#64d2ff"
            ax.plot([d.x for d in near], [d.y for d in near], "-o", color=col, lw=1, ms=2.5)
            ax.text(near[0].x, near[0].y - 8, t.material(), color=col, fontsize=7)
        ax.set_title(f"{g.name}: frame {k} with tracks over ±3 frames (red = plastic, green = natural)", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
