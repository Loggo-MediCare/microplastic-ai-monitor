# AI Microplastic Monitor

> **Research prototype on synthetic data.** The images, spectra and stations are simulated. The
> spectral signatures are illustrative and not measured polymer spectra. The results show how
> this pipeline behaves; they are not field-validated accuracy figures.

A working version of the AI monitoring system described in Chapter 16, "Intelligent Environmental
Surveillance: Integrating AI Systems for Comprehensive Microplastic Monitoring and Analysis". It
implements all four components of the chapter's Fig. 16.1:

| Fig. 16.1 component | Module | Implementation |
|---|---|---|
| Computer vision (16.3.2) | `vision.py` | Coarse-median background removal (handles haze), MAD-adaptive threshold (handles exposure), connected components, and 13 features: size, length/width, aspect, circularity, solidity, intensity, texture, 4 normalised spectral bands |
| Machine learning (16.3.1) | `ml.py` | Random forests for material (PE/PP/PS/PET/organic/mineral/none) and for shape (bead/fragment/fibre/film), self-training with 5 % labels, and a count calibration for merged particles |
| Real-time detection (16.3.3) | `network.py` | Per-sample edge analysis (~0.5 s per field on one CPU), action-level alarms, store to SQLite |
| Data processing & analysis (16.3.4, 16.4.4) | `network.py`, `report.py` | Monthly reference-material QC (80–120 % recovery), flagging after a failed QC, Mann–Kendall trends, hotspots, naive forecast |

`render.py` produces 2 mm × 2 mm, 4-band filter images with pixel-exact ground truth. The
particles are microplastics (PE, PP, PS, PET as beads, fragments, fibres and films) plus natural
look-alikes (organic matter, including plant fibres, and mineral grains). Three water conditions
are simulated: clean, turbid and weathered.

## What the experiments show (`docs/results.md`)

Checks against the chapter's claims:

| Chapter claim | Result in this prototype |
|---|---|
| "up to 95 % accuracy" | **Holds for classification** of segmented particles: 97–99 % material accuracy and 98–99 % polymer ID. It does **not** hold for counting particles on a dense filter: only ~80 % get their own segment, because touching or overlapping particles merge. On sparse filters the figure is 91 %. |
| False positives < 3 % | 0.8–3.4 % depending on water type (natural look-alikes called plastic) |
| "slightly lower" on weathered particles | Much worse than "slightly" when training data lacks weathered particles: particles found fall from 80 % to **15 %**. Adding weathered examples to training restores 80 %. |
| > 90 % detection in turbid water | Classification yes (99.7 % precision). Detection is limited by segmentation (83 %), and particles under 50 µm suffer most in turbid water. |
| Monthly reference-material QC | Necessary but slow. A spherical-bead standard only failed 10 weeks into a camera drift. In the network run, the drifting station reported a **false "down" trend** for 6–10 weeks before QC flagged it. Adding fibres to the reference catches the drift 2 weeks earlier. Real samples degrade earlier still, so a per-image sharpness check is recommended as well. |
| Multispectral / hyperspectral identification | Confirmed as essential here. A greyscale camera with morphology only drops to 59–67 % material accuracy and 43–52 % polymer ID. |
| Semi-supervised learning | Self-training with 5 % labelled segments reaches 95–97 % material accuracy, close to fully supervised. |

Other observations:

* Calibrating counts with a single proportional factor (true ≈ 1.21 × predicted) reduces the
  per-image count error from 15–22 % to 6–12 %. A free intercept over-corrected clean,
  low-concentration samples and was dropped.
* Trend tests have their own pitfalls. The beach station shows a significant "down" trend that
  is only half of a seasonal cycle. The lake shows a borderline "up" trend (p = 0.043) that is a
  false positive at α = 0.05.
* The three named systems in §16.4 (POSEIDON-ML, AutoPlastic Detector and SmartScan-MP) cite
  references [15]–[17]. Those references are papers on animal health, microplastics in sputum and
  FLIM imaging; none of them describes those systems, so their performance figures could not be
  traced. §16.6 is generic text about "companies" and "customers" with no microplastic results.

## In-line flow probe (transparent tube + camera, IoT node)

`flow.py` and `probe.py` add a second measurement mode: water is pumped continuously through a
transparent flow cell past a strobed camera, particles are tracked from frame to frame, and an
edge node publishes counts, a Poisson confidence interval and health data over MQTT. The node also
runs local rules: a wiper when the window fouls, a degasser warning for bubbles, and a no-flow
alarm. Results are in `docs/flow-results.md`; the Chinese README (`README.zh-TW.md`) also has the
hardware list. Key results:

* Use a flat-walled cell, not a round tube. In a 1 mm round tube only 49 % of the water is within
  the depth of field: 59 % of 20–50 µm particles are found, and bead recovery depends on size
  (93–226 %). In a 1 × 0.3 mm cell all sizes are found and recovery is 95–110 %.
* Tracking is required. Summing detections over frames overcounts 2.8–6.5×; tracking stays
  within ±5 %.
* The imaged volume saturates at field volume × frame rate: 0.74 mL/min at 20 fps. A 20 µs
  strobe is essential; with continuous light only 23 % of particles are found at 0.5 mL/min.
* Without a bubble class, bubbles double the plastic count. Particles stuck to a fouled window
  inflate it by 39 % unless stationary tracks are rejected.
* Counting statistics are the real limit: at 10 particles/L a ±20 % result takes 14 days
  without pre-concentration and 20 minutes with 1000× pre-concentration.

## Setup

```bash
cd D:\robot\microplastic-ai-monitor
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest                  # 18 tests, ~1.5 min
.venv\Scripts\python -m mpmon demo              # docs/demo.png: annotated detection on one field
.venv\Scripts\python -m mpmon report            # docs/results.md, docs/network.png, docs/network.sqlite3 (~3 min)
.venv\Scripts\python -m mpmon flow              # docs/flow-results.md, flow.png, flow-demo.png (~8 min)
```

## Next steps toward real samples

1. Replace `render.py` with real images, for example Nile-red fluorescence or FTIR/Raman maps
   of filters. Keep `vision.match_ground_truth` for annotated images.
2. Add watershed or instance segmentation to split touching particles. This is the largest
   remaining source of error.
3. Add a per-image focus/sharpness self-check to catch drift between monthly QC runs.
4. Replace the action level of 50 particles/L with a site-specific value. Microplastics in water
   have no regulatory limit.
