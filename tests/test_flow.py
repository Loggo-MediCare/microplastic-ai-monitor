import numpy as np
import pytest

from mpmon import render as R
from mpmon.flow import (FLAT_CELL, ROUND_TUBE, Detection, Optics, Probe, Water, count_window, counting_time_h,
                        poisson_ci, run_window, track, train_flow_model)
from mpmon.probe import Broker, FlowSensorSim, NodeConfig, ProbeNode, Scenario, WiperSim, simulate_deployment


def test_poiseuille_mean_velocity_and_wall():
    for g in (ROUND_TUBE, FLAT_CELL):
        _, z, vf = g.sample_cross_section(np.random.default_rng(0), 200_000)
        assert vf.mean() == pytest.approx(1.0, abs=0.01)  # uniform positions: mean of profile = mean velocity
        assert vf.max() <= g.v_max_factor and vf.min() >= 0


def test_imaged_fraction_saturates_above_field_refresh_rate():
    p = Probe(FLAT_CELL, flow_ml_min=0.25)
    assert p.imaged_fraction()["seen"] == pytest.approx(1.0)
    fast = Probe(FLAT_CELL, flow_ml_min=4 * p.max_imaging_ml_min)
    assert fast.imaged_fraction()["seen"] < 0.5
    assert Probe(ROUND_TUBE).imaged_fraction()["seen_in_focus"] < 0.6  # most of a 1 mm tube is out of focus


def test_defocus_and_motion_blur():
    def one(**kw):
        p = R.Particle(0, "PE", "bead", 60, 128, 64, 0.0, **kw)
        return R.render([p], R.CLEAN, np.random.default_rng(0), shape=(128, 256))

    def width(gt):
        cols = np.nonzero((gt > 0).any(axis=0))[0]
        return cols.max() - cols.min() + 1

    sharp, gt0 = one()
    blurred, _ = one(defocus_px=6)
    streak, gt = one(motion_px=40)
    assert blurred.max() < 0.7 * sharp.max()
    assert width(gt) > 2 * width(gt0)  # 60 um = 15 px bead smeared along the flow
    assert streak.max() < 0.7 * sharp.max()  # same light spread over a longer streak


def test_bubble_renders_as_ring():
    p = R.Particle(0, "bubble", "bubble", 160, 128, 64, 0.0)
    img, _ = R.render([p], R.CLEAN, np.random.default_rng(0), shape=(128, 256))
    inten = img.mean(axis=0)
    assert inten[64, 128] < inten[64, 128 + 19]  # centre darker than the rim (radius 20 px)


def test_tracker_links_moving_particles_and_flags_stuck_ones():
    probe = Probe(FLAT_CELL, flow_ml_min=0.25)
    row = np.zeros(13)
    dets = []
    for k in range(6):
        dets.append(Detection(k, 20 + 80 * k, 50, 100, row, 1, "PE"))  # fast
        dets.append(Detection(k, 30 + 30 * k, 150, 60, row, 2, "PP"))  # slow, other lane
        dets.append(Detection(k, 300, 220, 80, row, 3, "PE"))  # stuck on the window
    tracks = track(dets, probe)
    assert len(tracks) == 3
    assert sum(t.countable for t in tracks) == 2
    assert sum(t.stationary for t in tracks) == 1


def test_poisson_interval_and_counting_time():
    lo, hi = poisson_ci(100)
    assert lo == pytest.approx(81.4, abs=0.5) and hi == pytest.approx(121.6, abs=0.5)
    assert poisson_ci(0)[0] == 0
    assert counting_time_h(10, 1.0) == pytest.approx(96.04 / 0.6, rel=1e-3)


@pytest.fixture(scope="module")
def flow_model():
    return train_flow_model(seed=2, n_frames=25)


def test_flat_cell_finds_and_counts_plastic_once(flow_model):
    probe = Probe(FLAT_CELL, flow_ml_min=0.25)
    run = run_window(probe, Water(1500, 1500), 60, np.random.default_rng(4), flow_model)
    c = count_window(run)
    assert c["true_plastic"] > 10
    assert c["recall"] > 0.8
    assert 0.7 < c["count_ratio"] < 1.3
    assert c["naive_frame_sum"] > 1.5 * c["true_plastic"]  # without tracking each particle counts several times


def test_iot_node_alarms_wiper_and_commands(flow_model):
    broker = Broker()
    probe = Probe(FLAT_CELL, Optics(), 0.25)
    node = ProbeNode("test", probe, flow_model, broker, FlowSensorSim(0.25), WiperSim(lambda: None),
                     NodeConfig(action_level_per_l=1e9, concentration_factor=1000))
    out = simulate_deployment(node, Scenario(Water(1500, 1500), fouling_per_window=0.6, stuck_per_window=1,
                                             events={2: "pump_fault"}), 4, 20, np.random.default_rng(1))
    assert out[2]["status"] == "no_flow" and "flow_low" in out[2]["alarms"]
    assert node.wiper.count >= 1  # transmission 0.6 -> local wipe rule
    assert any("window_fouled" in p["alarms"] for p in out)
    assert {t for t, _ in broker.messages} >= {"mpmon/test/probe/telemetry", "mpmon/test/probe/alarm"}
    assert node.db.execute("SELECT COUNT(*) FROM probe_windows").fetchone()[0] == 4
    broker.publish("mpmon/test/probe/cmd", {"cmd": "set_flow", "ml_min": 0.4})
    broker.publish("mpmon/test/probe/cmd", {"cmd": "calibrate", "recovery": 0.8})
    assert node.flow_sensor.setpoint_ml_min == 0.4 and node.config.recovery == 0.8
