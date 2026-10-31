import numpy as np
import pytest

from mpmon import render as R
from mpmon.ml import Dataset, Model, evaluate, make_dataset
from mpmon.network import Station, analyse, qc_recovery, reference_material, run_network
from mpmon.vision import FEATURES, features, match_ground_truth, segment


def _one(material, shape, size_um, cond=R.CLEAN, seed=0):
    rng = np.random.default_rng(seed)
    p = R.Particle(0, material, shape, size_um, 256, 256, 0.3)
    img, gt = R.render([p], cond, rng)
    lab, bg = segment(img)
    X, _ = features(img, lab, bg)
    return X, match_ground_truth(lab, gt)


def test_render_is_deterministic_and_labels_particles():
    a = R.render(R.sample_particles(np.random.default_rng(1), 5, 3), R.CLEAN, np.random.default_rng(2))
    b = R.render(R.sample_particles(np.random.default_rng(1), 5, 3), R.CLEAN, np.random.default_rng(2))
    assert np.array_equal(a[0], b[0])
    assert a[0].shape == (R.BANDS, R.SIZE, R.SIZE)
    assert set(np.unique(a[1])) - {0}


def test_bead_is_segmented_with_correct_size():
    X, match = _one("PE", "bead", 200)
    assert len(X) == 1 and match[0] == 1
    assert X[0, FEATURES.index("eq_diam_um")] == pytest.approx(190, rel=0.15)
    assert X[0, FEATURES.index("aspect")] < 1.4


def test_fibre_is_elongated():
    X, _ = _one("PET", "fiber", 400)
    assert X[0, FEATURES.index("aspect")] > 4


def test_spectral_ratios_separate_pe_from_pet():
    pe, _ = _one("PE", "bead", 150)
    pet, _ = _one("PET", "bead", 150)
    r0 = FEATURES.index("r0")
    assert pe[0, r0] > pet[0, r0] + 0.1


def test_empty_image_has_no_segments():
    img, gt = R.render([], R.CLEAN, np.random.default_rng(0))
    lab, bg = segment(img)
    assert lab.max() == 0
    assert features(img, lab, bg)[0].shape == (0, len(FEATURES))


@pytest.fixture(scope="module")
def small_model():
    train = Dataset([im for i, c in enumerate((R.CLEAN, R.TURBID, R.WEATHERED))
                     for im in make_dataset(6, c, 10 + i).images])
    return Model().fit(train).calibrate(make_dataset(4, R.CLEAN, 20))


def test_classifier_quality_on_clean_water(small_model):
    e = evaluate(small_model, make_dataset(4, R.CLEAN, 30))
    assert e["accuracy_material"] > 0.9
    assert e["false_positive_rate"] < 0.1
    assert 0 < e["segmentation_recall"] <= 1


def test_calibration_corrects_undercount(small_model):
    a, b = small_model.count_fit
    assert a >= 1.0 and b == 0.0


def test_reference_material_recovered_when_in_focus(small_model):
    ref = reference_material(np.random.default_rng(0))
    assert len(ref) == 50 and len({(p.cx, p.cy) for p in ref}) == 50
    assert qc_recovery(small_model, R.CLEAN, np.random.default_rng(1)) >= 0.9


def test_defocus_drops_fibre_reference_recovery(small_model):
    blurred = Station("s", 10, drift_from_week=0).condition(12)
    assert qc_recovery(small_model, blurred, np.random.default_rng(2), "beads+fibers") < 0.8


def test_station_event_and_network_analysis(small_model):
    st = Station("x", 10, event=(2, 3, 3.0))
    assert st.true_conc(2) == pytest.approx(30) and st.true_conc(4) == pytest.approx(10)
    db = run_network(small_model, weeks=5, stations=(st, Station("y", 5)), seed=3)
    out = analyse(db)
    assert set(out) == {"x", "y"}
    assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 10
    assert db.execute("SELECT COUNT(*) FROM qc").fetchone()[0] == 4  # weeks 0 and 4, two stations
