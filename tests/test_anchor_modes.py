"""The simplest anchors, as arms of the system (docs/ABLATION.md): one offset for the whole map at export
(`global`), and each image's inverse depth fitted to its own triangulated keypoints (`image_affine`, VI-Depth's recipe)."""
import numpy as np
import pytest

from accord.mapper import mapper as mapper_module
from accord.mapper.mapper import Mapper, affine_inverse_depth, global_offset


def residuals(frames, log_error, n=60, seed=0):
    """Targets per frame: predicted depth exp(log_error) times the triangulated one, 1 to 4 m."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in range(frames):
        z_tri = rng.uniform(1., 4., n)
        z_pred = z_tri * np.exp(log_error[f])
        out[f] = [(np.log(z_pred) - np.log(z_tri), np.ones(n), z_pred, z_tri)]
    return out


@pytest.fixture
def bare(mapping, matcher, tmp_path, monkeypatch):
    monkeypatch.setattr(mapper_module, 'Features', lambda settings: matcher)
    def make(**over):
        return Mapper(dict(mapping, anchor_min_targets=30, **over), tmp_path)
    return make


def test_global_mode_keeps_the_depth_and_the_targets_for_the_export(bare):
    m = bare(anchor_mode='global')
    sm = dict(depth=np.full((3, 4, 4), 2., np.float32))
    report = m.anchor(sm, residuals(3, [.05, .05, .05]))
    assert np.all(sm['depth'] == 2.)                                       # nothing applied while mapping
    assert report['mode'] == 'global' and report['keypoints'] == 180
    np.testing.assert_allclose(sm['anchor_residuals'], .05)
    other = dict(anchor_residuals=np.full(10, -.05), anchor_weights=np.ones(10))
    expected = mapper_module.robust_mean(np.r_[np.full(180, .05), np.full(10, -.05)], np.ones(190))
    assert global_offset([sm, other, dict()], dict(anchor_max_log=.3)) == pytest.approx(expected, abs=1e-12)
    assert global_offset([dict()], dict(anchor_max_log=.3)) == 0.


def test_inverse_depth_affine_is_recovered():
    rng = np.random.default_rng(1)
    z_tri = rng.uniform(1., 5., 200)
    z_pred = 1. / ((1. / z_tri - .02) / 1.1)          # 1/z_tri = 1.1 / z_pred + 0.02
    a, b = affine_inverse_depth(z_pred, z_tri, np.ones(200))
    assert a == pytest.approx(1.1, rel=1e-6) and b == pytest.approx(.02, abs=1e-8)
    z_pred[:20] *= 3.                                  # outliers
    a, b = affine_inverse_depth(z_pred, z_tri, np.ones(200))
    assert a == pytest.approx(1.1, rel=1e-3) and b == pytest.approx(.02, abs=1e-3)


def test_image_affine_mode_aligns_each_image_to_its_own_targets(bare):
    m = bare(anchor_mode='image_affine')
    sm = dict(depth=np.stack([np.full((4, 4), 2. * np.exp(e), np.float32) for e in (.10, -.05, 0.)]))
    report = m.anchor(sm, residuals(3, [.10, -.05, 0.]))
    np.testing.assert_allclose(sm['depth'], 2., rtol=1e-5)                 # each image to its own scale
    assert report['mode'] == 'image_affine' and report['images_aligned'] == 3


def test_the_window_mode_is_the_system(bare):
    m = bare()
    sm = dict(depth=np.full((2, 4, 4), 2. * np.exp(.04), np.float32))
    report = m.anchor(sm, residuals(2, [.04, .04]))
    np.testing.assert_allclose(sm['depth'], 2., rtol=1e-5)
    assert report['log_offset'] == pytest.approx(.04, abs=1e-6)
