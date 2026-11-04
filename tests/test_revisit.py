import numpy as np
import pytest

from accord.mapper import revisit


@pytest.fixture
def corrections(wall):
    """The corrections of a wall's images, as one array, and their summary."""
    def run(errors, cfg, poses=None, scales=None):
        submaps, stand = wall(errors)
        applied, info = revisit.corrections(submaps, poses or stand, scales or np.ones(len(submaps)), cfg)
        return np.concatenate(applied), info
    return run


@pytest.fixture
def weak(mapping):
    """The settings with next to no prior on the corrections."""
    return dict(mapping, revisit_prior_pairs=1e-6)


def test_two_images_of_a_wall_measure_the_difference_of_their_scale_errors(wall, mapping):
    e = [.03, -.05, 0., .02]
    median, count = revisit.agreement(revisit.stack(*wall(e), np.ones(2), mapping), mapping)
    assert count[0, 2] > 2000
    assert median[0, 2] == pytest.approx(e[2] - e[0], abs=1e-4)
    assert median[2, 0] == pytest.approx(e[0] - e[2], abs=1e-4)


def test_an_occluder_or_a_hole_in_the_other_image_is_not_a_scale_error(wall, mapping):
    D = revisit.stack(*wall([0., 0., 0., 0.]), np.ones(2), mapping)
    whole = revisit.agreement(D, mapping)[1][0, 2]
    D['depth'][2][:, :250] = np.nan
    D['depth'][2][:, 250:300] = .5          # something in front of the wall
    median, count = revisit.agreement(D, mapping)
    assert median[0, 2] == pytest.approx(0., abs=1e-4) and count[0, 2] < .6 * whole


def test_the_differences_between_images_are_recovered_and_the_corrections_sum_to_zero(corrections, weak):
    e = np.array([.04, -.02, .00, .06, -.05, .01])
    c, info = corrections(e, weak)
    np.testing.assert_allclose(c, e - e.mean(), atol=2e-3)
    assert abs(c.sum()) < 1e-6
    assert info['images'] == 6 and info['with_pairs'] == 1. and info['pairs_per_image_median'] == 4      # not its own window


def test_the_map_loses_its_disagreement_and_keeps_its_mean_scale(corrections, mapping):
    e = .05 * np.random.default_rng(0).standard_normal(12)
    c, _ = corrections(e, mapping)
    left = e - c
    assert np.std(left) < .25 * np.std(e)
    assert np.mean(left) == pytest.approx(np.mean(e), abs=1e-9)


def test_an_image_that_shares_no_surface_is_left_alone(wall, corrections, mapping, weak):
    e = [.03, .03, -.03, -.03, .08, .08]
    _, poses = wall(e)
    poses[2][0, 3] = 50.                    # the third window looks at another part of the wall
    c, info = corrections(e, weak, poses)
    np.testing.assert_allclose(c, [.03, .03, -.03, -.03, 0., 0.], atol=2e-3)
    assert info['with_pairs'] == pytest.approx(4 / 6)
    # The prior weighs as one pair, and each image has two: (2 + 1) c + 2 c = 2 * 0.06.
    np.testing.assert_allclose(corrections(e, mapping, poses)[0][:2], .024, atol=2e-3)


def test_one_wrong_pair_does_not_move_the_others(wall, mapping):
    submaps, poses = wall(np.zeros(16), spacing=.05)
    D = revisit.stack(submaps, poses, np.ones(8), mapping)
    median, count = revisit.agreement(D, mapping)
    median[0, 5] += .4
    c, _ = revisit.solve(median, count, D['window'], mapping)
    assert np.abs(c).max() < .02


def test_a_window_whose_scale_already_repairs_its_depth_needs_no_correction(corrections, weak):
    e = [.06, .06, 0., 0., 0., 0.]
    assert corrections(e, weak)[0][:2].mean() > .03
    np.testing.assert_allclose(corrections(e, weak, scales=[np.exp(-.06), 1., 1.])[0], 0., atol=2e-3)


def test_the_agreement_kept_one_window_at_a_time_is_the_one_measured_at_once(wall, mapping):
    e = .05 * np.random.default_rng(1).standard_normal(8)
    submaps, poses = wall(e)
    scales = [1., 1.1, .95, 1.]
    batch, _ = revisit.corrections(submaps, poses, scales, mapping)
    kept = revisit.Agreement(mapping)
    for m, sm in enumerate(submaps):
        kept.place(m, poses[m], 1., sm)
    for m in range(len(submaps)):          # placed anew: a window the graph rescaled
        kept.place(m, poses[m], scales[m])
    online = kept.corrections()
    np.testing.assert_allclose(np.concatenate([online[m][0] for m in range(len(submaps))]), np.concatenate(batch), atol=1e-9)
    assert all((online[m][1] > 0).all() for m in online)


def test_images_too_far_apart_to_share_a_point_are_not_paired(wall, mapping):
    submaps, poses = wall([0., 0., .1, .1])
    poses[1][0, 3] = 100.
    kept = revisit.Agreement(mapping)
    for m, sm in enumerate(submaps):
        kept.place(m, poses[m], 1., sm)
    assert (kept.count == 0).all()
    assert all((c == 0).all() and (degree == 0).all() for c, degree in kept.corrections().values())


def test_a_coarser_agreement_finds_the_same_scale_errors(wall, weak):
    e = np.array([.04, -.02, .00, .06, -.05, .01])
    submaps, poses = wall(e)
    kept = revisit.Agreement(weak, shrink=2)
    for m, sm in enumerate(submaps):
        kept.place(m, poses[m], 1., sm)
    c = np.concatenate([kept.corrections()[m][0] for m in range(3)])
    np.testing.assert_allclose(c, e - e.mean(), atol=3e-3)


def kept(submaps, poses, cfg):
    agreement = revisit.Agreement(cfg)
    for m, sm in enumerate(submaps):
        agreement.place(m, poses[m], 1., sm)
    return agreement


def test_the_window_that_disagrees_with_all_the_others_is_left_out_and_they_are_kept(wall, mapping):
    e = .01 * np.random.default_rng(2).standard_normal(10)
    e[4:6] += .3
    assert kept(*wall(e), mapping).rejected(.15) == {(2, 0), (2, 1)}


def test_of_two_windows_that_disagree_the_earlier_is_left_out(wall, mapping):
    assert kept(*wall([.3, .3, 0., 0.]), mapping).rejected(.15) == {(0, 0), (0, 1)}
    assert kept(*wall([0., 0., .3, .3]), mapping).rejected(.15) == {(0, 0), (0, 1)}


def test_a_scale_that_drifts_along_the_path_leaves_no_image_out(wall, mapping):
    # Images 0.5 m apart, 2 m from the wall: within 30 degrees of each other's view, an image shares the wall with the
    # next two or three only.
    e = np.repeat(np.linspace(0., .25, 6), 2)       # 0.25 from end to end, 0.05 from window to window
    agreement = kept(*wall(e, spacing=.5), mapping)
    ok, _ = agreement.pairs()
    assert ok[0, 2] and not ok[0, 3:].any()
    assert agreement.rejected(.15) == set()
