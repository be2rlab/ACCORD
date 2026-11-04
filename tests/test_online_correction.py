"""The agreement correction while the map is built (mapper/online.py, revisit.OnlineCorrection)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from accord.mapper import revisit
from accord.mapper.fusion import world_points

ROOT = Path(__file__).resolve().parents[1]
W, H = 240, 180
K = np.array([[200., 0., 120.], [0., 200., 90.], [0., 0., 1.]])


def config(**over):
    cfg = yaml.safe_load((ROOT / 'config/accord.yaml').read_text())['mapping']
    cfg.update(revisit_min_shared=100, online_correction_shrink=1, online_correction_move_m=0.02)
    cfg.update(over)
    return cfg


def camera(x):
    T = np.eye(4)
    T[0, 3] = x
    return T


def submap(key, xs, factor=1.):
    """A window looking at the plane z = 3 from cameras at x in ``xs`` (in the window's frame), its depth times
    ``factor``."""
    n = len(xs)
    return dict(id=key, depth=np.full((n, H, W), 3. * factor, np.float32), conf=np.ones((n, H, W), np.float32),
                intrinsics=np.repeat(K[None], n, axis=0), poses=np.stack([camera(x) for x in xs]),
                rgb=np.zeros((n, H, W, 3), np.uint8), frame_ids=[f'{key}{i}' for i in range(n)])


def test_world_points_applies_log_correction():
    sm = submap('a', [0.])
    plain, _ = world_points(sm, 0, np.eye(4), pixel_step=8)
    sm['log_corrections'] = np.array([np.log(2.)])
    corrected, _ = world_points(sm, 0, np.eye(4), pixel_step=8)
    assert np.allclose(plain[:, 2], 3.)
    assert np.allclose(corrected[:, 2], 1.5)                 # depth times exp(-c)


def test_online_correction_measures_relative_scale():
    online = revisit.OnlineCorrection(config())
    a, b = submap('a', [0., .2]), submap('b', [.1, .3], factor=1.04)      # b's depth 4 % long
    nodes = {'a': np.eye(4)}
    online.update_batch([('a', revisit.prepare(a, online.agreement.cfg, 1))], nodes)
    nodes['b'] = np.eye(4)
    values = online.update_batch([('b', revisit.prepare(b, online.agreement.cfg, 1))], nodes)
    assert set(values) == {'a', 'b'} and len(values['a']) == 2 and len(values['b']) == 2
    gap = values['b'].mean() - values['a'].mean()
    assert gap > 0                                           # the long window gets the larger correction
    assert gap == pytest.approx(np.log(1.04), rel=.4)        # the zero prior shrinks it a little
    assert abs(np.concatenate(list(values.values())).sum()) < 1e-6      # the mean scale is kept


def test_stored_corrections_go_to_their_submaps_only():
    submaps = [submap('a', [0., .2]), submap('b', [.1, .3])]
    index = {'submaps': {'a': dict(log_corrections=[.01, -.02]), 'b': dict()}}
    revisit.attach_stored(submaps, index)
    assert np.allclose(submaps[0]['log_corrections'], [.01, -.02])
    assert 'log_corrections' not in submaps[1]               # a window the run never solved: no correction


def shifted(node, dx):
    out = np.array(node, float)
    out[0, 3] += dx
    return out


def counting(online):
    """The number of images projected since the last call, through ``online``'s agreement."""
    calls, pair = [0], online.agreement.pair

    def counted(i, targets):
        calls[0] += 1
        return pair(i, targets)
    online.agreement.pair = counted

    def take():
        n, calls[0] = calls[0], 0
        return n
    return take


def test_a_rigid_shift_of_the_whole_map_is_no_move_of_the_whole_map_alone(wall, mapping):
    submaps, poses = wall([.03, .03, -.02, -.02, .05, .05])
    online = revisit.OnlineCorrection(config(revisit_min_shared=500, online_correction_shrink=2))
    nodes = {m: poses[m] for m in range(3)}
    first = online.update_batch([(m, revisit.prepare(sm, online.agreement.cfg, 2)) for m, sm in enumerate(submaps)], nodes)
    calls = counting(online)
    moved = online.update_batch([], {m: shifted(n, .3) for m, n in nodes.items()})
    assert calls() == 0
    for m in first:
        np.testing.assert_allclose(moved[m], first[m])


def test_only_the_pairs_of_a_window_moved_against_the_others_are_measured_of_a_window_moved_against_the_others(wall, mapping):
    submaps, poses = wall([.03, .03, -.02, -.02, .05, .05])
    online = revisit.OnlineCorrection(config(revisit_min_shared=500, online_correction_shrink=2))
    nodes = {m: poses[m] for m in range(3)}
    online.update_batch([(m, revisit.prepare(sm, online.agreement.cfg, 2)) for m, sm in enumerate(submaps)], nodes)
    calls = counting(online)
    online.update_batch([], nodes | {0: shifted(nodes[0], .05)})
    assert 0 < calls() <= 8                                  # window 0 against 1 and 2, not 1 against 2
    online.update_batch([], nodes | {0: shifted(nodes[0], .06)})
    assert calls() == 0                                      # 1 cm more: within move_m


def test_the_correction_process_takes_windows_in_batches_and_answers_the_end(wall, mapping):
    import queue
    import threading
    from accord.mapper import online
    submaps, poses = wall([.03, .03, -.02, -.02, .05, .05])
    cfg = config(revisit_min_shared=500, online_correction_shrink=2)
    inbox, outbox = queue.Queue(), queue.Queue()
    client = online.Client(inbox, outbox)
    nodes = {}
    for m, sm in enumerate(submaps):
        nodes[f'{m:06d}'] = poses[m]
        client.submit(f'{m:06d}', revisit.prepare(sm, cfg, 2), dict(nodes))
    loop = threading.Thread(target=online.correction_loop, args=(inbox, outbox, cfg))
    loop.start()
    final = client.finish(timeout_s=60.)
    loop.join(timeout=10.)
    assert final is not None and final['final'] and final['through'] == '000002'
    assert set(final['values']) == {'000000', '000001', '000002'}
    assert final['stats']['windows'] == 3 and final['stats']['batches'] >= 1
    expected = revisit.OnlineCorrection(cfg).update_batch(
        [(f'{m:06d}', revisit.prepare(sm, cfg, 2)) for m, sm in enumerate(submaps)], nodes)
    for k in expected:
        np.testing.assert_allclose(final['values'][k], expected[k], atol=1e-12)


def test_the_correction_process_runs_as_a_process_of_its_own(wall, mapping):
    from accord.mapper import online
    submaps, poses = wall([.03, .03, -.02, -.02])
    cfg = config(revisit_min_shared=500, online_correction_shrink=2)
    process = online.Process(cfg)
    client = online.Client(*process.queues)
    for m, sm in enumerate(submaps):
        client.submit(f'{m:06d}', revisit.prepare(sm, cfg, 2), {f'{k:06d}': poses[k] for k in range(m + 1)})
    final = client.finish(timeout_s=60.)
    process.close()
    assert final is not None and 'error' not in final and set(final['values']) == {'000000', '000001'}
    assert not process.process.is_alive()
