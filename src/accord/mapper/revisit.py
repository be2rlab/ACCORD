import numpy as np

from . import sim3
from .filtering import usable


def stack(submaps, poses, scales, cfg):
    depth, K, T, window = [], [], [], []
    for m, sm in enumerate(submaps):
        factor = np.float32(scales[m])
        for i in range(len(sm['depth'])):
            ok = usable(sm['depth'][i], cfg, sm['sky'][i] if 'sky' in sm else None, sm['conf'][i])
            depth.append(np.where(ok, sm['depth'][i] * factor, np.nan).astype(np.float32))
            K.append(np.asarray(sm['intrinsics'][i], float))
            T.append(np.asarray(poses[m], float) @ np.asarray(sm['poses'][i], float))
            window.append(m)
    return dict(depth=np.stack(depth), K=np.stack(K), T=np.stack(T), window=np.asarray(window))


def project(i, D, cos_min, step, gate):
    h, w = D['depth'].shape[1:]
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    v, u = v.ravel(), u.ravel()
    z = D['depth'][i][v, u]
    keep = np.isfinite(z)
    u, v, z = u[keep], v[keep], z[keep].astype(np.float64)
    n = len(D['depth'])
    median = np.full(n, np.nan)
    if len(z) < 100:
        return median, np.zeros(n, int)
    cam = np.column_stack((u, v, np.ones(len(z)))) @ np.linalg.inv(D['K'][i]).T * z[:, None]
    X = (cam @ D['T'][i][:3, :3].T + D['T'][i][:3, 3]).astype(np.float32)
    R, c = D['T'][:, :3, :3].astype(np.float32), D['T'][:, :3, 3].astype(np.float32)
    rel = X[None] - c[:, None]
    Xc = np.matmul(rel, R)
    zc = Xc[..., 2]
    uvw = np.matmul(Xc, D['K'].astype(np.float32).transpose(0, 2, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        ub = np.rint(uvw[..., 0] / zc).astype(np.int32)
        vb = np.rint(uvw[..., 1] / zc).astype(np.int32)
        own = X - c[i]
        cos = (np.einsum('jnk,nk->jn', rel, own)
               / (np.linalg.norm(rel, axis=2) * np.linalg.norm(own, axis=1)[None]))
    ok = (zc > .15) & (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h) & (cos >= cos_min)
    ub, vb = np.where(ok, ub, 0), np.where(ok, vb, 0)
    zb = D['depth'][np.arange(n)[:, None], vb, ub]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(zb) - np.log(zc)
    ok &= np.isfinite(r) & (np.abs(r) < gate)       # beyond the gate: an occlusion, not a scale error
    count = ok.sum(1)
    rows = np.flatnonzero(count >= 50)
    if len(rows):
        median[rows] = np.nanmedian(np.where(ok, r, np.nan)[rows], axis=1)
    return median, count


def agreement(D, cfg):
    n = len(D['depth'])
    cos_min = float(np.cos(np.deg2rad(cfg['revisit_max_angle_deg'])))
    median, count = np.full((n, n), np.nan), np.zeros((n, n), np.int32)
    for i in range(n):
        median[i], count[i] = project(i, D, cos_min, int(cfg['revisit_step']), float(cfg['revisit_gate_log']))
    return median, count


def solve(median, count, window, cfg, rounds=10):
    median, window = np.asarray(median, float), np.asarray(window)
    need = int(cfg['revisit_min_shared'])
    a_all = (median - median.T) / 2
    ok = (count >= need) & (count.T >= need) & (window[:, None] != window[None, :]) & np.isfinite(a_all)
    n = len(median)
    i, j = np.nonzero(np.triu(ok, 1))
    degree = np.bincount(np.r_[i, j], minlength=n)
    if not len(i):
        return np.zeros(n), degree
    a = a_all[i, j]
    prior = np.diag(np.broadcast_to(np.asarray(cfg['revisit_prior_pairs'], float), n))
    c, w = np.zeros(n), np.ones(len(a))
    for _ in range(rounds):
        L, b = np.zeros((n, n)), np.zeros(n)
        np.add.at(L, (i, i), w)
        np.add.at(L, (j, j), w)
        np.add.at(L, (i, j), -w)
        np.add.at(L, (j, i), -w)
        np.add.at(b, j, w * a)
        np.add.at(b, i, -w * a)
        c = np.linalg.solve(L + prior, b)
        r = a - (c[j] - c[i])
        sigma = max(1.4826 * np.median(np.abs(r)), 1e-4)        # about zero, not about the median
        w = np.minimum(1., float(cfg['revisit_huber']) * sigma / np.maximum(np.abs(r), 1e-12))
    return c, degree


def corrections(submaps, poses, scales, cfg):
    D = stack(submaps, poses, scales, cfg)
    c, degree = solve(*agreement(D, cfg), D['window'], cfg)
    info = dict(images=int(len(c)), with_pairs=float(np.mean(degree > 0)),
                pairs_per_image_median=float(np.median(degree[degree > 0])) if (degree > 0).any() else 0.,
                correction_abs_median=float(np.median(np.abs(c))),
                correction_abs_max=float(np.abs(c).max()))
    return [c[D['window'] == m] for m in range(len(submaps))], info


def prepare(sm, cfg, shrink):
    """A window's images as ``Agreement`` keeps them: the usable depth at 1/``shrink`` of its grid (NaN elsewhere), the
    intrinsics of that grid, the camera in the window's frame, and the distance of the farthest point at scale 1."""
    s = int(shrink)
    grid, out = np.diag([1. / s, 1. / s, 1.]), []
    for i in range(len(sm['depth'])):
        ok = usable(sm['depth'][i], cfg, sm['sky'][i] if 'sky' in sm else None, sm['conf'][i])
        depth = np.where(ok, sm['depth'][i], np.nan)[::s, ::s].astype(np.float32)
        K = grid @ np.asarray(sm['intrinsics'][i], float)
        v, u = np.nonzero(np.isfinite(depth))
        rays = np.linalg.norm(np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(K).T, axis=1)
        out.append(dict(depth=depth, K=K, local=np.asarray(sm['poses'][i], float),
                        radius=float((rays * depth[v, u]).max()) if len(u) else 0.))
    return out


class Agreement:
    """``corrections`` one window at a time, as a run maps them. Each image keeps its usable depth (at 1/``shrink`` of
    its resolution, the source pixels as sparse as at export) and its camera; a pair of images of two windows is
    measured when either arrives or its window is placed anew, and only if the two can see a common point."""

    CHUNK = 64          # images projected into at once

    def __init__(self, cfg, shrink=1, shared_only=False):
        """``shared_only``: an image of another window is projected back into this window's only where this one's
        projection found the surface shared enough for ``solve`` to count the pair (``revisit_min_shared``, each way);
        the corrections are the same, the pairs it would not count stay unmeasured."""
        self.cfg, self.shrink, self.shared_only = cfg, int(shrink), shared_only
        self.step = max(1, int(cfg['revisit_step']) // self.shrink)
        self.cos_min = float(np.cos(np.deg2rad(cfg['revisit_max_angle_deg'])))
        self.gate = float(cfg['revisit_gate_log'])
        self.rows = {}                  # window -> its images' rows, in the order of its images
        self.depth, self.K, self.local, self.radius, self.window = [], [], [], [], []      # by row; depth at scale 1
        self.T, self.scale = [], []
        self.median, self.count = np.full((0, 0), np.nan), np.zeros((0, 0), np.int32)

    def place(self, key, pose, scale, sm=None):
        """A window where the map has it: new (with its submap ``sm``) or moved; its pairs are measured again."""
        if key not in self.rows:
            self.add(key, sm)
        self.set(key, pose, scale)
        self.measure(self.rows[key])

    def set(self, key, pose, scale):
        """A window where the map has it now, its pairs as they were measured."""
        for r in self.rows[key]:
            self.T[r], self.scale[r] = np.asarray(pose, float) @ self.local[r], float(scale)

    def add(self, key, sm=None, images=None):
        """A new window, from its submap ``sm`` or from its ``images`` as ``prepare`` keeps them."""
        first = len(self.depth)
        for image in images if images is not None else prepare(sm, self.cfg, self.shrink):
            self.depth.append(image['depth'])
            self.K.append(image['K'])
            self.local.append(image['local'])
            self.radius.append(image['radius'])
            self.window.append(len(self.rows))
            self.T.append(None)
            self.scale.append(1.)
        self.rows[key] = list(range(first, len(self.depth)))
        n = len(self.depth)
        median, count = np.full((n, n), np.nan), np.zeros((n, n), np.int32)
        median[:first, :first], count[:first, :first] = self.median, self.count
        self.median, self.count = median, count

    def measure(self, rows, windows=None):
        """The pairs of the images ``rows`` (of one window) with those of the other windows, or of ``windows`` only."""
        rows = np.asarray(rows)
        window = np.asarray(self.window)
        others = np.flatnonzero(window != window[rows[0]])
        if windows is not None:
            others = others[np.isin(window[others], np.asarray(list(windows)))]
        self.median[np.ix_(rows, others)], self.median[np.ix_(others, rows)] = np.nan, np.nan
        self.count[np.ix_(rows, others)], self.count[np.ix_(others, rows)] = 0, 0
        if not len(others):
            return
        centre = np.array([T[:3, 3] for T in self.T])
        reach = np.asarray(self.radius) * np.asarray(self.scale)
        gap = np.linalg.norm(centre[others][:, None] - centre[rows][None], axis=2)
        near = others[(gap <= reach[others][:, None] + reach[rows][None]).any(axis=1)]
        for r in rows:
            for k in range(0, len(near), self.CHUNK):
                self.pair(r, near[k:k + self.CHUNK])
        need = int(self.cfg['revisit_min_shared'])
        for j in near:
            targets = rows[self.count[rows, j] >= need] if self.shared_only else rows
            if len(targets):
                self.pair(j, targets)

    def pair(self, i, targets):
        index = np.r_[i, targets]
        D = dict(depth=np.stack([self.depth[k] * np.float32(self.scale[k]) for k in index]),
                 K=np.stack([self.K[k] for k in index]), T=np.stack([self.T[k] for k in index]))
        median, count = project(0, D, self.cos_min, self.step, self.gate)
        self.median[i, targets], self.count[i, targets] = median[1:], count[1:]

    def corrections(self):
        """Each window's corrections and how many images of other windows each of its images shares a surface with."""
        if not self.depth:
            return {}
        c, degree = solve(self.median, self.count, np.asarray(self.window), self.cfg)
        return {key: (c[rows], degree[rows]) for key, rows in self.rows.items()}

    def pairs(self):
        """The pairs ``solve`` counts, and their log ratio (the second image's depth over the first's)."""
        need, window = int(self.cfg['revisit_min_shared']), np.asarray(self.window)
        a = (self.median - self.median.T) / 2
        return (self.count >= need) & (self.count.T >= need) & (window[:, None] != window[None, :]) & np.isfinite(a), a

    def rejected(self, limit):
        """The images to leave out of a map drawn as it is: while an image puts the surface it shares with most of the
        others it is paired with more than ``limit`` (log) away from where they put it, the worst of them, one at a
        time; of equals, the one of the earlier window. Not ``corrections``: its robust solve gives up an image that
        disagrees with every other rather than measure it. As (window, image) pairs."""
        if not self.depth:
            return set()
        ok, a = self.pairs()
        conflict = ok & (np.abs(a) > limit)
        alive, out, window = np.ones(len(ok), bool), [], np.asarray(self.window)
        while True:
            paired, against = (ok & alive).sum(1), (conflict & alive).sum(1)
            share = np.where(alive, against / np.maximum(paired, 1), 0.)
            worst = np.flatnonzero(share > .5)
            if not len(worst):
                break
            r = worst[np.lexsort((window[worst], -against[worst], -share[worst]))[0]]
            alive[r] = False
            out.append(r)
        keys, index = list(self.rows), {r: i for rows in self.rows.values() for i, r in enumerate(rows)}
        return {(keys[window[r]], index[r]) for r in out}


class OnlineCorrection:
    """The agreement correction while the map is built, in batches (mapper/online.py). A pair of windows is measured
    again only when the placement of one in the other's frame has moved by more than ``online_correction_move_m`` (or its
    rotation or log-scale by as much at three metres); a shift of the whole map is no move. An image is projected back
    only into the images that share enough with it. The corrections are those of the same measurements made in full."""

    REACH_M = 3.

    def __init__(self, cfg):
        self.agreement = Agreement(cfg, shrink=int(cfg['online_correction_shrink']), shared_only=True)
        self.move = float(cfg['online_correction_move_m'])
        self.placed, self.values, self.keys = {}, {}, []
        self.back = np.zeros((0, 0, 4, 4))     # per pair of windows: the inverse of b's placement in a's frame, as measured

    def update_batch(self, windows, nodes):
        """The new ``windows`` [(key, images as ``prepare`` keeps them)], in the order they were mapped, and the graph's
        current Sim(3) ``nodes`` (a window the graph no longer holds keeps its last placement): every window's per-image
        log corrections, solved now, {key: array}."""
        for key, images in windows:
            self.agreement.add(key, images=images)
            self.keys.append(key)
        for key in self.keys:
            if key in nodes:
                self.placed[key] = np.asarray(nodes[key], float)
            self.agreement.set(key, sim3.pose(self.placed[key]), sim3.scale(self.placed[key]))
        n, old = len(self.keys), len(self.back)
        if not n:
            return {}
        N = np.stack([self.placed[k] for k in self.keys])
        rel = np.linalg.inv(N)[:, None] @ N[None]                   # [a, b]: b in a's frame
        pending = np.ones((n, n), bool)
        if old:
            pending[:old, :old] = self.changed(self.back @ rel[:old, :old])
        np.fill_diagonal(pending, False)
        pending |= pending.T
        back = np.tile(np.eye(4), (n, n, 1, 1))
        back[:old, :old] = self.back
        while pending.any():        # from the window with the most pairs to measure: one pass over its images
            a = int(np.argmax(pending.sum(1)))
            partners = np.flatnonzero(pending[a])
            self.agreement.measure(self.agreement.rows[self.keys[a]], windows=partners)
            back[a, partners], back[partners, a] = np.linalg.inv(rel[a, partners]), np.linalg.inv(rel[partners, a])
            pending[a, partners] = pending[partners, a] = False
        self.back = back
        self.values = {k: c for k, (c, _) in self.agreement.corrections().items()}
        return self.values

    def changed(self, delta):
        """Of relative Sim(3) changes ``delta`` [..., 4, 4]: those beyond ``move`` (a turn or log scale as much at REACH_M)."""
        s = np.cbrt(np.linalg.det(delta[..., :3, :3]))
        R = delta[..., :3, :3] / s[..., None, None]
        shift = np.linalg.norm(delta[..., :3, 3], axis=-1)
        turn = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1., 1.))
        return np.maximum(shift, (turn + np.abs(np.log(s))) * self.REACH_M) > self.move


def attach_stored(submaps, index):
    """The online corrections a run held at its end (``map_index.json``), onto the submaps they belong to; a window the
    run never solved gets none."""
    for sm in submaps:
        stored = index['submaps'].get(sm['id'], {}).get('log_corrections')
        if stored is not None:
            sm['log_corrections'] = np.asarray(stored, float)
