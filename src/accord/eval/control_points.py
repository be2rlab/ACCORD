"""Trajectory accuracy at surveyed control points (Hilti-Oxford), the dataset's own protocol (Zhang et al., RA-L
2023): the trajectory is fitted to the control points by one rigid transform, and each point scores 10, 6, 3, 1 or
0 for an error below 1, 3, 6 or 10 cm, or above; the sequence's score is the per cent of the maximum."""
import numpy as np

from .metrics import conditioning, se3_align

SCORE_EDGES_M = (.01, .03, .06, .10)
SCORE_POINTS = (10, 6, 3, 1, 0)
MAX_BRACKET_S = .5          # an estimate is read at a control point only between poses this close in time
MIN_POINTS = 3


def point_score(error_m):
    return SCORE_POINTS[int(np.searchsorted(SCORE_EDGES_M, float(error_m), side='right'))]


def positions_at(trajectory, times, max_bracket_s=MAX_BRACKET_S):
    """The estimated position at each time, linear between the poses around it -> (p, found)."""
    t = np.asarray([row[0] for row in trajectory], float)
    p = np.asarray([row[1] for row in trajectory], float).reshape(len(t), 3)
    out, found = np.zeros((len(times), 3)), np.zeros(len(times), bool)
    for k, when in enumerate(np.asarray(times, float)):
        j = int(np.searchsorted(t, when))
        if j == 0 or j == len(t) or t[j] - t[j - 1] > max_bracket_s:
            continue
        w = (when - t[j - 1]) / (t[j] - t[j - 1])
        out[k], found[k] = (1. - w) * p[j - 1] + w * p[j], True
    return out, found


def fit(trajectory, times, points, max_bracket_s=MAX_BRACKET_S):
    """The rigid transform of the trajectory onto the control points it covers -> (R, t, found, estimate), or None."""
    estimate, found = positions_at(trajectory, times, max_bracket_s)
    if int(found.sum()) < MIN_POINTS:
        return None
    rot, trans, _ = se3_align(estimate[found].T, np.asarray(points, float)[found].T)
    return rot, trans, found, estimate


def score(trajectory, times, points, max_bracket_s=MAX_BRACKET_S):
    """Errors at the control points after the fit, and the dataset's score. A point the trajectory does not cover
    scores zero and still counts, so a trajectory that stops early cannot raise the score; the fit and the RMSE use
    the covered points."""
    points = np.asarray(points, float).reshape(-1, 3)
    fitted = fit(trajectory, times, points, max_bracket_s)
    if fitted is None:
        return dict(n_points=len(points), n_covered=int(positions_at(trajectory, times, max_bracket_s)[1].sum()),
                    rmse_m=None, errors_m=None, score=None, conditioning=None,
                    reason=f'fewer than {MIN_POINTS} control points are covered by the trajectory')
    rot, trans, found, estimate = fitted
    covered = np.linalg.norm((rot @ estimate[found].T + trans).T - points[found], axis=1)
    errors = np.full(len(points), np.nan)
    errors[found] = covered
    return dict(n_points=len(points), n_covered=int(found.sum()), rmse_m=float(np.sqrt(np.mean(covered ** 2))),
                mean_m=float(covered.mean()), max_m=float(covered.max()),
                errors_m=[None if np.isnan(e) else float(e) for e in errors],
                score=float(100. * sum(point_score(e) for e in covered) / (10. * len(points))),
                conditioning=conditioning(points[found].T), reason='')
