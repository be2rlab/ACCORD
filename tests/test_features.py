"""The keypoint matcher's wrapper (accord.mapper.features), without XFeat: a stand-in model."""
from types import SimpleNamespace

import numpy as np

from accord.mapper.features import Features


def matcher(match):
    features = object.__new__(Features)
    features.torch = SimpleNamespace(as_tensor=lambda x, device: x)
    features.model = SimpleNamespace(dev='cpu', match_lighterglue=match)
    return features


def kit(n):
    return dict(pixels=np.zeros((n, 2)), descriptors=np.zeros((n, 64)), shape=(640, 480))


def test_matches_come_back_as_index_pairs():
    pairs = matcher(lambda a, b: (None, None, np.array([[0., 1.], [2., 0.]])))
    out = pairs.match(kit(3), kit(2))
    assert out.dtype.kind == 'i' and out.tolist() == [[0, 1], [2, 0]]


def test_lightglue_pruning_every_point_of_an_image_is_no_match():
    def pruned(a, b):       # kornia 0.8.2, filter_matches on an image left with no point
        raise IndexError('max(): Expected reduction dim 1 to have non-zero size.')
    out = matcher(pruned).match(kit(2048), kit(2048))
    assert out.shape == (0, 2) and out.dtype.kind == 'i'
