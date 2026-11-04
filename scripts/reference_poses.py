#!/usr/bin/env python3
"""A sequence's reference camera poses as a TUM file (t x y z qx qy qz qw, camera to world), for `scripts/run.py
--poses`: the reference the scorers use (EuRoC: OpenVINS's corrected file; ORI: the poses registered to the scan; TUM-VI:
the motion capture), carried from the body to the camera by the dataset's calibration.

    python3 scripts/reference_poses.py --dataset euroc --data data/euroc --sequence V1_01_easy --out FILE
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    from accord.datasets import open_dataset
    from accord.geometry import tum_line
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    ds = open_dataset(a.dataset, a.data, a.sequence)
    ref = ds.groundtruth()
    camera_to_body = np.eye(4)
    camera_to_body[:3, :3], camera_to_body[:3, 3] = ds.R_CtoI, ds.p_IC
    with a.out.open('w') as stream:
        for t, p_, R in zip(ref.t, ref.p, ref.R):
            body = np.eye(4)
            body[:3, :3], body[:3, 3] = R, p_
            stream.write(tum_line(float(t), body @ camera_to_body))
    print(f'{len(ref.t)} poses -> {a.out}')


if __name__ == '__main__':
    main()
