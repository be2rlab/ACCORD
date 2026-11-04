#!/usr/bin/env python3
"""An iPhone recording from Stray Scanner (rgb.mp4, odometry.csv, imu.csv) as an Accord sequence: data/iphone/<seq>/mav0,
the frames scaled to 640x480 and turned 90 degrees clockwise, upright for a phone held in portrait (or 180 degrees, for
one held in landscape with the camera's image upside down: --turn 180; or not turned, for one held in landscape the
other way: --turn 0), as JPEG under cam0/data/<ns>.jpg, the IMU as an ASL imu0/data.csv; beside it ARKit's own
trajectory (arkit.tum, not a reference) and conversion.json.

Every time is the recording's own (the phone's uptime: one clock for the frames and the IMU). The IMU is CoreMotion's:
the accelerometer in g with gravity's sign (a phone lying face up reads z = -1), written negated in m/s^2 as the
specific force the filter expects, and the gyroscope in rad/s as it is. Stray Scanner's poses are of OpenCV's camera of
the recorded frame (x right, y down, z forward), not of ARKit's (y up, z back). The camera is the rig's (config/iphone,
or --rig) and the recording is checked against it: ARKit's angular velocity, carried onto the IMU by the rig's rotation,
must match the gyroscope; the accelerometer must point against ARKit's gravity; the turned frames must be upright;
ARKit's intrinsics must be the rig's; and, once the frames are written, the rotation between frames their images give
(essential matrices) must match the gyroscope through the rig too. That last check is the one that sees how the camera
sits on the image: a wrong camera convention for the poses cancels out of the others. The camera-IMU time shift the fit
finds is reported, not applied.

    python3 scripts/convert_iphone.py data/Iphone/ID [--out data/iphone/ID]               # held in portrait
    python3 scripts/convert_iphone.py data/Iphone/ID --rig config/rig_iphone --turn 0      # the phone rig's iPhone, in landscape
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / 'config/iphone/kalibr_imucam_chain.yaml'
G = 9.80665                       # CoreMotion's unit
WIDTH = 640                       # the frames are scaled to this width before they are turned
# The camera of a frame turned clockwise, in the camera of the frame as recorded. 90 degrees: its x is the old -y, its
# y the old x (pixel (u, v) of a W x H frame goes to (H - 1 - v, u)); 180 degrees: x and y reversed ((W - 1 - u, H - 1 - v));
# 0 degrees: as recorded.
R_TURNED = {0: np.eye(3), 90: np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]), 180: np.diag([-1., -1., 1.])}
MAX_ROTATION_DEG = 3.             # the fitted camera-IMU rotation against the rig's
MAX_GYRO_RESIDUAL = .1            # rad/s, median
MIN_GRAVITY_COS = .95             # median cosine of the accelerometer against ARKit's up
MIN_UPRIGHT = .9                  # share of turned frames whose image down is within 45 degrees of gravity
MAX_FOCAL_ERROR = .03             # ARKit's focal length against the rig's, relative
MAX_CENTRE_ERROR_PX = 4.
PAIR_GAP, PAIR_STEP = 5, 6        # the image check's pairs: frames this far apart, one pair every this many frames
MIN_PAIR_ROTATION = .05           # rad: pairs that turn less say little about the axes
MIN_PAIRS = 20
MAX_IMAGE_RESIDUAL = .05          # rad, median, over pairs turning ~0.1 rad


def read_recording(src):
    """Frame times (s), ARKit's camera-to-world (p, q_xyzw) and intrinsics (fx fy cx cy) per frame, and the IMU
    (t, gyro rad/s, accel in g as CoreMotion gives it), sorted by time (CoreMotion delivers a sample late now and then:
    1 of 2343 in 214574fd05, by 2.5 ms), one sample per time."""
    odometry = np.genfromtxt(src / 'odometry.csv', delimiter=',', skip_header=1, usecols=range(13))
    imu = np.loadtxt(src / 'imu.csv', delimiter=',', skiprows=1)
    if np.any(np.diff(odometry[:, 0]) <= 0):
        raise SystemExit('the times of odometry.csv do not increase')
    if not np.array_equal(odometry[:, 1], np.arange(len(odometry))):
        raise SystemExit('odometry.csv does not number its frames 0, 1, 2, ...')
    late = int(np.sum(np.diff(imu[:, 0]) <= 0))
    imu = imu[np.argsort(imu[:, 0], kind='stable')]
    imu = imu[np.r_[True, np.diff(imu[:, 0]) > 0]]
    return dict(t=odometry[:, 0], p=odometry[:, 2:5], q=odometry[:, 5:9], K=odometry[:, 9:13],
                imu_t=imu[:, 0], gyro=imu[:, 4:7], accel=imu[:, 1:4], imu_out_of_order=late)


def camera_rotations(q_xyzw, turn=90):
    """Stray Scanner's camera orientations as the turned frame's camera to ARKit's world (y up)."""
    return Rotation.from_quat(q_xyzw).as_matrix() @ R_TURNED[turn]


def turned_intrinsics(K, width, height, turn=90):
    """fx fy cx cy of a width x height frame, scaled to WIDTH and turned clockwise."""
    s = WIDTH / width
    fx, fy, cx, cy = K[0] * s, K[1] * s, (K[2] + .5) * s - .5, (K[3] + .5) * s - .5
    if turn == 0:
        return np.array([fx, fy, cx, cy])
    if turn == 180:
        return np.array([fx, fy, WIDTH - 1 - cx, round(height * s) - 1 - cy])
    return np.array([fy, fx, round(height * s) - 1 - cy, cx])


def angular_velocity(t, R_wc):
    """The camera's angular velocity in its own frame between consecutive frames, at their midpoints."""
    w = Rotation.from_matrix(np.einsum('nji,njk->nik', R_wc[:-1], R_wc[1:])).as_rotvec() / np.diff(t)[:, None]
    return (t[:-1] + t[1:]) / 2, w


def fit_rotation(t, w_cam, imu_t, gyro, offsets=np.arange(-.1, .1001, .001)):
    """The rotation R (gyro ~ R w_cam) and the shift dt (the IMU's time of a frame's motion: its own time + dt) that fit
    best, with the median residual (rad/s)."""
    best = None
    for dt in offsets:
        inside = (t + dt > imu_t[0]) & (t + dt < imu_t[-1])
        g = np.column_stack([np.interp(t[inside] + dt, imu_t, gyro[:, k]) for k in range(3)])
        U, _, Vt = np.linalg.svd(g.T @ w_cam[inside])
        R = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
        residual = float(np.median(np.linalg.norm(g - w_cam[inside] @ R.T, axis=1)))
        if best is None or residual < best[2]:
            best = (R, float(dt), residual)
    return best


def gyro_rotation(t0, t1, imu_t, gyro, steps=50):
    """The IMU's rotation from t0 to t1 (R_I0_I1), the gyroscope interpolated and integrated."""
    s = np.linspace(t0, t1, steps)
    w = np.column_stack([np.interp(s, imu_t, gyro[:, k]) for k in range(3)])
    R = np.eye(3)
    for k in range(steps - 1):
        R = R @ Rotation.from_rotvec(w[k] * (s[k + 1] - s[k])).as_matrix()
    return R


def image_rotation_fit(images, t, imu_t, gyro, K, cv2):
    """The camera-IMU rotation the images give, as fit_rotation's but from the frames: each pair's rotation from its
    essential matrix (ORB matches, RANSAC) against the gyroscope integrated over the pair. ``images(i)`` is frame i,
    grey. Returns (R, median residual in rad, pairs)."""
    orb, matcher = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    camera, imu = [], []
    for i in range(0, len(t) - PAIR_GAP, PAIR_STEP):
        j = i + PAIR_GAP
        if t[i] < imu_t[0] or t[j] > imu_t[-1]:
            continue
        rv_imu = Rotation.from_matrix(gyro_rotation(t[i], t[j], imu_t, gyro)).as_rotvec()
        if np.linalg.norm(rv_imu) < MIN_PAIR_ROTATION:
            continue
        (ka, da), (kb, db) = (orb.detectAndCompute(images(k), None) for k in (i, j))
        if da is None or db is None:
            continue
        matches = matcher.match(da, db)
        if len(matches) < 50:
            continue
        pa = np.float32([ka[m.queryIdx].pt for m in matches])
        pb = np.float32([kb[m.trainIdx].pt for m in matches])
        E, inliers = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, .999, 1.)
        if E is None or E.shape != (3, 3):
            continue
        n, R_ba, _, _ = cv2.recoverPose(E, pa, pb, K, mask=inliers)      # x_b = R_ba x_a + t
        if n >= 40:
            camera.append(Rotation.from_matrix(R_ba.T).as_rotvec())
            imu.append(rv_imu)
    if len(camera) < MIN_PAIRS:
        return None, None, len(camera)
    camera, imu = np.asarray(camera), np.asarray(imu)
    U, _, Vt = np.linalg.svd(imu.T @ camera)
    R = U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt
    return R, float(np.median(np.linalg.norm(imu - camera @ R.T, axis=1))), len(camera)


def rotation_deg(A, B):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(A.T @ B).as_rotvec())))


def upright(image, cv2, turn=90):
    """A recorded frame scaled to WIDTH and turned clockwise."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    return small if turn == 0 else cv2.rotate(small, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}[turn])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(rec, rig, size, rig_name='config/iphone', turn=90):
    """The recording against the rig: the numbers conversion.json keeps; a SystemExit where one is out of bounds."""
    R_imu_cam = np.asarray(rig['T_imu_cam'], float)[:3, :3]
    R_wc = camera_rotations(rec['q'], turn)
    tm, w_cam = angular_velocity(rec['t'], R_wc)
    R_fit, dt, residual = fit_rotation(tm, w_cam, rec['imu_t'], rec['gyro'])
    up_imu = np.einsum('nji,j->ni', R_wc @ R_imu_cam.T, [0., 1., 0.])        # ARKit's up, in the IMU frame
    accel = np.column_stack([np.interp(rec['t'] + dt, rec['imu_t'], rec['accel'][:, k]) for k in range(3)])
    gravity_cos = float(np.median(np.sum(up_imu * -accel, 1) / np.linalg.norm(accel, axis=1)))
    down_in_camera = np.einsum('nji,j->ni', R_wc, [0., -1., 0.])             # gravity, in the turned camera
    upright_share = float(np.mean(down_in_camera[:, 1] > np.cos(np.radians(45))))
    K_rig = np.asarray(rig['intrinsics'], float)
    K = np.array([turned_intrinsics(k, *size, turn) for k in rec['K']])
    focal_error = float(np.max(np.abs(K[:, :2] / K_rig[:2] - 1)))
    centre_error = float(np.max(np.abs(K[:, 2:] - K_rig[2:])))
    out = dict(rotation_to_rig_deg=round(rotation_deg(R_fit, R_imu_cam), 3), gyro_residual_rad_s=round(residual, 4),
               imu_shift_s=round(dt, 3), gravity_cos=round(gravity_cos, 4), upright_share=round(upright_share, 3),
               focal_error=round(focal_error, 4), centre_error_px=round(centre_error, 2),
               intrinsics_median=[round(float(v), 3) for v in np.median(K, 0)])
    failures = [name for name, bad in (
        (f'the gyroscope fits a rotation {out["rotation_to_rig_deg"]} deg from the rig\'s', out['rotation_to_rig_deg'] > MAX_ROTATION_DEG),
        (f'the gyroscope does not follow ARKit (median residual {residual:.3f} rad/s)', residual > MAX_GYRO_RESIDUAL),
        (f'the accelerometer does not point against ARKit\'s gravity (cosine {gravity_cos:.3f})', gravity_cos < MIN_GRAVITY_COS),
        (f'the frames turned {turn} degrees are upright for {upright_share:.0%} of them', upright_share < MIN_UPRIGHT),
        (f'ARKit\'s focal length is {focal_error:.1%} from the rig\'s', focal_error > MAX_FOCAL_ERROR),
        (f'ARKit\'s principal point is {centre_error:.1f} px from the rig\'s', centre_error > MAX_CENTRE_ERROR_PX)) if bad]
    if failures:
        raise SystemExit(f'not converted, the recording is not of the rig ({rig_name}): ' + '; '.join(failures))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src', type=Path, help="Stray Scanner's folder of the recording")
    p.add_argument('--out', type=Path, help='default: data/iphone/<the folder\'s name>')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--rig', type=Path, default=RIG.parent.relative_to(ROOT), help='the rig directory (default config/iphone)')
    p.add_argument('--turn', type=int, choices=tuple(R_TURNED), default=90,
                   help='degrees clockwise: 90 for a phone held in portrait, 180 in landscape with the image upside down, '
                        '0 in landscape the other way')
    p.add_argument('--max-image-rotation', type=float, default=MAX_ROTATION_DEG,
                   help='the limit in degrees on the rotation the images fit against the rig; raising it is a decision to report, '
                        'for a rig already checked on other recordings (the rig is rigid, the fit is what varies)')
    a = p.parse_args(argv)
    import cv2

    out = a.out or ROOT / 'data/iphone' / a.src.name
    if out.exists():
        raise SystemExit(f'{out} exists; remove it or give another --out')
    rig = yaml.safe_load((ROOT / a.rig / 'kalibr_imucam_chain.yaml').read_text().replace('%YAML:1.0', ''))['cam0']
    rec = read_recording(a.src)
    # The first frame's packet is before the video's start (time -1/60 s), hidden by the file's edit list, so decoders
    # skip it; read with the list ignored, there is one frame per row of odometry.csv, the others unchanged.
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'ignore_editlist;1'
    video = cv2.VideoCapture(str(a.src / 'rgb.mp4'))
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size[0] * 3 != size[1] * 4:
        raise SystemExit(f'the frames are {size[0]}x{size[1]}, not 4:3 landscape')
    checks = check(rec, rig, size, str(a.rig), a.turn)
    print(json.dumps(checks))

    frames = out / 'mav0/cam0/data'
    frames.mkdir(parents=True)
    (out / 'mav0/imu0').mkdir()
    stamps = [round(t * 1e9) for t in rec['t']]
    for i, stamp in enumerate(stamps):
        ok, image = video.read()
        if not ok:
            raise SystemExit(f'rgb.mp4 ends at frame {i}; odometry.csv has {len(stamps)}')
        image = upright(image, cv2, a.turn)
        if [image.shape[1], image.shape[0]] != list(rig['resolution']):
            raise SystemExit(f'a turned frame is {image.shape[1]}x{image.shape[0]}, the rig {rig["resolution"]}')
        cv2.imwrite(str(frames / f'{stamp}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
    if video.read()[0]:
        raise SystemExit(f'rgb.mp4 has more frames than odometry.csv ({len(stamps)})')
    fx, fy, cx, cy = rig['intrinsics']
    R_images, residual, pairs = image_rotation_fit(
        lambda i: cv2.imread(str(frames / f'{stamps[i]}.jpg'), cv2.IMREAD_GRAYSCALE), rec['t'], rec['imu_t'],
        rec['gyro'], np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), cv2)
    R_rig = np.asarray(rig['T_imu_cam'], float)[:3, :3]
    if R_images is None or rotation_deg(R_images, R_rig) > a.max_image_rotation or residual > MAX_IMAGE_RESIDUAL:
        shutil.rmtree(out)
        raise SystemExit(f'not converted: the images do not turn with the gyroscope through the rig ({pairs} pairs'
                         + ('' if R_images is None else f'; the images fit a rotation {rotation_deg(R_images, R_rig):.1f} '
                            f'deg from the rig\'s, {np.round(R_images, 3).tolist()}, residual {residual:.3f} rad') + ')')
    checks.update(images_rotation_to_rig_deg=round(rotation_deg(R_images, R_rig), 3),
                  images_residual_rad=round(residual, 4), image_pairs=pairs, max_image_rotation_deg=a.max_image_rotation)

    with (out / 'mav0/cam0/data.csv').open('w') as f:
        f.write('#timestamp [ns],filename\n')
        f.writelines(f'{t},{t}.jpg\n' for t in stamps)
    with (out / 'mav0/imu0/data.csv').open('w') as f:
        f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        f.writelines(f'{round(t * 1e9)},' + ','.join(f'{x:.9g}' for x in (*w, *(-G * acc))) + '\n'
                     for t, w, acc in zip(rec['imu_t'], rec['gyro'], rec['accel']))
    # ARKit's trajectory of the IMU, by the rig's rotation; the lever arm is not known, so its position is the camera's.
    R_wi = camera_rotations(rec['q'], a.turn) @ np.asarray(rig['T_imu_cam'], float)[:3, :3].T
    q_wi = Rotation.from_matrix(R_wi).as_quat()
    (out / 'arkit.tum').write_text('# t x y z qx qy qz qw: ARKit\'s own estimate (visual-inertial, LiDAR-aided), not a '
                                   'reference; the IMU\'s orientation, the camera\'s position\n' + ''.join(
        f'{t:.9f} ' + ' '.join(f'{v:.7f}' for v in (*p, *q)) + '\n' for t, p, q in zip(rec['t'], rec['p'], q_wi)))
    seconds = lambda t: float(t[-1] - t[0])    # noqa: E731
    record = dict(source=str(a.src), rig=str(a.rig), sha256={n: sha256(a.src / n) for n in ('rgb.mp4', 'odometry.csv', 'imu.csv')},
                  frames=len(stamps), recorded_size=list(size), resolution=list(rig['resolution']),
                  turned=f'{a.turn} degrees clockwise', camera_rate_hz=round((len(stamps) - 1) / seconds(rec['t']), 2),
                  frame_gaps_ms=sorted({round(float(d) * 1000) for d in np.diff(rec['t'])}),
                  imu_samples=len(rec['imu_t']), imu_out_of_order=rec['imu_out_of_order'], imu_rate_hz=round((len(rec['imu_t']) - 1) / seconds(rec['imu_t']), 1),
                  duration_s=round(seconds(rec['t']), 2), first_frame_ns=stamps[0],
                  imu_span_ns=[round(rec['imu_t'][0] * 1e9), round(rec['imu_t'][-1] * 1e9)],
                  arkit_path_m=round(float(np.sum(np.linalg.norm(np.diff(rec['p'], axis=0), axis=1))), 2),
                  checks=checks, jpeg_quality=a.jpeg_quality)
    (out / 'conversion.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
