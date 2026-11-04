"""EuRoC, ORI, TUM-VI, Hilti-Oxford, Oxford Spires, phone and iPhone sequences: camera frames, IMU, calibration, and the
reference trajectory. The reference is read by the scorers only; nothing online touches it.
"""
from collections import namedtuple
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[2]
ImuSample = namedtuple('ImuSample', 't gyro accel')         # s, rad/s, m/s^2


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    D: np.ndarray
    model: str              # radtan | equidistant
    resolution: tuple       # width, height


@dataclass
class Reference:
    """Body poses in the reference's world frame."""
    t: np.ndarray
    p: np.ndarray
    R: np.ndarray
    orientation_reliable: bool = True


def load_imu(path):
    """An ASL imu0/data.csv, sorted by time."""
    samples = []
    with open(path, encoding='utf-8') as handle:
        for row in csv.reader(handle):
            if row and not row[0].startswith('#'):
                samples.append(ImuSample(int(row[0]) * 1e-9, np.array([float(x) for x in row[1:4]]),
                                         np.array([float(x) for x in row[4:7]])))
    samples.sort(key=lambda s: s.t)
    return samples


def read_image(path, color=False):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'could not read image: {path}')
    if color:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[:, :, 0] if image.ndim == 3 else image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rig_camera(name):
    """The camera of a dataset's rig file (config/NAME) and its pose on the IMU (T_imu_cam)."""
    cam = yaml.safe_load((ROOT / f'config/{name}/kalibr_imucam_chain.yaml').read_text().replace('%YAML:1.0', ''))['cam0']
    fx, fy, cx, cy = cam['intrinsics']
    camera = Camera(K=np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), D=np.asarray(cam['distortion_coeffs'], float),
                    model=cam['distortion_model'], resolution=tuple(cam['resolution']))
    return camera, np.asarray(cam['T_imu_cam'], float)


class Sequence:
    extension = '.png'

    def open(self, root, frames, imu):
        self.root, self.frames = Path(root), Path(frames)
        self.stamps = np.array(sorted(int(p.stem) for p in self.frames.glob('*' + self.extension)))
        if self.stamps.size < 2:
            raise FileNotFoundError(f'no frames under {self.frames}')
        self.imu_path, self._imu, self._reference = Path(imu), None, None

    def imu(self):
        if self._imu is None:
            self._imu = load_imu(self.imu_path)
        return self._imu

    def image_stamps(self):
        return self.stamps

    def load_filter_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}{self.extension}')

    def reference_provenance(self):
        """Which reference file the scores rest on."""
        reference = self.groundtruth()
        return dict(variant=self.reference_variant, path=str(self.reference_path), sha256=sha256(self.reference_path),
                    samples=int(len(reference.t)), orientation_reliable=bool(reference.orientation_reliable))


class Euroc(Sequence):
    """EuRoC MAV: 752x480 at 20 Hz, radial-tangential distortion; Vicon or Leica reference."""
    name = 'euroc'
    # cam0 of mav0/cam0/sensor.yaml: camera to body.
    T_BS = np.array([[0.0148655, -0.99988093, 0.0041403, -0.02164015],
                     [0.99955725, 0.01496721, 0.02571553, -0.06467699],
                     [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
                     [0.0, 0.0, 0.0, 1.0]])
    # The orientation of these sequences' own reference is documented as inaccurate; OpenVINS distributes a corrected file.
    ORIENTATION_SUSPECT = ('V1_01_easy',)

    def __init__(self, root, seq, reference='dataset'):
        root = Path(root) / seq / 'mav0'
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.seq = seq
        self.camera = Camera(K=np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]]),
                             D=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]),
                             model='radtan', resolution=(752, 480))
        self.R_CtoI, self.p_IC = self.T_BS[:3, :3], self.T_BS[:3, 3]
        if reference not in ('dataset', 'openvins'):
            raise ValueError('reference must be dataset or openvins')
        self.reference_variant = reference
        self.reference_path = (ROOT / f'thirdparty/open_vins/ov_data/euroc_mav/{seq}.csv' if reference == 'openvins'
                               else root / 'state_groundtruth_estimate0/data.csv')

    def groundtruth(self):
        if self._reference is None:
            rows = []
            with self.reference_path.open() as handle:
                for row in csv.reader(handle):
                    if row and not row[0].lstrip().startswith('#'):
                        rows.append([float(x) for x in row[:8]])
            table = np.asarray(rows, float)

            def rotation(q):        # Hamilton, w first
                w, x, y, z = q / np.linalg.norm(q)
                return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            R = np.asarray([rotation(q) for q in table[:, 4:8]])
            reliable = not (self.reference_variant == 'dataset' and self.seq in self.ORIENTATION_SUSPECT)
            self._reference = Reference(t=table[:, 0] * 1e-9, p=table[:, 1:4], R=R, orientation_reliable=reliable)
        return self._reference


class Ori(Sequence):
    name = 'ori'
    extension = '.jpg'

    def __init__(self, root, seq):
        self.seq_root, self.seq = Path(root) / seq, seq
        root = self.seq_root / 'accord'
        if not (root / 'conversion.json').is_file():
            raise FileNotFoundError(f'{root} lacks conversion.json; run scripts/convert_ori.py first')
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.camera, T_ic = rig_camera('ori')
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.reference_variant = 'camera poses registered to the scan, carried to the body by the calibration'
        self.reference_path = next(self.seq_root.glob('*_gt')) / 'poses_gt.txt'

    def load_color_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}.jpg', color=True)

    def groundtruth(self):
        if self._reference is None:
            table = np.loadtxt(self.reference_path)         # t x y z qx qy qz qw, camera to world
            R_wc = Rotation.from_quat(table[:, 4:8]).as_matrix()
            R_ci, p_ci = self.R_CtoI.T, -self.R_CtoI.T @ self.p_IC
            self._reference = Reference(t=table[:, 0], p=table[:, 1:4] + np.einsum('nij,j->ni', R_wc, p_ci), R=R_wc @ R_ci)
        return self._reference


class Custom(Sequence):
    """A phone recording in ASL layout: colour JPEG frames and the IMU, no reference; the camera is config/phone's."""
    name = 'custom'
    rig = 'phone'
    extension = '.jpg'

    def __init__(self, root, seq):
        root = Path(root) / seq / 'mav0'
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.seq = seq
        self.camera, T_ic = rig_camera(self.rig)
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.reference_variant, self.reference_path = None, None

    def load_filter_image(self, stamp):
        import cv2
        return cv2.cvtColor(self.load_color_image(stamp), cv2.COLOR_RGB2GRAY)

    def load_color_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}.jpg', color=True)

    def groundtruth(self):
        raise LookupError(f'{self.seq} has no reference')


class RealSense(Custom):
    """A RealSense recording (scripts/convert_realsense.py), no reference; the camera is config/realsense's."""
    name = 'realsense'
    rig = 'realsense'


class Iphone(Custom):
    """An iPhone recording (scripts/convert_iphone.py), no reference; the camera is config/iphone's."""
    name = 'iphone'
    rig = 'iphone'


class PhoneRig(Custom):
    """The phone rig dataset: one walk recorded by phones carried together, a recording <seq>/<device>/mav0 in the EuRoC
    layout. The reference is RTAB-Map's loop-closed graph from a LiDAR iPhone carried with them, about 1 Hz, on this
    device's clock and in its IMU's orientation, keeping the LiDAR iPhone's position (the lever arms are not measured),
    and its map, which reaches about 3.5 m from the track: a map point further out than SURFACE_REACH_M has nothing to
    be scored against."""
    device = None
    DENSE_RATE_HZ = 100.
    GAP_S = 3.5             # the graph's nodes are about 1 s apart, 3.1 s at most: longer gaps are not interpolated across
    SURFACE_REACH_M = 3.

    def __init__(self, root, seq):
        seq_root = Path(root) / seq
        root = seq_root / self.device / 'mav0'
        if not root.is_dir():
            raise FileNotFoundError(f'{root}: this walk has no {self.device} recording')
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.seq = seq
        self.camera, T_ic = rig_camera(self.rig)
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.dense = True
        self.surface_path = seq_root / 'reference/map.ply'
        self.surface_reach_m = self.SURFACE_REACH_M
        self.reference_path = seq_root / f'reference/trajectory_{self.device}.tum'
        self.reference_variant = (f"RTAB-Map's loop-closed graph on a LiDAR iPhone, on the {self.device}'s clock and in "
                                  f'its IMU\'s orientation, {self.position}, interpolated to {self.DENSE_RATE_HZ:.0f} Hz')

    def groundtruth(self):
        if self._reference is None:
            self._reference = Reference(*densify(*read_poses(self.reference_path), self.DENSE_RATE_HZ, self.GAP_S))
        return self._reference


class PhoneRigAndroid(PhoneRig):
    """The Samsung Galaxy Note 9 of the phone rig (config/rig_android)."""
    name = 'rig_android'
    rig = 'rig_android'
    device = 'android'
    position = "the LiDAR iPhone's position, at which the Note 9 rode (the authors)"


class PhoneRigIphone(PhoneRig):
    """The iPhone of the phone rig (config/rig_iphone)."""
    name = 'rig_iphone'
    rig = 'rig_iphone'
    device = 'iphone'
    position = "the LiDAR iPhone's position lowered by 13.7 cm, the iPhone's height below it (the release's CHANGES.md)"


class TumVi(Sequence):
    """TUM-VI rooms (512x512 equidistant, 16-bit grey, 20 Hz; IMU 200 Hz), motion capture of the IMU."""
    name = 'tumvi'

    def __init__(self, root, seq):
        root = Path(root) / seq / 'mav0'
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.seq = seq
        self.camera, T_ic = rig_camera('tumvi')
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.reference_variant = 'motion capture of the IMU'
        self.reference_path = root / 'mocap0/data.csv'

    def load_filter_image(self, stamp):
        image = read_image(self.frames / f'{int(stamp)}.png')
        return (image >> 8).astype(np.uint8) if image.dtype == np.uint16 else image

    def groundtruth(self):
        if self._reference is None:
            table = np.loadtxt(self.reference_path, delimiter=',', comments='#', ndmin=2)
            R = Rotation.from_quat(table[:, [5, 6, 7, 4]]).as_matrix()      # the file is w x y z
            self._reference = Reference(t=table[:, 0] * 1e-9, p=table[:, 1:4], R=R)
        return self._reference


def read_poses(path):
    """Rows ``t x y z qx qy qz qw`` -> (t, p, q_xyzw); lines starting with '#' are comments."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not rows or any(len(row) != 8 for row in rows):
        raise ValueError(f'{path}: expected rows of t x y z qx qy qz qw')
    table = np.asarray(rows, float)
    if np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f'{path}: timestamps are not increasing')
    return table[:, 0], table[:, 1:4], table[:, 4:8]


def is_control_points(q_xyzw):
    """The sparse files of Hilti-Oxford carry positions only and fill the rotation with the identity."""
    return bool(np.allclose(q_xyzw, [0., 0., 0., 1.]))


def densify(t, p, q_xyzw, rate_hz, gap_s):
    """A pose track interpolated to ``rate_hz`` (linear position, spherical rotation), never across a gap
    longer than ``gap_s`` -> (t, p, R)."""
    t = np.asarray(t, float)
    grid = np.arange(t[0], t[-1], 1. / rate_hz)
    left = np.clip(np.searchsorted(t, grid, side='right') - 1, 0, len(t) - 2)
    grid = grid[t[left + 1] - t[left] <= gap_s]
    p = np.stack([np.interp(grid, t, np.asarray(p, float)[:, k]) for k in range(3)], axis=1)
    return grid, p, Slerp(t, Rotation.from_quat(np.asarray(q_xyzw, float)))(grid).as_matrix()


class Hilti(Sequence):
    """Hilti-Oxford: cam0 of the Alphasense rig (720x540 equidistant, 40 Hz) and its IMU at 400 Hz. The reference, the
    IMU in the scan's frame (<seq>/gt/), is the dense poses where they exist (exp14, exp16, exp18; 10 Hz, interpolated
    to 100 Hz), else the surveyed control points."""
    name = 'hilti'
    DENSE_RATE_HZ = 100.
    GAP_S = .25             # reference samples further apart are a gap, never interpolated across

    def __init__(self, root, seq):
        self.seq_root, self.seq = Path(root) / seq, seq
        root = self.seq_root / 'accord'
        if not (root / 'conversion.json').is_file():
            raise FileNotFoundError(f'{root} lacks conversion.json; run scripts/convert_hilti.py first')
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.camera, T_ic = rig_camera('hilti')
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        folder = self.seq_root / 'gt'
        dense = sorted(folder.glob('*_imu.txt'))
        sparse = sorted(folder.glob('*_imu_3dof.txt')) or sorted(p for p in folder.glob('*.txt') if not p.name.endswith('_imu.txt'))
        if len(dense) > 1 or len(sparse) > 1:
            raise ValueError(f'{folder}: more than one reference of a kind')
        self.control_points_path = sparse[0] if sparse else None
        self.dense = bool(dense)
        self.reference_path = dense[0] if dense else self.control_points_path
        self.reference_variant = (f'dense poses of the IMU registered to the laser scan, interpolated to {self.DENSE_RATE_HZ:.0f} Hz'
                                  if dense else 'surveyed control points of the IMU, position only')

    def control_points(self):
        """(t, p): the surveyed positions of the IMU, or None."""
        if self.control_points_path is None:
            return None
        t, p, q = read_poses(self.control_points_path)
        if not is_control_points(q):
            raise ValueError(f'{self.control_points_path}: expected position-only control points')
        return t, p

    def groundtruth(self):
        if self._reference is None:
            if self.reference_path is None:
                raise LookupError(f'{self.seq_root}/gt holds no reference; run scripts/fetch_hilti.py')
            if not self.dense:
                t, p = self.control_points()
                self._reference = Reference(t=t, p=p, R=np.tile(np.eye(3), (len(t), 1, 1)), orientation_reliable=False)
                return self._reference
            t, p, q = read_poses(self.reference_path)
            if is_control_points(q):
                raise ValueError(f'{self.reference_path}: expected poses with orientation')
            self._reference = Reference(*densify(t, p, q, self.DENSE_RATE_HZ, self.GAP_S))
        return self._reference


def pose_matrix(t_xyz_q_xyzw):
    """[x y z qx qy qz qw] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(t_xyz_q_xyzw[3:], float)).as_matrix()
    T[:3, 3] = np.asarray(t_xyz_q_xyzw[:3], float)
    return T


class Spires(Sequence):
    """Oxford Spires: the front colour fisheye of the Alphasense Core rig (1440x1080 equidistant, 20 Hz; about 5 Hz on
    CC05) and its IMU at 400 Hz. The reference, gt-tum.txt (the base frame in the scan's frame), is taken to the IMU by
    T_imu_base = T_imu_cam . T_cam_lidar . inv(T_base_lidar) and interpolated to 100 Hz."""
    name = 'spires'
    extension = '.jpg'
    DENSE_RATE_HZ = 100.
    GAP_S = .25
    # calibration/README.md of the release: "The base frame is manually set with respect to the lidar".
    BASE_LIDAR_T_XYZ_Q_XYZW = (0., 0., .124, 0., 0., 1., 0.)

    def __init__(self, root, seq):
        self.seq_root, self.seq = Path(root) / seq, seq
        root = self.seq_root / 'accord'
        if not (root / 'conversion.json').is_file():
            raise FileNotFoundError(f'{root} lacks conversion.json; run scripts/convert_spires.py first')
        self.open(root, root / 'cam0/data', root / 'imu0/data.csv')
        self.camera, T_ic = rig_camera('spires')
        self.R_CtoI, self.p_IC = T_ic[:3, :3], T_ic[:3, 3]
        self.T_imu_base = self.imu_from_base(T_ic)
        self.dense = True
        self.reference_path = self.seq_root / 'gt/gt-tum.txt'
        self.reference_variant = ('poses of the base frame registered to the laser scan, carried to the IMU by the '
                                  f'calibration, interpolated to {self.DENSE_RATE_HZ:.0f} Hz')

    @classmethod
    def imu_from_base(cls, T_imu_cam, calibration=None):
        """T_imu_base from the rig's T_imu_cam and the release's cam-lidar-imu.yaml (config/spires/source/)."""
        calibration = Path(calibration or ROOT / 'config/spires/source/cam-lidar-imu.yaml')
        T_cam_lidar = np.asarray(yaml.safe_load(calibration.read_text())['cam0']['T_cam_lidar'], float)
        return np.asarray(T_imu_cam, float) @ T_cam_lidar @ np.linalg.inv(pose_matrix(cls.BASE_LIDAR_T_XYZ_Q_XYZW))

    def load_filter_image(self, stamp):
        import cv2
        image = cv2.imread(str(self.frames / f'{int(stamp)}.jpg'), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'could not read image: {self.frames / f"{int(stamp)}.jpg"}')
        return image

    def load_color_image(self, stamp):
        return read_image(self.frames / f'{int(stamp)}.jpg', color=True)

    def groundtruth(self):
        if self._reference is None:
            t, p, q = read_poses(self.reference_path)               # world from base
            T_base_imu = np.linalg.inv(self.T_imu_base)
            R_wb = Rotation.from_quat(q).as_matrix()
            p_wi = p + R_wb @ T_base_imu[:3, 3]
            R_wi = R_wb @ T_base_imu[:3, :3]
            self._reference = Reference(*densify(t, p_wi, Rotation.from_matrix(R_wi).as_quat(), self.DENSE_RATE_HZ,
                                                 self.GAP_S))
        return self._reference


def open_dataset(name, root, seq, reference=None):
    """``reference`` (EuRoC only): 'dataset', the sequence's own file, or 'openvins', the corrected
    file OpenVINS distributes, which is the default."""
    if name == 'euroc':
        return Euroc(root, seq, reference or 'openvins')
    if name == 'ori':
        return Ori(root, seq)
    if name == 'custom':
        return Custom(root, seq)
    if name == 'tumvi':
        return TumVi(root, seq)
    if name == 'hilti':
        return Hilti(root, seq)
    if name == 'realsense':
        return RealSense(root, seq)
    if name == 'spires':
        return Spires(root, seq)
    if name == 'iphone':
        return Iphone(root, seq)
    if name == 'rig_android':
        return PhoneRigAndroid(root, seq)
    if name == 'rig_iphone':
        return PhoneRigIphone(root, seq)
    raise ValueError('dataset must be euroc, ori, custom, tumvi, hilti, realsense, spires, iphone, rig_android or rig_iphone')
