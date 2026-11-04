"""The visual-inertial filter: OpenVINS, unmodified, through its binding (native/openvins_binding.cpp)."""
from collections import deque
from pathlib import Path

import numpy as np
import yaml


class Filter:
    def __init__(self, config_path, standstill=None):
        """``standstill``: the update at rest of config/accord.yaml, or None. While the IMU says the rig stands, OpenVINS's
        ZUPT measurement of the frame's IMU samples, then a zero velocity."""
        import openvins_ext
        self.vio = openvins_ext.VioManager(str(config_path))
        self.standstill, self.recent, self.zero_velocity_updates = standstill, deque(), 0
        if standstill is not None:
            estimator = read_yaml(config_path)
            imu = read_yaml(Path(config_path).parent / estimator['relative_config_imu'])['imu0']
            self.at_rest = dict(sigma_w=float(imu['gyroscope_noise_density']), sigma_a=float(imu['accelerometer_noise_density']),
                                gravity_mag=float(estimator['gravity_mag']),
                                noise_multiplier=float(estimator['zupt_noise_multiplier']))

    def feed(self, packet):
        """One camera frame and the IMU samples up to it -> the state at that frame, or None before initialization."""
        for t, gyro, accel in packet['imu']:
            self.vio.feed_imu(t, np.asarray(gyro), np.asarray(accel))
            if self.standstill is not None:
                self.recent.append((t, np.asarray(gyro, float), np.asarray(accel, float)))
        self.vio.feed_camera(packet['t'], packet['image'], 0)
        if not self.vio.initialized():
            return None
        if self.standstill is not None and self.still(packet['t']):
            t, gyro, accel = (np.array(x, float) for x in zip(*packet['imu']))
            self.vio.zero_motion_update(t, gyro, accel, sigma_v=float(self.standstill['sigma_m_s']), **self.at_rest)
            self.zero_velocity_updates += 1
        state = self.vio.get_state()
        if not all(np.isfinite(np.asarray(state[k], float)).all()
                   for k in ('t', 'q_GtoI', 'p', 'v', 'bg', 'ba', 'q_ItoC', 'p_IinC')):
            raise ValueError('OpenVINS emitted a non-finite state')
        if abs(float(state['t']) - packet['t']) > .002:      # a state of an earlier frame, not an update
            return None
        return state

    def time_offset(self):
        """The camera-IMU time offset the filter holds (s)."""
        try:
            return float(self.vio.get_state()['dt_CAMtoIMU'])
        except (AttributeError, KeyError, RuntimeError):
            return None

    def still(self, t):
        """Over the last window_s, the mean gyroscope norm and the accelerometer's spread are under their limits."""
        cfg = self.standstill
        while self.recent and self.recent[0][0] < t - float(cfg['window_s']):
            self.recent.popleft()
        if len(self.recent) < 2 or self.recent[-1][0] - self.recent[0][0] < .9 * float(cfg['window_s']):
            return False
        gyro = np.array([g for _, g, _ in self.recent])
        accel = np.array([a for _, _, a in self.recent])
        return bool(np.linalg.norm(gyro, axis=1).mean() < float(cfg['gyro_rad_s'])
                    and np.linalg.norm(accel.std(axis=0)) < float(cfg['accel_spread_m_s2']))


def read_yaml(path):
    return yaml.safe_load('\n'.join(s for s in Path(path).read_text().splitlines() if not s.startswith('%YAML')))


def read_camera(rig):
    """The cam0 entry of a rig's camera-IMU file."""
    rig = Path(rig)
    return read_yaml(rig.parent / read_yaml(rig)['relative_config_imucam'])['cam0']


class OpenCvDumper(yaml.SafeDumper):
    """cv::FileStorage rejects PyYAML's indentless sequences."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def write_yaml(path, data):
    Path(path).write_text('%YAML:1.0\n' + yaml.dump(data, Dumper=OpenCvDumper, sort_keys=False))
    return Path(path)


def materialize(rig, directory, overrides):
    """The filter's configuration for one run: the rig's estimator file with ``overrides``, its IMU and camera files
    beside it. Returns the estimator file."""
    rig, directory = Path(rig).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg = read_yaml(rig)
    cfg.update(overrides)
    if int(cfg.get('max_cameras', 1)) != 1:
        raise ValueError('One camera only')
    for key in ('relative_config_imu', 'relative_config_imucam'):
        cfg[key] = write_yaml(directory / (key + '.yaml'), read_yaml((rig.parent / cfg[key]).resolve())).name
    cfg.update(record_timing_information=False, save_total_state=False, record_init_pose=False, record_init_timing=False)
    return write_yaml(directory / 'estimator.yaml', cfg)
