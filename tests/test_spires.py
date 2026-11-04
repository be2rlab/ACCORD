"""Oxford Spires: the rig from the release's calibration, the conversion, the reference carried from the base frame to
the IMU, the laser scan's reader and crop, and the fetch plan."""
import hashlib
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from accord.datasets import Spires, open_dataset, pose_matrix, rig_camera
from accord.filter import read_camera, read_yaml

T0 = 1_710_755_132


def release_calibration(root):
    source = root / 'config/spires/source'
    cam = yaml.safe_load((source / 'cam0.yaml').read_text())
    chain = yaml.safe_load((source / 'cam-lidar-imu.yaml').read_text())['cam0']
    return cam, chain, yaml.safe_load((source / 'imu.yaml').read_text())


def test_the_rig_is_cam0_of_the_release_with_its_camera_from_imu_transform_inverted(root, cfg):
    cam, chain, imu = release_calibration(root)
    T_cam_imu = pose_matrix([*chain['C_r_CI'], *chain['C_q_CI']])
    rig = read_camera(root / cfg['rigs']['spires'])
    np.testing.assert_allclose(rig['T_imu_cam'], np.linalg.inv(T_cam_imu), atol=1e-12)
    K = np.asarray(cam['camera_matrix']['data']).reshape(3, 3)
    assert rig['intrinsics'] == [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    assert rig['distortion_coeffs'] == cam['distortion_coefficients']['data'] and rig['distortion_model'] == 'equidistant'
    assert tuple(rig['resolution']) == (1440, 1080) and rig['timeshift_cam_imu'] == 0.
    noise = read_yaml(root / 'config/spires/kalibr_imu_chain.yaml')['imu0']
    for key in ('accelerometer_noise_density', 'accelerometer_random_walk', 'gyroscope_noise_density', 'gyroscope_random_walk'):
        assert noise[key] == imu[key]
    estimator = read_yaml(root / cfg['rigs']['spires'])
    assert estimator['track_frequency'] > 20. and estimator['calib_cam_timeoffset'] is False
    assert 'spires' not in cfg['standstill']['rigs']


def test_the_quaternions_of_the_release_are_x_y_z_w_of_the_matrix_beside_them(root):
    _cam, chain, _imu = release_calibration(root)
    np.testing.assert_allclose(pose_matrix([*chain['C_r_CL'], *chain['C_q_CL']]), np.asarray(chain['T_cam_lidar']),
                               atol=1e-12)


def test_the_base_frame_is_where_the_release_says_it_is(root):
    """calibration/README.md of the release: the base frame has x forward, as the front camera looks, and z up."""
    _camera, T_imu_cam = rig_camera('spires')
    T_imu_base = Spires.imu_from_base(T_imu_cam)
    R_base_cam = np.linalg.inv(T_imu_base)[:3, :3] @ T_imu_cam[:3, :3]
    assert R_base_cam[:, 2] @ [1., 0., 0.] > .99                # the optical axis is the base's x
    assert R_base_cam[:, 1] @ [0., 0., -1.] > .99               # image down is the base's -z
    _cam, chain, _imu = release_calibration(root)
    lidar = np.asarray(chain['T_cam_lidar'], float)
    np.testing.assert_allclose(T_imu_base, T_imu_cam @ lidar @ np.linalg.inv(pose_matrix(Spires.BASE_LIDAR_T_XYZ_Q_XYZW)))


def jpeg(value, shape=(6, 8)):
    ok, data = cv2.imencode('.jpg', np.full((*shape, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def release_sequence(root, name='cc03', frames=8, seconds=1.):
    """A sequence as scripts/fetch_spires.py leaves it: images.zip with three cameras, imu.csv, gt/gt-tum.txt."""
    seq = root / name
    seq.mkdir(parents=True)
    stamps = [(T0 + i // 20, (i % 20) * 50_000_000 + 7) for i in range(frames)]
    with zipfile.ZipFile(seq / 'images.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for camera in ('cam0', 'cam1', 'cam2'):
            archive.writestr(f'{camera}/', b'')
            for k, (s, ns) in enumerate(stamps):
                archive.writestr(f'{camera}/{s}.{ns:09d}.jpg', jpeg(10 * k + (camera == 'cam0') * 100))
    rows = ['secs,nsecs,acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z']
    for i in range(int(seconds * 400) + 8):
        ns = i * 2_500_000
        rows.append(f'{T0 + ns // 10**9},{ns % 10**9},0.1{i % 10},0.2,9.81,1.{i % 10},2.5,-3.0')
    (seq / 'imu.csv').write_text('\n'.join(rows) + '\n')
    return seq, [s * 10**9 + ns for s, ns in stamps]


def test_frame_names_are_seconds_and_nine_digits_of_nanoseconds(script):
    convert = script('convert_spires')
    assert convert.frame_stamp('1710927121.688489629.jpg') == 1710927121688489629
    assert convert.frame_stamp('1710927121.000000007.jpg') == 1710927121000000007
    for name in ('1710927121.68848962.jpg', '1710927121.jpg', 'a.000000001.jpg', '1710927121.688489629.png'):
        with pytest.raises(ValueError):
            convert.frame_stamp(name)


def test_the_conversion_keeps_cam0_byte_for_byte_and_puts_the_gyroscope_first(tmp_path, script):
    seq, stamps = release_sequence(tmp_path)
    report = script('convert_spires').convert(seq, log=lambda *a, **k: None)
    out = seq / 'accord'
    assert sorted(int(p.stem) for p in (out / 'cam0/data').glob('*.jpg')) == stamps
    with zipfile.ZipFile(seq / 'images.zip') as archive:
        s, ns = divmod(stamps[3], 10**9)
        assert (out / f'cam0/data/{stamps[3]}.jpg').read_bytes() == archive.read(f'cam0/{s}.{ns:09d}.jpg')
    lines = (out / 'imu0/data.csv').read_text().splitlines()
    assert lines[0].startswith('#timestamp [ns],w_RS_S_x')
    assert lines[2].split(',') == [str(T0 * 10**9 + 2_500_000), '1.1', '2.5', '-3.0', '0.11', '0.2', '9.81']
    assert report['images']['count'] == 8 and report['image_size'] == [8, 6] and report['imu']['count'] == 408
    assert json.loads((out / 'conversion.json').read_text())['camera'] == 'cam0'
    with pytest.raises(SystemExit):
        script('convert_spires').convert(seq)                   # never repeated in place
    ds = open_dataset('spires', tmp_path, 'cc03')
    assert isinstance(ds, Spires) and ds.image_stamps().tolist() == stamps and len(ds.imu()) == 408
    np.testing.assert_allclose(ds.imu()[1].gyro, [1.1, 2.5, -3.]) and np.testing.assert_allclose(ds.imu()[1].accel, [.11, .2, 9.81])
    gray, colour = ds.load_filter_image(stamps[0]), ds.load_color_image(stamps[0])
    assert gray.ndim == 2 and gray.dtype == np.uint8 and colour.shape == (6, 8, 3)
    assert ds.camera.model == 'equidistant' and ds.camera.resolution == (1440, 1080)


def test_the_imu_columns_are_read_by_name(tmp_path, script):
    rows = script('convert_spires').imu_rows(io.StringIO('nsecs,secs,ang_vel_x,ang_vel_y,ang_vel_z,acc_x,acc_y,acc_z\n'
                                                         '5,2,1,2,3,4,5,6\n'))
    assert list(rows) == [(2_000_000_005, ['1', '2', '3', '4', '5', '6'])]
    with pytest.raises(ValueError):
        list(script('convert_spires').imu_rows(io.StringIO('secs,nsecs,acc_x\n1,2,3\n')))


def imu_walk(seconds=4., rate=20.):
    """World from IMU: a walk along x that turns about z and pitches a little."""
    t = T0 + .1 + np.arange(0., seconds, 1. / rate)
    p = np.stack([t - t[0], np.sin(t - t[0]), np.full_like(t, 1.5)], axis=1)
    R = Rotation.from_euler('zy', np.stack([.4 * (t - t[0]), .1 * np.sin(t - t[0])], axis=1)).as_matrix()
    return t, p, R


def test_the_reference_is_the_base_frame_carried_to_the_imu(tmp_path, script):
    seq, _stamps = release_sequence(tmp_path, seconds=5.)
    script('convert_spires').convert(seq, log=lambda *a, **k: None)
    ds = Spires(tmp_path, 'cc03')
    t, p_wi, R_wi = imu_walk()
    T_wi = np.repeat(np.eye(4)[None], len(t), axis=0)
    T_wi[:, :3, :3], T_wi[:, :3, 3] = R_wi, p_wi
    T_wb = T_wi @ ds.T_imu_base                                  # world from base = world from IMU . IMU from base
    q = Rotation.from_matrix(T_wb[:, :3, :3]).as_quat()
    (seq / 'gt').mkdir()
    (seq / 'gt/gt-tum.txt').write_text(''.join(f'{a:.9f} {b[0]} {b[1]} {b[2]} {c[0]} {c[1]} {c[2]} {c[3]}\n'
                                               for a, b, c in zip(t, T_wb[:, :3, 3], q)))
    reference = ds.groundtruth()
    assert np.median(np.diff(reference.t)) == pytest.approx(.01) and reference.orientation_reliable and ds.dense
    for k in (0, 17, 40):
        j = int(np.argmin(np.abs(reference.t - t[k])))
        assert abs(reference.t[j] - t[k]) < 1e-5                # the 100 Hz grid, at a stamp of 1.7e9 s
        np.testing.assert_allclose(reference.p[j], p_wi[k], atol=1e-4)
        np.testing.assert_allclose(reference.R[j], R_wi[k], atol=1e-4)
    assert ds.reference_provenance()['variant'].startswith('poses of the base frame')


def pcd_bytes(points, fields='Intensity rgb x y z _', size='4 4 4 4 4 1', kind='F F F F F U', count='1 1 1 1 1 4',
              data='binary'):
    record = np.dtype([('Intensity', '<f4'), ('rgb', '<f4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('_', 'u1', 4)])
    body = np.zeros(len(points), record)
    body['x'], body['y'], body['z'] = points.T
    body['Intensity'] = 7.
    head = (f'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS {fields}\nSIZE {size}\nTYPE {kind}\n'
            f'COUNT {count}\nWIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA {data}\n')
    return head.encode() + body.tobytes()


def test_the_scan_reader_takes_the_24_byte_points_of_the_release_in_chunks(tmp_path, script):
    reader = script('spires_reference')
    points = np.random.default_rng(0).uniform(-50., 50., (1001, 3)).astype(np.float32)
    (tmp_path / 'scan.pcd').write_bytes(pcd_bytes(points))
    chunks = list(reader.read_pcd_xyz(tmp_path / 'scan.pcd', chunk_points=300))
    assert [len(c) for c in chunks] == [300, 300, 300, 101]
    np.testing.assert_array_equal(np.concatenate(chunks), points.astype(float))
    (tmp_path / 'short.pcd').write_bytes(pcd_bytes(points)[:-5])
    with pytest.raises(ValueError):
        list(reader.read_pcd_xyz(tmp_path / 'short.pcd'))
    (tmp_path / 'ascii.pcd').write_bytes(pcd_bytes(points, data='ascii'))
    with pytest.raises(ValueError):
        list(reader.read_pcd_xyz(tmp_path / 'ascii.pcd'))


def room(size=8., height=4., step=.05):
    """The floor, the ceiling and four walls of a room, as scanned points."""
    a = np.arange(0., size + 1e-9, step)
    h = np.arange(0., height + 1e-9, step)
    floor = np.stack(np.meshgrid(a, a), -1).reshape(-1, 2)
    wall = np.stack(np.meshgrid(a, h), -1).reshape(-1, 2)
    parts = [np.c_[floor, np.zeros(len(floor))], np.c_[floor, np.full(len(floor), height)]]
    for x in (0., size):
        parts.append(np.c_[np.full(len(wall), x), wall])
        parts.append(np.c_[wall[:, 0], np.full(len(wall), x), wall[:, 1]])
    return np.concatenate(parts).astype(np.float32)


def test_the_scan_is_cropped_to_the_sequence_and_the_frame_checked(tmp_path, script):
    data = tmp_path / 'spires'
    seq, _stamps = release_sequence(data)
    script('convert_spires').convert(seq, log=lambda *a, **k: None)
    (data / 'scans').mkdir()
    (data / 'scans/christ-church_merged-cloud-1cm.pcd').write_bytes(pcd_bytes(room()))
    t = T0 + np.arange(0., 10., .05)
    base = np.stack([2. + .4 * (t - t[0]), np.full_like(t, 4.), np.full_like(t, 1.2)], axis=1)
    (seq / 'gt').mkdir()
    (seq / 'gt/gt-tum.txt').write_text(''.join(f'{a:.6f} {b[0]} {b[1]} {b[2]} 0 0 0 1\n' for a, b in zip(t, base)))
    out = script('spires_reference').write_reference(seq, margin=3., voxel=.05, log=lambda *a: None)
    report = json.loads((out.parent / 'reference.json').read_text())
    assert out == seq / 'accord/pointcloud0/data.ply' and 0 < report['points'] < len(room())
    assert report['frame_check']['median_m'] < 3. and report['free_space_check']['inside_surface'] == 0
    imu_height = 1.2 + np.linalg.inv(Spires.imu_from_base(rig_camera('spires')[1]))[2, 3]
    assert report['free_space_check']['height_above_floor_median_m'] == pytest.approx(imu_height, abs=.03)
    with pytest.raises(SystemExit):
        script('spires_reference').write_reference(seq, margin=3., voxel=.05)       # never rebuilt in place


def test_the_fetch_plan_names_every_file_of_a_sequence_with_its_hash(root, script):
    fetch = script('fetch_spires')
    plan = fetch.load_plan()
    assert set(plan['sequences']) == {'cc02', 'cc03', 'cc05'} and len(plan['revision']) == 40
    wanted = dict((relative, source) for source, relative in fetch.targets(plan, ['cc03']))
    assert {'cc03/images.zip', 'cc03/imu.csv', 'cc03/gt/gt-tum.txt', 'scans/christ-church_merged-cloud-1cm.pcd',
            'calibration/cam0.yaml', 'calibration/cam-lidar-imu.yaml', 'calibration/imu.yaml'} <= set(wanted)
    for source in wanted.values():
        assert plan['files'][source].get('sha256') or plan['files'][source].get('git_oid')
    for name in ('cam0.yaml', 'cam-lidar-imu.yaml', 'imu.yaml'):          # the rig's sources are the release's files
        assert fetch.verify(root / 'config/spires/source' / name, plan['files'][f'calibration/{name}']) is None


def test_the_git_blob_id_is_gits(tmp_path, script):
    (tmp_path / 'hello').write_bytes(b'hello\n')
    assert script('fetch_spires').git_oid(tmp_path / 'hello') == 'ce013625030ba8dba906f756967f9e9ca394464a'
    assert hashlib.sha1(b'blob 6\0hello\n').hexdigest() == 'ce013625030ba8dba906f756967f9e9ca394464a'


def test_spires_is_a_dataset_the_runner_takes(root, cfg, script, capsys):
    assert (root / cfg['rigs']['spires']).is_file()
    with pytest.raises(SystemExit):
        script('run').main(['--dataset', 'spires', '--data', str(root), '--sequence', 'x', '--out', str(root)])
    err = capsys.readouterr().err
    assert 'invalid choice' not in err and '--out must not exist' in err
