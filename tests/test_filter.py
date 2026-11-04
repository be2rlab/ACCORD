import numpy as np
import pytest

from accord.filter import Filter, materialize, read_camera, read_yaml


def packet(t):
    return dict(t=t, image=np.zeros((4, 4), np.uint8), imu=[(t - .005, np.zeros(3), np.zeros(3)), (t + .001, np.zeros(3), np.zeros(3))])


@pytest.fixture
def rig(root):
    return root / 'config/euroc/estimator_config.yaml'


def test_the_filter_gives_no_state_until_it_is_initialized_then_the_state_of_the_frame(openvins):
    filter = Filter('estimator.yaml')
    assert filter.vio.config_path == 'estimator.yaml'
    assert filter.feed(packet(1.)) is None and filter.feed(packet(1.05)) is None
    state = filter.feed(packet(1.1))
    assert state['t'] == 1.1 and state['p'][0] == pytest.approx(.3)
    assert filter.vio.imu == pytest.approx([.995, 1.001, 1.045, 1.051, 1.095, 1.101])
    openvins.silent_after = 3
    assert filter.feed(packet(1.15)) is None            # the state of an earlier frame is not an update


def test_a_run_gets_its_own_copy_of_the_rig_with_the_settings_written_over_it(rig, tmp_path, cfg):
    estimator = materialize(rig, tmp_path / 'filter', cfg['openvins'])
    assert estimator == tmp_path / 'filter/estimator.yaml' and estimator.read_text().startswith('%YAML:1.0\n')
    written, source = read_yaml(estimator), read_yaml(rig)
    assert source['calib_cam_intrinsics'] is True and written['calib_cam_intrinsics'] is False
    assert written['init_dyn_bias_g'] == [0., 0., 0.] and written['verbosity'] == 'WARNING'
    assert written['max_clones'] == source['max_clones']
    assert not any(written[key] for key in ('record_timing_information', 'save_total_state', 'record_init_pose'))
    for key in ('relative_config_imu', 'relative_config_imucam'):
        assert written[key] == key + '.yaml'
        assert read_yaml(tmp_path / 'filter' / written[key]) == read_yaml(rig.parent / source[key])
    assert read_camera(estimator) == read_camera(rig) and read_camera(rig)['resolution'] == [752, 480]
    with pytest.raises(FileExistsError):                # a run's directory is never written twice
        materialize(rig, tmp_path / 'filter', {})


STANDSTILL = dict(window_s=.5, gyro_rad_s=.02, accel_spread_m_s2=.3, sigma_m_s=.01)


def frames(start, count, gyro=0., wobble=0., rate=40, imu_rate=400):
    """Packets of a rig at rest (or turning at ``gyro`` rad/s, its accelerometer shaken by ``wobble``)."""
    rng, out = np.random.default_rng(0), []
    for k in range(count):
        t = start + k / rate
        times = np.arange(t - 1 / rate, t + 1e-9, 1 / imu_rate)[1:]
        out.append(dict(t=t, image=np.zeros((4, 4), np.uint8),
                        imu=[(s, np.array([0., 0., gyro]) + 1e-3 * rng.standard_normal(3),
                              np.array([0., 0., 9.81]) + wobble * rng.standard_normal(3) + .01 * rng.standard_normal(3)) for s in times]))
    return out


@pytest.fixture
def estimator(rig, tmp_path, cfg):
    return materialize(rig, tmp_path / 'filter', cfg['openvins'])


def test_a_rig_at_rest_gets_an_update_at_each_frame_and_one_that_moves_none(openvins, estimator):
    filter = Filter(estimator, standstill=STANDSTILL)
    for p in frames(1., 40):                            # a second at rest: from half a second on, every frame
        filter.feed(p)
    assert len(filter.vio.zero_velocity) == filter.zero_velocity_updates >= 18
    before = filter.zero_velocity_updates
    for p in frames(2., 40, gyro=.3):                   # turning
        filter.feed(p)
    after_rest = filter.zero_velocity_updates
    for p in frames(3., 40, wobble=1.):                 # carried, shaken
        filter.feed(p)
    # The window still holds the rest when the turn begins: one or two frames late, then none.
    assert before <= after_rest <= before + 2 and filter.zero_velocity_updates == after_rest


def test_without_the_setting_no_update_at_rest_is_made(openvins):
    filter = Filter('estimator.yaml')
    for p in frames(1., 40):
        filter.feed(p)
    assert filter.vio.zero_velocity == [] and filter.zero_velocity_updates == 0


def test_at_rest_the_imu_model_gets_the_frame_s_samples_and_the_rig_s_noise(openvins, rig, tmp_path, estimator):
    filter = Filter(estimator, standstill=STANDSTILL)
    for p in frames(1., 40):
        filter.feed(p)
    imu = read_yaml(tmp_path / 'filter' / read_yaml(estimator)['relative_config_imu'])['imu0']
    calls = filter.vio.zero_velocity
    assert len(calls) == filter.zero_velocity_updates >= 18
    _, sigma_v, samples, sigma_w, sigma_a, gravity, multiplier = calls[-1]
    assert sigma_v == .01 and samples == 10                     # 400 Hz over a 40 Hz frame
    assert (sigma_w, sigma_a) == (imu['gyroscope_noise_density'], imu['accelerometer_noise_density'])
    assert gravity == read_yaml(rig)['gravity_mag'] and multiplier == read_yaml(rig)['zupt_noise_multiplier']
