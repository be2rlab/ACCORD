import pytest

from accord_ros.sync import SensorBuffer


def test_image_waits_for_bracketing_imu_and_preserves_samples():
    buffer = SensorBuffer()
    buffer.push_image(20, 'frame')
    buffer.push_imu(10, (1, 2, 3), (4, 5, 6))
    assert buffer.pop_ready() is None

    buffer.push_imu(20, (7, 8, 9), (10, 11, 12))
    stamp, image, samples = buffer.pop_ready()
    assert (stamp, image) == (20, 'frame')
    assert [sample[0] for sample in samples] == pytest.approx([10e-9, 20e-9])
    assert buffer.pop_ready() is None


def test_each_image_receives_only_unconsumed_imu_samples():
    buffer = SensorBuffer()
    buffer.push_image(15, 'first')
    buffer.push_image(25, 'second')
    for stamp in (10, 20, 30):
        buffer.push_imu(stamp, (0, 0, 0), (0, 0, 0))

    first = buffer.pop_ready()
    second = buffer.pop_ready()
    assert [sample[0] for sample in first[2]] == pytest.approx([10e-9, 20e-9])
    assert [sample[0] for sample in second[2]] == pytest.approx([30e-9])


def test_sensor_timestamps_must_increase_and_buffers_are_bounded():
    buffer = SensorBuffer(max_images=1, max_imu=1)
    buffer.push_image(1, 'frame')
    with pytest.raises(BufferError):
        buffer.push_image(2, 'overflow')
    with pytest.raises(ValueError, match='increase'):
        buffer.push_image(1, 'out of order')

    buffer.push_imu(1, (0, 0, 0), (0, 0, 0))
    with pytest.raises(BufferError):
        buffer.push_imu(2, (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match='increase'):
        buffer.push_imu(1, (0, 0, 0), (0, 0, 0))
