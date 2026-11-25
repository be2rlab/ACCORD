from types import SimpleNamespace

import numpy as np
import pytest
from cv_bridge import CvBridge

from accord_ros.node import AccordNode
from accord_ros.sync import SensorBuffer


@pytest.mark.parametrize(
    ('encoding', 'pixels', 'expected_rgb'),
    [
        ('rgb8', [[[255, 0, 0], [0, 255, 0]]], [[[255, 0, 0], [0, 255, 0]]]),
        ('bgr8', [[[255, 0, 0], [0, 0, 255]]], [[[0, 0, 255], [255, 0, 0]]]),
        ('mono8', [[30, 200]], [[[30, 30, 30], [200, 200, 200]]]),
    ],
)
def test_process_ready_passes_grayscale_and_rgb_from_the_same_frame(encoding, pixels, expected_rgb):
    bridge = CvBridge()
    message = bridge.cv2_to_imgmsg(np.array(pixels, dtype=np.uint8), encoding=encoding)
    buffer = SensorBuffer()
    buffer.push_image(1_000_000_000, (message, 42.0))
    buffer.push_imu(1_000_000_001, (0, 0, 0), (0, 0, 0))
    packets = []
    node = SimpleNamespace(
        failure=None, closed=False, buffer=buffer, bridge=bridge,
        camera_resolution=(2, 1), engine=SimpleNamespace(step=packets.append),
        report_vision_failure=lambda: None,
    )

    AccordNode.process_ready(node)

    assert len(packets) == 1
    packet = packets[0]
    assert packet['t'] == 1.0 and packet['image'].shape == (1, 2)
    assert packet['arrival_wall'] == 42.0
    np.testing.assert_array_equal(packet['image'], bridge.imgmsg_to_cv2(message, desired_encoding='mono8'))
    np.testing.assert_array_equal(packet['color'], np.array(expected_rgb, dtype=np.uint8))


def test_resolution_check_uses_raw_camera_not_rectified_view():
    bridge = CvBridge()
    message = bridge.cv2_to_imgmsg(np.zeros((480, 640), dtype=np.uint8), encoding='mono8')
    buffer = SensorBuffer()
    buffer.push_image(1_000_000_000, (message, 42.0))
    buffer.push_imu(1_000_000_001, (0, 0, 0), (0, 0, 0))
    packets = []
    node = SimpleNamespace(
        failure=None, closed=False, buffer=buffer, bridge=bridge, camera_resolution=(640, 480),
        engine=SimpleNamespace(view=SimpleNamespace(resolution=(512, 512)), step=packets.append),
        report_vision_failure=lambda: None,
    )
    AccordNode.process_ready(node)
    assert len(packets) == 1


def test_vision_failure_is_reported_once():
    errors = []
    node = SimpleNamespace(engine=SimpleNamespace(vision_failed=True), vision_error_reported=False,
                           get_logger=lambda: SimpleNamespace(error=errors.append))
    AccordNode.report_vision_failure(node)
    AccordNode.report_vision_failure(node)
    assert len(errors) == 1 and 'events.jsonl' in errors[0]
