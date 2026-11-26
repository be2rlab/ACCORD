import numpy as np
import pytest

from accord.mapper.fusion import write_ply
from accord_ros.node import AccordNode
from accord_ros.pointcloud import pack_xyzrgb, read_active_ply
from builtin_interfaces.msg import Time


class Publisher:
    topic_name = '/accord/points'

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Logger:
    def info(self, message):
        pass

    def error(self, message):
        raise AssertionError(message)


def test_read_active_ply_returns_xyz_and_rgb(tmp_path):
    path = tmp_path / 'active.ply'
    points = np.array([[1.25, -2.0, 3.5], [0.0, 4.0, -5.0]], dtype=np.float32)
    colors = np.array([[255, 16, 2], [3, 128, 240]], dtype=np.uint8)
    write_ply(path, points, colors)

    actual_points, actual_colors = read_active_ply(path)
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_colors, colors)


def test_pack_xyzrgb_uses_the_pointcloud2_packed_rgb_layout():
    points = np.array([[1, 2, 3]], dtype=np.float32)
    colors = np.array([[255, 16, 2]], dtype=np.uint8)

    packed = pack_xyzrgb(points, colors)

    assert packed.dtype.itemsize == 16
    np.testing.assert_array_equal([packed['x'][0], packed['y'][0], packed['z'][0]], points[0])
    assert packed['rgb'].view(np.uint32)[0] == 0x00FF1002


def test_pack_xyzrgb_rejects_mismatched_point_and_colour_arrays():
    with pytest.raises(ValueError, match='shape'):
        pack_xyzrgb(np.zeros((2, 3)), np.zeros((1, 3)))


def test_node_publishes_and_republishes_active_map_as_coloured_pointcloud2(tmp_path):
    path = tmp_path / 'active.ply'
    points = np.array([[1.25, -2.0, 3.5]], dtype=np.float32)
    colors = np.array([[255, 16, 2]], dtype=np.uint8)
    write_ply(path, points, colors)

    class Clock:
        @staticmethod
        def now():
            return type('Now', (), {'to_msg': staticmethod(lambda: Time(sec=12, nanosec=34))})()

    node = type('NodeStub', (), {})()
    node.active_map_path = path
    node.last_pointcloud_mtime = None
    node.last_pointcloud = None
    node.world_frame = 'map'
    node.pointcloud_pub = Publisher()
    node.pointcloud_topic = '/accord/points'
    node.get_clock = lambda: Clock()
    node.get_logger = lambda: Logger()

    AccordNode.publish_pointcloud(node)
    AccordNode.publish_pointcloud(node)

    assert len(node.pointcloud_pub.messages) == 2
    message = node.pointcloud_pub.messages[0]
    assert (message.header.frame_id, message.header.stamp.sec, message.header.stamp.nanosec) == ('map', 12, 34)
    assert message.width == 1 and message.height == 1 and message.point_step == 16
    assert [field.name for field in message.fields] == ['x', 'y', 'z', 'rgb']
    assert len(message.data) == 16
