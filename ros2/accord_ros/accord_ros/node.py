"""ROS 2 node that feeds camera and IMU messages to ACCORD's online engine."""
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from tf2_ros import TransformBroadcaster
import yaml
from scipy.spatial.transform import Rotation

from .pointcloud import pack_xyzrgb, read_active_ply
from .sync import SensorBuffer


class AccordNode(Node):
    def __init__(self):
        super().__init__('accord')
        self.declare_parameter('accord_root', os.environ.get('ACCORD_ROOT', str(Path.cwd())))
        root = Path(self.get_parameter('accord_root').value).expanduser().resolve()
        if not (root / 'src/accord/engine.py').is_file():
            raise FileNotFoundError(f'Cannot find ACCORD Python sources under {root}')
        sys.path.insert(0, str(root / 'src'))

        self.declare_parameter('config', 'config/accord.yaml')
        self.declare_parameter('model', '')
        self.declare_parameter('rig_config', 'config/phone/estimator_config.yaml')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('mapping_enabled', True)
        self.declare_parameter('seed', 0)
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odometry_topic', '/accord/odometry')
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('max_pending_images', 10)
        self.declare_parameter('max_pending_imu', 20000)
        self.declare_parameter('pointcloud_topic', '/accord/points')
        self.declare_parameter('pointcloud_publish_period_s', 1.0)

        def in_root(value):
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        config_path = in_root(self.get_parameter('config').value)
        rig_path = in_root(self.get_parameter('rig_config').value)
        if not config_path.is_file():
            raise FileNotFoundError(f'ACCORD config does not exist: {config_path}')
        if not rig_path.is_file():
            raise FileNotFoundError(f'Rig estimator config does not exist: {rig_path}')

        cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        cfg['seed'] = int(self.get_parameter('seed').value)
        cfg['mapping']['enabled'] = bool(self.get_parameter('mapping_enabled').value)
        if self.get_parameter('model').value:
            cfg['model'] = str(self.get_parameter('model').value)
        if cfg['mapping']['enabled']:
            model_name = str(cfg['model'])
            model = Path(model_name).expanduser()
            candidate = model if model.is_absolute() else root / model
            if candidate.is_dir():
                cfg['model'] = str(candidate.resolve())
            elif model_name.count('/') != 1 or model_name.startswith(('/', '.')):
                raise FileNotFoundError(f'Depth model directory does not exist: {candidate}')

        output = self.get_parameter('output_dir').value
        if not output:
            output = root / 'runs' / f"ros2_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        else:
            output = in_root(output)

        pointcloud_period = float(self.get_parameter('pointcloud_publish_period_s').value)
        if pointcloud_period <= 0:
            raise ValueError('pointcloud_publish_period_s must be positive')

        from accord.engine import Engine
        from accord.filter import read_camera

        self.camera_resolution = tuple(int(value) for value in read_camera(rig_path)['resolution'])
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.buffer = SensorBuffer(
            self.get_parameter('max_pending_images').value,
            self.get_parameter('max_pending_imu').value,
        )
        self.bridge = CvBridge()
        self.odometry_pub = self.create_publisher(
            Odometry, str(self.get_parameter('odometry_topic').value), 10
        )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter('pointcloud_topic').value), qos_profile_sensor_data
        )
        self.pointcloud_topic = str(self.get_parameter('pointcloud_topic').value)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.active_map_path = Path(output) / 'map' / 'active.ply'
        self.last_pointcloud_mtime = None
        self.last_pointcloud = None
        self.engine = Engine(
            cfg,
            rig_path,
            output,
            meta=dict(dataset='ros2', sequence=Path(output).name, data_root=str(root)),
            pose_callback=self.publish_pose,
        )
        self.failure = None
        self.closed = False
        self.vision_error_reported = False
        self.report_vision_failure()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        imu_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.image_sub = self.create_subscription(Image, self.image_topic, self.on_image, image_qos)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.on_imu, imu_qos)
        self.timer = self.create_timer(0.005, self.process_ready)
        self.pointcloud_timer = None
        if cfg['mapping']['enabled']:
            self.pointcloud_timer = self.create_timer(pointcloud_period, self.publish_pointcloud)
        else:
            self.get_logger().info('PointCloud2 publishing is inactive because mapping_enabled is false')
        self.get_logger().info(
            f'ACCORD ready: image={self.image_topic}, imu={self.imu_topic}, '
            f'output={output}, mapping={cfg["mapping"]["enabled"]}'
        )

    @staticmethod
    def stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_image(self, msg):
        try:
            self.buffer.push_image(self.stamp_ns(msg.header.stamp), (msg, time.monotonic()))
        except (BufferError, ValueError) as exc:
            self.fail(f'Invalid camera stream: {exc}')

    def on_imu(self, msg):
        try:
            self.buffer.push_imu(
                self.stamp_ns(msg.header.stamp),
                (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
                (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z),
            )
        except (BufferError, ValueError) as exc:
            self.fail(f'Invalid IMU stream: {exc}')

    def process_ready(self):
        if self.failure is not None or self.closed:
            return
        try:
            while True:
                item = self.buffer.pop_ready()
                if item is None:
                    return
                stamp_ns, (msg, arrival_wall), imu = item
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                if (image.shape[1], image.shape[0]) != self.camera_resolution:
                    raise ValueError(
                        f'Image resolution {image.shape[1]}x{image.shape[0]} does not match rig '
                        f'{self.camera_resolution[0]}x{self.camera_resolution[1]}'
                    )
                self.engine.step(dict(
                    t=stamp_ns * 1e-9,
                    image=image,
                    color=self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'),
                    imu=imu,
                    arrival_wall=arrival_wall,
                ))
                self.report_vision_failure()
        except Exception as exc:
            self.fail(f'ACCORD processing failed: {type(exc).__name__}: {exc}', exception=True)

    def report_vision_failure(self):
        if self.engine.vision_failed and not self.vision_error_reported:
            self.get_logger().error('ACCORD mapping stopped; check events.jsonl in the run directory for details')
            self.vision_error_reported = True

    def publish_pose(self, t, _body, _camera, map_body, map_camera):
        stamp_ns = int(round(t * 1e9))
        stamp = TimeMsg(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        pose = Odometry()
        pose.header.stamp = stamp
        pose.header.frame_id = self.world_frame
        pose.child_frame_id = self.base_frame
        self.set_pose(pose.pose.pose, map_body)
        pose.twist.covariance[0] = -1.0
        self.odometry_pub.publish(pose)

        transforms = []
        world_to_base = TransformStamped()
        world_to_base.header.stamp = stamp
        world_to_base.header.frame_id = self.world_frame
        world_to_base.child_frame_id = self.base_frame
        self.set_transform(world_to_base.transform, map_body)
        transforms.append(world_to_base)

        base_to_camera = np.linalg.inv(map_body) @ map_camera
        body_camera = TransformStamped()
        body_camera.header.stamp = stamp
        body_camera.header.frame_id = self.base_frame
        body_camera.child_frame_id = self.camera_frame
        self.set_transform(body_camera.transform, base_to_camera)
        transforms.append(body_camera)
        self.tf_broadcaster.sendTransform(transforms)

    def publish_pointcloud(self):
        try:
            mtime = self.active_map_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self.last_pointcloud_mtime and self.last_pointcloud is not None:
            self.last_pointcloud.header.stamp = self.get_clock().now().to_msg()
            self.pointcloud_pub.publish(self.last_pointcloud)
            return
        try:
            points, colors = read_active_ply(self.active_map_path)
            packed = pack_xyzrgb(points, colors)
        except (OSError, ValueError) as exc:
            self.get_logger().error(f'Could not publish ACCORD active map: {exc}')
            return

        cloud = PointCloud2()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = self.world_frame
        cloud.height = 1
        cloud.width = len(packed)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = packed.dtype.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = bool(np.isfinite(points).all())
        cloud.data = packed.tobytes()
        self.pointcloud_pub.publish(cloud)
        self.last_pointcloud_mtime = mtime
        self.last_pointcloud = cloud
        self.get_logger().info(f'Published {cloud.width} points on {self.pointcloud_topic}')

    @staticmethod
    def set_pose(target, matrix):
        target.position.x, target.position.y, target.position.z = matrix[:3, 3]
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = q

    @staticmethod
    def set_transform(target, matrix):
        target.translation.x, target.translation.y, target.translation.z = matrix[:3, 3]
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        target.rotation.x, target.rotation.y, target.rotation.z, target.rotation.w = q

    def fail(self, message, exception=False):
        if self.failure is not None:
            return
        self.failure = message
        if exception:
            self.get_logger().error(message)
        else:
            self.get_logger().fatal(message)
        self.shutdown(error=message)
        rclpy.try_shutdown()

    def shutdown(self, error=None):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'engine'):
            self.engine.close(error=error)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AccordNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
