"""Timestamp pairing for a camera stream and its bracketing IMU samples."""
from collections import deque


class SensorBuffer:
    """Keep ordered sensor data until an image has IMU coverage through its timestamp."""

    def __init__(self, max_images=10, max_imu=20000):
        self.max_images, self.max_imu = int(max_images), int(max_imu)
        if self.max_images < 1 or self.max_imu < 1:
            raise ValueError('Sensor buffer limits must be positive')
        self.images = deque()
        self.imu = deque()
        self.last_image_stamp = None
        self.last_imu_stamp = None

    def push_image(self, stamp_ns, image):
        stamp_ns = int(stamp_ns)
        if self.last_image_stamp is not None and stamp_ns <= self.last_image_stamp:
            raise ValueError('Image timestamps must increase')
        if len(self.images) >= self.max_images:
            raise BufferError(f'Image queue reached its limit ({self.max_images})')
        self.images.append((stamp_ns, image))
        self.last_image_stamp = stamp_ns

    def push_imu(self, stamp_ns, gyro, accel):
        stamp_ns = int(stamp_ns)
        if self.last_imu_stamp is not None and stamp_ns <= self.last_imu_stamp:
            raise ValueError('IMU timestamps must increase')
        if len(self.imu) >= self.max_imu:
            raise BufferError(f'IMU queue reached its limit ({self.max_imu})')
        self.imu.append((stamp_ns, gyro, accel))
        self.last_imu_stamp = stamp_ns

    def pop_ready(self):
        """Return ``(image_stamp_ns, image, imu_samples)`` or None until the IMU brackets the image."""
        if not self.images or not self.imu:
            return None
        image_stamp, image = self.images[0]
        if self.imu[-1][0] < image_stamp:
            return None

        samples = []
        while self.imu:
            stamp, gyro, accel = self.imu.popleft()
            samples.append((stamp * 1e-9, gyro, accel))
            if stamp >= image_stamp:
                break
        self.images.popleft()
        return image_stamp, image, samples
