from setuptools import find_packages, setup


package_name = 'accord_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cuong Huynh',
    maintainer_email='huynhtancuong.hcm@gmail.com',
    description='ROS 2 camera-IMU wrapper for ACCORD.',
    license='GPL-3.0-only',
    entry_points={
        'console_scripts': [
            'accord_node = accord_ros.node:main',
        ],
    },
)
