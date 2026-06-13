import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nedo_vision'

# ros2_ws/configs/: ros2_ws/src/nedo_vision/ -> ../../configs/
ws_configs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs'))
config_files = [os.path.relpath(f) for f in glob(os.path.join(ws_configs, '*.yaml'))]

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'configs'), config_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hirokazu Tanaka',
    maintainer_email='hiro28me@gmail.com',
    description='NEDO Challenge Vision Recognition Node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recognition_node = nedo_vision.recognition_node:main',
        ],
    },
)
