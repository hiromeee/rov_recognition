from setuptools import find_packages, setup

package_name = 'vision_pipeline_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hirokazu Tanaka',
    maintainer_email='hiro28me@gmail.com',
    description='Core vision pipeline for NEDO Challenge (ROS2-independent)',
    license='MIT',
)
