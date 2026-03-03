from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ps5'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/ps5']),
        ('share/ps5', ['package.xml']),
        (os.path.join('share', 'ps5', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='nguyenhuytronglqd2007@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ps5_haptic = ps5.ps5_haptic:main',
        ],
    },
)
