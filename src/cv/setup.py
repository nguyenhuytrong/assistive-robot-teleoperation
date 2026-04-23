from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'cv', 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'tensorrt-cu13', 
        'torch',           
        'transformers',  
        'opencv-python', 
        'pycuda',        
        'numpy',
        'rclpy',
        'std_msgs',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='huytrongnghia2007@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'seg_node = cv.seg_node:main',
            'compressed_node = cv.compressed_node:main',
            'poly_node = cv.poly_node:main',
        ],
    },
)
