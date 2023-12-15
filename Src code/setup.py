from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'prius_sdc_pkg'
config_module = 'prius_sdc_pkg/config'
detection_module = 'prius_sdc_pkg/Detection'

lane_detection_module = 'prius_sdc_pkg/Detection/Lanes'
lane_detection_Stage1 = 'prius_sdc_pkg/Detection/Lanes/Stage1_Segmentation'
lane_detection_Stage2 = 'prius_sdc_pkg/Detection/Lanes/Stage2_Estimation'
lane_detection_Stage3 = 'prius_sdc_pkg/Detection/Lanes/Stage3_Cleaning'
lane_detection_Stage4 = 'prius_sdc_pkg/Detection/Lanes/Stage4_DataExtraction'

sign_detection_module = 'prius_sdc_pkg/Detection/Signs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, config_module, detection_module, lane_detection_module, lane_detection_Stage1,lane_detection_Stage2, lane_detection_Stage3, lane_detection_Stage4, sign_detection_module],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('lib', package_name), glob('scripts/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anhbui_cse',
    maintainer_email='anhbui_cse@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recorder_node = prius_sdc_pkg.video_recorder:main',
            'spawner_node = prius_sdc_pkg.sdf_spawner:main',
            'driver_node = prius_sdc_pkg.driving_node:main',
            'computer_vision_node = prius_sdc_pkg.carView:main',
        ],
    },
)
