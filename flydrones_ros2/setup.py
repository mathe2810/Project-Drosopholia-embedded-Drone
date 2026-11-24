from setuptools import find_packages, setup
import glob
import os

package_name = "flydrones_ros2"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob.glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob.glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Matheo",
    maintainer_email="matheo@flydrones.dev",
    description="FlyDrones ROS 2 node: Webots + Drosophila brain integration",
    license="MIT",
    entry_points={
        "console_scripts": [
            "flydrones_node = flydrones_ros2.node:main",
        ],
    },
)