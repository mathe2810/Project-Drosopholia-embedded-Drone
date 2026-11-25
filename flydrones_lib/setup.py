from setuptools import setup, find_packages

setup(
    name="flydrones",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"flydrones": ["defaults.yaml"]},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "matplotlib>=3.7",
        "pillow>=9.0",
    ],
    extras_require={
        "data": ["pyarrow>=14", "pandas>=2.0"],
        "crazyflie": ["cflib>=0.1.25"],
    },
    entry_points={
        "console_scripts": [
            "flydrones=flydrones.cli:main",
        ],
    },
)