from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

setup(
    name="searaft",
    version="0.1.0",
    description="SEA-RAFT: Scene-Adapted RAFT Optical Flow",
    author="limacv",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            'searaft=searaft:main',
        ],
    },
    include_package_data=True,
    python_requires='>=3.7',
)
