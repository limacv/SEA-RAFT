#!/usr/bin/env python3
"""Setup script for SEA-RAFT package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='searaft',
    version='0.1.0',
    author='limacv',
    author_email='',
    description='SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/limacv/SEA-RAFT',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.7',
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        'searaft': ['*.py'],
        'core': ['*.py'],
        'core.utils': ['*.py'],
    },
    entry_points={
        'console_scripts': [
            'searaft-demo=searaft.demo:main',
        ],
    },
    keywords='optical flow, computer vision, deep learning, pytorch',
    project_urls={
        'Bug Reports': 'https://github.com/limacv/SEA-RAFT/issues',
        'Source': 'https://github.com/limacv/SEA-RAFT',
        'Paper': 'https://arxiv.org/abs/2405.14793',
    },
)
