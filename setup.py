"""
TurboLane Server — Phase 2
RL-optimized parallel TCP file transfer application
"""
from setuptools import setup, find_packages

setup(
    name="turbolane-server",
    version="2.0.0",
    description="RL-optimized parallel TCP file transfer CLI (Phase 2)",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        # turbolane engine (Phase 1) must already be installed or on PYTHONPATH
    ],
    entry_points={
        "console_scripts": [
            "turbolane-server=turbolane_server.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: System :: Networking",
        "Programming Language :: Python :: 3.10",
    ],
)
