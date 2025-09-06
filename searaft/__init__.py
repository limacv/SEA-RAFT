"""
SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow

This package provides the SEA-RAFT model for optical flow estimation.
"""

from .model import SEA_RAFT, remap

__version__ = "0.1.0"
__author__ = "limacv"
__email__ = ""

__all__ = ["SEA_RAFT", "remap"]
