"""Architectures for all models."""
from .detector import CenterNet
from .generator import Generator
from .basegenerator import BaseGenerator
__all__ = ["Generator","CenterNet","BaseGenerator"]