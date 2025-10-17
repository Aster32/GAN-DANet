"""Legacy import shim for GAN-DANet models."""
import models as _models
from models import *  # noqa: F401,F403

__all__ = _models.__all__
