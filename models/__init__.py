"""GAN-DANet model components."""
from .generator import (
    CBAMBlock,
    FlexibleUpsamplingModule,
    OriginalRelationshipLearner,
    SqueezeExcitation,
)
from .hydro_downscaler import HydroTWSADownscaler
from .discriminator import Discriminator1, SRGAND
from .losses import (
    ConservationLoss,
    HydrologyLossBundle,
    HeteroscedasticGaussianLoss,
    PerceptualLoss,
    SpatialGradientLoss,
    SSIM,
    TVLoss,
)
from .utils import weights_init_normal

__all__ = [
    "CBAMBlock",
    "FlexibleUpsamplingModule",
    "OriginalRelationshipLearner",
    "SqueezeExcitation",
    "HydroTWSADownscaler",
    "Discriminator1",
    "SRGAND",
    "PerceptualLoss",
    "HeteroscedasticGaussianLoss",
    "ConservationLoss",
    "SpatialGradientLoss",
    "HydrologyLossBundle",
    "SSIM",
    "TVLoss",
    "weights_init_normal",
]
