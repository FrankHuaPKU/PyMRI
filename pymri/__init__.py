# A Python package for data analysis of MRI-driven turbulence 

__version__ = "0.1.0"

from .turbulence import ScalarField, VectorField, Turbulence
from .spectra import Spectrum, MagneticSpectra

__all__ = [
    "ScalarField",
    "VectorField",
    "Turbulence",
    "Spectrum",
    "MagneticSpectra"
]
