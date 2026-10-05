"""Declarative realizations and a local browser viewer.

Install ``mathfirst[viz]`` to run the viewer. Importing these objects does not
load the optional numerical or web backends.
"""

from .realization import FunctionGraph
from .viewer import Viewer

__all__ = ["FunctionGraph", "Viewer"]
