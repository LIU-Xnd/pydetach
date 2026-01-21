from typing import (
    Iterable as _Iterable,
    Literal as _Literal,
)
from scanpy import AnnData as _AnnData
from matplotlib.axes import Axes as _Axes
from numpy import ndarray as _NDArray

_NumberType = int | float
_Nx2ArrayType = _NDArray
_1DArrayType = _NDArray
from scipy.sparse import (
    csr_matrix as _csr_matrix,
    coo_matrix as _coo_matrix,
    dok_matrix as _dok_matrix,
    lil_matrix as _lil_matrix,
)


class _UndefinedType:
    """
    Placeholder indicating an undefined field.
    """
    def copy(self):
        return self

    def __repr__(self):
        return "_UNDEFINED"


# >>> Const
_UNDEFINED: _UndefinedType = _UndefinedType()
# <<< End of Const
