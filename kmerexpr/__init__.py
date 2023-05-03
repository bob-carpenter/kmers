# BMW: Any content here will be available in the top level `kmerexpr` package
#     e.g., importable with `from kmerexpr import *`
import os
from ctypes import CDLL, c_int
from ctypes.util import find_library

_libmkl_rt_path = find_library('mkl_rt')
_libmkl_rt = CDLL(_libmkl_rt_path)

_libmkl_rt.MKL_Set_Interface_Layer
_libmkl_rt.MKL_Set_Interface_Layer.restype = c_int
_libmkl_rt.MKL_Set_Interface_Layer.argtypes = [c_int]


def _set_interface_layer():
    index_code: int = 1
    old_interface: int  = _get_interface_layer()
    if old_interface >= 2:  # GNU
        index_code += 2

    actual_interface = _libmkl_rt.MKL_Set_Interface_Layer(index_code)

    if actual_interface % 2 != 1:
        raise ImportError(
            """Unable to set MKL interface layer to use 64 bit ints. Try importing kmerexpr before other packages that rely on numpy.
            E.g. numpy, scipy, matplotlib, pandas, etc"""
        )


def _get_interface_layer() -> int:
    code: int = 0
    env: str = os.environ.get('MKL_INTERFACE_LAYER', "")
    if 'ILP64' in env:
        code += 1
    if 'GNU' in env:
        code += 2
    return code


_set_interface_layer()
