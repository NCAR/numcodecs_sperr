import itertools


import numpy as np
import pytest


try:
    from numcodecs.sperr import Sperr
except ImportError:  # pragma: no cover
    pytest.skip(
        "numcodecs.sperr not available", allow_module_level=True
    )


from numcodecs.tests.common import (check_encode_decode_array, check_config, check_repr,
                                    check_backwards_compatibility,
                                    check_err_decode_object_buffer,
                                    check_err_encode_object_buffer)


codecs = [
    #Sperr(),
    Sperr(mode=3, level=0.001),
    Sperr(mode=1, level=24),
    #Sperr(mode=2, level=60),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.linspace(1000, 1001, 1000, dtype="f4").reshape(100,10),
    np.linspace(1000, 1001, 1000, dtype="f8").reshape(100,10),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.normal(loc=1000, scale=1, size=(10, 10, 10)),
]


def test_encode_decode():
    i=0
    precision=1
    for arr, codec in itertools.product(arrays, codecs):
        print('i=',i,arr.shape,arr.dtype)
        check_encode_decode_array(arr, codec, precision=precision)
        i=i+1


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Sperr(mode=1,level=16)")


def test_backwards_compatibility():
    precision=[0.01, 0.001]
    check_backwards_compatibility(Sperr.codec_id, arrays, codecs,precision=precision)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Sperr())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Sperr())
