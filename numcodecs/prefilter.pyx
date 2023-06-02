# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3

import sys
import operator
import functools
import cython
from libc.stdlib cimport malloc, free
from cython cimport view
from libc.stdint cimport int32_t, uint32_t

from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from .compat_ext cimport Buffer
from .compat_ext import Buffer
from .abc import Codec
import math
from numcodecs.compat import ensure_contiguous_ndarray,ensure_ndarray


import numpy as np
cimport numpy as np


cdef extern from "MURaMKit_CAPI.h":

    int mkit_smart_log(void* buf,int is_float,size_t buf_len,void** meta)
    int mkit_smart_exp(void* buf, int is_float,size_t buf_len,const void* meta)
    size_t mkit_log_meta_len(const void* meta)

def smart_log(arr):
    
    cdef void* meta = NULL
    cdef char* dest_ptr
    cdef int32_t is_float
    cdef size_t meta_len
    cdef char* meta_buf
    cdef bytes meta_str = None
    cdef size_t src_len
    cdef Buffer dest_buffer = None



    src_len = arr.size

    dest = arr.copy()
    dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    dest_ptr = dest_buffer.ptr

    buf_shape = arr.shape
    if arr.dtype == 'f4':
        is_float = 1
    elif arr.dtype == 'f8': 
        is_float = 0
    mkit_smart_log(dest_ptr, is_float, src_len, &meta)
    meta_len = mkit_log_meta_len(&meta[0])

    meta_buf = <char *> &meta[0]
    meta_str = (<char *>meta_buf)[:meta_len]

    if dest_buffer is not None:
        dest_buffer.release()

    return dest, meta_str


def smart_exp(logged_arr,meta_str):

    cdef size_t src_len
    cdef char* dest_ptr
    cdef char* meta_ptr
    cdef int32_t is_float
    cdef Buffer dest_buffer = None

    if logged_arr.dtype == 'f4':
        is_float = 1
    elif logged_arr.dtype == 'f8': 
        is_float = 0
    src_len = logged_arr.size

    dest = logged_arr.copy()
    dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    dest_ptr = dest_buffer.ptr

    meta_buffer = Buffer(meta_str, PyBUF_ANY_CONTIGUOUS)
    meta_ptr = meta_buffer.ptr
    mkit_smart_exp(dest_ptr,is_float,src_len, meta_ptr)

    meta_buffer.release()
    if dest_buffer is not None:
        dest_buffer.release()

    return dest

class Prefilter(Codec):

    codec_id = 'prefilter'

    def encode(self,buf):
        return smart_log(buf)

    def decode(self,buf,meta):
        return smart_exp(buf,meta)

    def __repr__(self):
        r= "%s" %(type(self).__name__)
        return r
