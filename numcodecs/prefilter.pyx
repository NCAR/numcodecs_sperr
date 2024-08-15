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
    int mkit_slice_norm(void* buf, int is_float,size_t dim_fast,
        size_t dim_mid, size_t dim_slow, void** meta)
    int mkit_inv_slice_norm(void* buf, int is_float, size_t dim_fast,
        size_t dim_mid, size_t dim_slow, const void* meta)
    size_t mkit_slice_norm_meta_len(const void* meta)
    int mkit_bitmask_zero(const void* inbuf, int is_float, size_t len,
        void** output)
    int mkit_inv_bitmask_zero(const void* inbuf, void** output)
    size_t mkit_bitmask_zero_buf_len(const void* input)

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

    if arr.dtype == 'f4':
        is_float = 1
    elif arr.dtype == 'f8': 
        is_float = 0
    mkit_smart_log(dest_ptr, is_float, src_len, &meta)
    meta_len = mkit_log_meta_len(meta)
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

def slice_norm(arr):
    
    cdef void* meta = NULL
    cdef char* dest_ptr
    cdef int32_t is_float
    cdef size_t meta_len
    cdef char* meta_buf
    cdef bytes meta_str = None
    cdef size_t shape1, shape2, shape3
    cdef Buffer dest_buffer = None

    shape1 = arr.shape[0]
    shape2 = arr.shape[1]
    shape3 = arr.shape[2]

    dest = arr.copy()
    dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    dest_ptr = dest_buffer.ptr

    if arr.dtype == 'f4':
        is_float = 1
    elif arr.dtype == 'f8': 
        is_float = 0
    mkit_slice_norm(dest_ptr, is_float, shape3, shape2, shape1, &meta)
    meta_len = mkit_slice_norm_meta_len(meta)
    meta_buf = <char *> &meta[0]
    meta_str = (<char *>meta_buf)[:meta_len]

    if dest_buffer is not None:
        dest_buffer.release()

    return dest, meta_str


def inv_slice_norm(normed_arr,meta_str):

    cdef char* dest_ptr
    cdef char* meta_ptr
    cdef int32_t is_float
    cdef size_t shape1, shape2, shape3
    cdef Buffer dest_buffer = None
    cdef Buffer meta_buffer = None

    shape1 = normed_arr.shape[0]
    shape2 = normed_arr.shape[1]
    shape3 = normed_arr.shape[2]
    if normed_arr.dtype == 'f4':
        is_float = 1
    elif normed_arr.dtype == 'f8': 
        is_float = 0

    dest = normed_arr.copy()
    dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    dest_ptr = dest_buffer.ptr

    meta_buffer = Buffer(meta_str, PyBUF_ANY_CONTIGUOUS)
    meta_ptr = meta_buffer.ptr
    mkit_inv_slice_norm(dest_ptr,is_float, shape3, shape2, shape1, meta_ptr)

    meta_buffer.release()
    if dest_buffer is not None:
        dest_buffer.release()

    return dest

def bitmask_zero(arr):

    cdef void* dest = NULL
    cdef char* source_ptr
    cdef int32_t is_float
    cdef size_t dest_len
    cdef char* dest_buf
    cdef bytes dest_str = None
    cdef size_t src_len
    cdef Buffer source_buffer = None

    src_len = arr.size

    source_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    source_ptr = source_buffer.ptr

    buf_shape = arr.shape
    if arr.dtype == 'f4':
        is_float = 1
    elif arr.dtype == 'f8': 
        is_float = 0
    mkit_bitmask_zero(source_ptr, is_float, src_len, &dest)
    dest_len = mkit_bitmask_zero_buf_len(&dest[0])

    dest_buf = <char *> &dest[0]
    dest_str = (<char *>dest_buf)[:dest_len]

    if source_buffer is not None:
        source_buffer.release()

    return dest_str

def inv_bitmask_zero(masked_arr,meta_str):

    cdef char* dest_ptr
    cdef void* outarr = NULL
    cdef int32_t is_float
    cdef size_t shape1, shape2, shape3
    cdef Buffer dest_buffer = None

    shape1 = masked_arr.shape[0]
    shape2 = masked_arr.shape[1]
    shape3 = masked_arr.shape[2]

    if masked_arr.dtype == 'f4':
        is_float = 1
    elif masked_arr.dtype == 'f8': 
        is_float = 0

    dest = masked_arr.copy()
    dest_buffer = Buffer(dest, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
    dest_ptr = dest_buffer.ptr

    meta_buffer = Buffer(meta_str, PyBUF_ANY_CONTIGUOUS)
    meta_ptr = meta_buffer.ptr
    mkit_inv_bitmask_zero(dest_ptr, &outarr)

    meta_buffer.release()
    if dest_buffer is not None:
        dest_buffer.release()

    return dest
class Prefilter(Codec):

    codec_id = 'prefilter'
    def __init__(
           self,
           mode = 'log',
           mask = 0 ,
           meta_str = ''
        ):
           self.mode = mode
           self.mask = mask
           self.meta_str = meta_str
    
    def encode(self,buf):
        if self.mode == 'log':
            ret_buf, self.meta_str = smart_log(buf)
        elif self.mode == 'norm':
            ret_buf, self.meta_str = slice_norm(buf)
            #with open("file.bin", "wb") as f:
            #    f.write(self.meta_str)

        elif self.mode == 'bmask':
            ret_buf = bitmask_zero(buf)
        else:
            ret_buf = buf
        return ret_buf

    def decode(self,buf):
        if self.mode == 'log':
            ret_buf = smart_exp(buf,self.meta_str)
        elif self.mode == 'norm':
            ret_buf = inv_slice_norm(buf,self.meta_str) 
        elif self.mode == 'bmask':
            ret_buf = inv_bitmask_zero(buf,self.meta_str)
        else:
            ret_buf = buf
        return ret_buf

    def __repr__(self):
        r= "%s" %(type(self).__name__)
        return r
