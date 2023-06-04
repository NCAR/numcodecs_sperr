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
from .prefilter import Prefilter
import math
from numcodecs.compat import ensure_contiguous_ndarray,ensure_ndarray


import numpy as np
cimport numpy as np


cdef extern from "SPERR_C_API.h":

    # function declarations
    int sperr_comp_2d(const void* src, int32_t is_float, size_t dimx, size_t dimy, int32_t mode,  double quality, void** dst, size_t* dst_len) nogil
    int sperr_decomp_2d(const void* src, size_t serc_len, int32_t output_float, size_t* dimx, size_t* dimy, void** dst) nogil
    void sperr_parse_header(const void*, int32_t* , int32_t*, int32_t*, int32_t*, uint32_t*, uint32_t*, uint32_t*) 
    int sperr_decomp_user_mem(const void*, size_t, int32_t, size_t, void*) nogil
    int sperr_comp_3d(const void* src,int32_t is_float, size_t dimx, size_t dimy, size_t dimz, size_t chunkx, size_t chunky, size_t chunkz, int32_t mode, double quality, int32_t nthreads,void** dst, size_t* dst_len) nogil
    int sperr_decomp_3d(const void* src, size_t serc_len, int32_t output_float,int32_t nthreads, size_t* dimx, size_t* dimy, size_t * dimz, void** dst) nogil


@cython.final
cdef class Memory:
    cdef void* data
    def __cinit__(self, size_t size):
        self.data = malloc(size)
        if self.data == NULL:
            raise MemoryError()
    cdef void* __enter__(self):
        return self.data
    def __exit__(self, exc_type, exc_value, exc_tb):
        free(self.data)

def compress(
    arr, 
    int32_t mode = 3,
    double level = 0.01,
    int32_t nthreads = 1,
    int32_t autolevel = 0,
):
    cdef int ndim = arr.ndim
    cdef int32_t is_float 
    cdef void* dst = NULL
    cdef char* src_ptr
    cdef char* buf
    cdef size_t dst_len
    cdef int32_t qlev = 8
    cdef bytes compress_str = None
    cdef int32_t ret
    cdef size_t shape1, shape2, shape3 

    #print('input arr shape ', arr.ndim)
    if arr.dtype == object:
        raise TypeError('object arrays are not supported')
    if arr.ndim < 2 or arr.ndim > 3:
        raise TypeError('1D or 4D arrays are not supported')
    source_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS)
    src_ptr = source_buffer.ptr
    src_len = source_buffer.nbytes


    # Pick a compression level from max, min, average
    #print('mode=',mode,'level=',level,'autolevel = ',autolevel)
    if autolevel == 1:
        threshold = np.abs(np.min(arr))
        threshold2 = np.min(np.absolute(arr))
        #print('1 2 ',threshold, threshold2)
        if threshold2 < 1:
           exp = int(np.log10(np.abs(np.min(arr))))
           if exp == 0:
               level = 0.01
           else:
               level = 0.1**abs(exp+2)
           #print('exp=',exp)
        elif threshold >= 1:
           level = 0.1
        #print(level)

    # Input validation
    if arr is None:
        raise TypeError("Input array cannot be None")
    if arr.dtype == 'f4':
        is_float = 1
    elif arr.dtype == 'f8': 
        is_float = 0
    if arr.ndim == 2:
        shape1 = arr.shape[0]
        shape2 = arr.shape[1]
        #print("before 2d compression",level,is_float,arr.shape)
        with nogil:
            ret=sperr_comp_2d(src_ptr,is_float,shape1,shape2,mode, level, &dst,&dst_len)
    elif arr.ndim == 3:
        #print("before 3d compression",level)
        shape1 = arr.shape[0]
        shape2 = arr.shape[1]
        shape3 = arr.shape[2]
        with nogil:
            sperr_comp_3d(src_ptr,is_float,shape1,shape2,shape3,shape1, shape2,shape3,mode, level,nthreads, &dst,&dst_len)
    else:
        print("Array dimension should be 2D or 3D")
    buf = <char *> &dst[0]
    #print("compress_str length", dst_len)
    compress_str = (<char *> buf)[:dst_len]


    return compress_str


def decompress(
    source, dest=None, nthreads=1, output_float=0):

    cdef int32_t nthreads_i=nthreads
    cdef int32_t output_float_i = output_float

    cdef float* float_buf
    cdef double* double_buf
    cdef int32_t version_major
    cdef int32_t zstd_applied
    cdef int32_t is_3d,orig_is_float
    cdef uint32_t dim_x, dim_y, dim_z
    cdef char *src_ptr
    cdef char *dst_ptr
    cdef Buffer dest_buffer = None


    source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    src_ptr = source_buffer.ptr
    src_len = source_buffer.nbytes
    #print('src_len = ',src_len)


    sperr_parse_header(src_ptr,&version_major, &zstd_applied, &is_3d, &orig_is_float, &dim_x, &dim_y, &dim_z)
    #print(version_major, zstd_applied, is_3d, 'orig is float ',orig_is_float, dim_x, dim_y, dim_z)
    if orig_is_float == 0:
        the_type = np.float64
        output_float = 0
    else:
        the_type = np.float32
        output_float = 1
    if is_3d == 0:
        buf_shape = (dim_x, dim_y)
        datashape = dim_x * dim_y 
    elif is_3d :
        buf_shape = (dim_x, dim_y, dim_z)
        datashape = dim_x * dim_y * dim_z

    if dest is None: 
        dest = PyBytes_FromStringAndSize(NULL,datashape*np.dtype(the_type).itemsize)
        dest_ptr = PyBytes_AS_STRING(dest)
    else:
        if dest.ndim < 2 and dest.ndim > 3:
            raise TypeError('1D or 4D arrays are not supported')
        arr = ensure_contiguous_ndarray(dest,flatten=False)
        dest_buffer = Buffer(arr, PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
        dest_ptr = dest_buffer.ptr
        #print('dest shape ',dest.shape, dest.dtype) 

   
    with nogil:
        sperr_decomp_user_mem(src_ptr, src_len, output_float_i, nthreads_i, dest_ptr)

    dest = ensure_ndarray(dest).view(the_type)
    dst = dest.reshape(buf_shape)

    # release buffers
    source_buffer.release()
    if dest_buffer is not None:
        dest_buffer.release()

    return dst


class Sperr(Codec):
    """Codec providing compression using SPERR via the Python standard library.
 
    Attributes
    ----------
    mode(mode=1)
       Fixed bit per pixel compression.
    level : double 
       Bit per pixel, default 2.5
    
    mode(mode=2)
       Fixed point signal noise ratio compression.
    level : double
       Point signal noise ratio targe value, can be 60 and more

    mode(mode=3)
      Fixed point-wise error compression.
    level : double
      Compressed bits per floating-point value, can be 0-64 for double precision
    """
    codec_id = 'sperr'

    def __init__(
       self,
       mode = 3,
       level = 0.01,
       autolevel = 0,
       pre = 0,
       meta = None
    ):
       self.mode = mode
       self.level = level
       self.autolevel = autolevel
       self.pre = pre
       self.meta = meta
           
       

    def encode(self,buf):
        #buf = ensure_contiguous_ndarray(buf)
        #self.datatype=buf.dtype
        if self.pre:
           pr =  Prefilter()
           buf,meta=pr.encode(buf)
           self.meta = meta
        return compress(buf, self.mode, self.level, autolevel=self.autolevel)

    def decode(self,buf,out=None):
        buf=ensure_contiguous_ndarray(buf,flatten=False)
        buf_out=decompress(buf,out)
        if self.pre:
            pr = Prefilter()
            buf_out = pr.decode(buf_out,self.meta)
        return buf_out
        #return decompress(buf,out)

    def __repr__(self):
        r = "%s(mode=%r,level=%s,autolevel=%r,pre=%r,meta=%s)" % (
            type(self).__name__,
            self.mode,
            self.level,
            self.autolevel,
            self.pre,
            self.meta,
	)
        return r
