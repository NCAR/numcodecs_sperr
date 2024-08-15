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
from libc.stdint cimport int32_t, uint32_t,uint8_t,uintptr_t

from cpython.buffer cimport PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE, PyObject_GetBuffer, PyBuffer_GetPointer, Py_buffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

from .compat_ext cimport Buffer
from .compat_ext import Buffer
from .abc import Codec
from .prefilter import Prefilter
import math
from numcodecs.compat import ensure_contiguous_ndarray,ensure_ndarray


import numpy as np
cimport numpy as cnp


cdef extern from "SPERR_C_API.h":

    # function declarations
    int sperr_comp_2d(const void* src, int32_t is_float, size_t dimx, size_t dimy, int32_t mode,  double quality, int out_inc_header, void** dst, size_t* dst_len) nogil
    int sperr_decomp_2d(const void* src, size_t serc_len, int32_t output_float, size_t dimx, size_t dimy, void** dst) nogil
    void sperr_parse_header(const void*, size_t* , size_t*, size_t*, int32_t*)
    int sperr_comp_3d(const void* src,int32_t is_float, size_t dimx, size_t dimy, size_t dimz, size_t chunkx, size_t chunky, size_t chunkz, int32_t mode, double quality, int32_t nthreads,void** dst, size_t* dst_len) nogil
    int sperr_decomp_3d(const void* src, size_t serc_len, int32_t output_float,int32_t nthreads, size_t* dimx, size_t* dimy, size_t * dimz, void** dst) nogil

    int sperr_compress(const void* src, int32_t is_float, size_t num_vals, int32_t num_dims, const size_t* dims, const size_t* chunks, int32_t mode, double quality, size_t num_thread, void** dst, size_t* dst_len) nogil
    int sperr_decompress(const void* src, size_t src_len, int32_t output_float, size_t num_threads, size_t* out_dims, void** dst) nogil

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

cdef class A:
    cdef float * ptr

    def get_ptr(self):
        return <uintptr_t>self.ptr
cdef class B:
    cdef float * f_ptr

    cpdef submit(self, uintptr_t ptr_var):
        self.f_ptr= <float *>ptr_var

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
    cdef int out_inc_header = 1
    cdef size_t dims[3]
    cdef size_t chunk_sizes[3]
    cdef size_t num_vals
    cdef double quality
    
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
        dims[0]= arr.shape[0]
        dims[1]= arr.shape[1]
        shape1 = arr.shape[0]
        shape2 = arr.shape[1]
        num_vals = shape1*shape2
        with nogil:
            #Sperr need the fastest dimension in the first
            ret=sperr_comp_2d(src_ptr,is_float,shape2,shape1,mode, level, out_inc_header, &dst,&dst_len)
    elif arr.ndim == 3:
        #print("before 3d compression",level)
        dims[0]= arr.shape[0]
        dims[1]= arr.shape[1]
        dims[2]= arr.shape[2]
        shape1 = arr.shape[0]
        shape2 = arr.shape[1]
        shape3 = arr.shape[2]
        num_vals = shape1*shape2*shape3
        with nogil:
            #Sperr need the fastest dimension in the first
            sperr_comp_3d(src_ptr,is_float,shape3,shape2,shape1,shape3,shape2,shape1,mode,level,nthreads, &dst,&dst_len)
    else:
        print("Array dimension should be 2D or 3D")
    #sperr_compress(src_ptr,is_float,num_vals,arr.ndim,dims,chunk_sizes,mode,quality,nthreads,&dst,&dst_len)
    buf = <char *> &dst[0]
    compress_str = (<char *> buf)[:dst_len]


    if source_buffer is not None:
        source_buffer.release()
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
    cdef size_t dim_x, dim_y, dim_z
    cdef void *src_ptr
    cdef void *src_ptr2
    cdef char *dst_ptr
    cdef Buffer dest_buffer = None
    cdef void* dst = NULL
    cdef void* compressed_ptr = NULL

    cdef Py_ssize_t header_len=10
    cdef Py_buffer buffer
    #source_buffer = Buffer(source, PyBUF_ANY_CONTIGUOUS)
    PyObject_GetBuffer(source, &(buffer), PyBUF_ANY_CONTIGUOUS)
    src_ptr = <char *> buffer.buf
    src_ptr2 = PyBuffer_GetPointer(&buffer,&header_len)
    src_len = buffer.len - header_len
    print('src_len = ',src_len)

    #src_ptr=<void*>(<const uint8_t*>source+header_len)
    sperr_parse_header(src_ptr,&dim_x, &dim_y, &dim_z, &orig_is_float)
    print('src_len = ',src_len)
    print('orig is float ',orig_is_float, dim_x, dim_y, dim_z)
    if orig_is_float == 0:
        the_type = np.float64
        output_float_i = 0
    else:
        the_type = np.float32
        output_float_i = 1
    if dim_z == 1:
        buf_shape = (dim_x, dim_y)
        datashape = dim_x * dim_y 
    else:
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

   
    print(dim_x,dim_y,dim_z)    
    with nogil:
        if dim_z == 1: 
            #Sperr need the fastest dimension in the first
            sperr_decomp_2d(src_ptr2, src_len, output_float_i, dim_y, dim_x, &dst)
        else:
            #Sperr need the fastest dimension in the first
            sperr_decomp_3d(src_ptr, buffer.len, output_float_i, nthreads_i, &dim_z, &dim_y, &dim_x, &dst)

    if output_float_i:
        if dim_z == 1:
            dst_arr = np.asarray(<cnp.float32_t[:dim_x,:dim_y]> dst)
        else:
            dst_arr = np.asarray(<cnp.float32_t[:dim_x,:dim_y,:dim_z]> dst) 
    else:
        if dim_z == 1:
            dst_arr = np.asarray(<cnp.float64_t[:dim_x,:dim_y]> dst)
        else:
            dst_arr = np.asarray(<cnp.float64_t[:dim_x,:dim_y,:dim_z]> dst) 

    #print(dst_arr[0,0])
    # release buffers
    PyBuffer_Release(&buffer)
    if dest_buffer is not None:
        dest_buffer.release()

    return dst_arr


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
       pre = '',
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
           pr =  Prefilter(mode=self.pre)
           buf,meta=pr.encode(buf)
           self.meta = meta
           #print('942',buf[4,7,2])
        return compress(buf, self.mode, self.level, autolevel=self.autolevel)

    def decode(self,buf,out=None):
        print('prefilter =',self.pre,type(buf))
        buf=ensure_contiguous_ndarray(buf,flatten=False)
        buf_out=decompress(buf,out)
        print('2 prefilter =',self.pre)
        if self.pre:
            pr = Prefilter(self.pre)
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
