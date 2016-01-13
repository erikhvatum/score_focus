# The MIT License (MIT)
#
# Copyright (c) 2016 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

import cython
from libcpp cimport bool
cimport numpy
import numpy

cdef extern from "_image_stack_median.h":
    void _image_stack_median(
        const numpy.float32_t* image_stack, const Py_ssize_t* image_stack_shape, const Py_ssize_t* image_stack_strides,
        const numpy.uint8_t* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
        numpy.float32_t* median, const Py_ssize_t* median_shape, const Py_ssize_t* median_strides) nogil

@cython.boundscheck(False)
cpdef image_stack_median(numpy.float32_t[:, :, :] image_stack, numpy.uint8_t[:, :] mask, numpy.float32_t[:, :] median):
    if image_stack.strides[2] > image_stack.strides[0] or image_stack.strides[2] > image_stack.strides[1]:
        raise ValueError('image_stack axis 2 must be depth')
    if image_stack.shape[2] % 2 == 0:
        raise ValueError('image_stack depth (axis 2) must be odd')
    order = image_stack.strides[0] < image_stack.strides[1]
    if (order != (mask.strides[0] < mask.strides[1]) or
        order != (median.strides[0] < median.strides[1])):
        raise ValueError('image_stack width and height stride ordering must be the same as mask and median.')
    if (image_stack.shape[0], image_stack.shape[1]) != (mask.shape[0], mask.shape[1]) or (image_stack.shape[0], image_stack.shape[1]) != (median.shape[0], median.shape[1]):
        raise ValueError('image_stack width and height must be the same as mask and median.')
    with nogil:
        _image_stack_median(
            &image_stack[0][0][0], &image_stack.shape[0], &image_stack.strides[0],
            &mask[0][0], &mask.shape[0], &mask.strides[0],
            &median[0][0], &median.shape[0], &median.strides[0])
