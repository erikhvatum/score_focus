// The MIT License (MIT)
//
// Copyright (c) 2016 WUSTL ZPLAB
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Authors: Erik Hvatum <ice.rikh@gmail.com>

#pragma once
#include <Python.h>
#include <numpy/npy_common.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstddef>

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1] 
void reorder_to_inner_outer(const Py_ssize_t* u_shape, const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,       Py_ssize_t* o_strides);

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1]. 
// Additionally, u_slave_shape is copied to o_slave_shape and u_slave_strides is copied to o_slave_strides, reversing 
// the elements of each if u_strides[0] < u_strides[1]. 
void reorder_to_inner_outer(const Py_ssize_t* u_shape,       const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,             Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape, const Py_ssize_t* u_slave_strides,
                                  Py_ssize_t* o_slave_shape,       Py_ssize_t* o_slave_strides);

// Copies u_shape to o_shape and u_strides to o_strides, reversing the elements of each if u_strides[0] < u_strides[1]. 
// Additionally, u_slave_shape0-1 is copied to o_slave_shape0-1 and u_slave_strides0-1 is copied to o_slave_strides0-1, 
// reversing the elements of each if u_strides[0] < u_strides[1]. 
void reorder_to_inner_outer(const Py_ssize_t* u_shape,        const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,              Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape0, const Py_ssize_t* u_slave_strides0,
                                  Py_ssize_t* o_slave_shape0,       Py_ssize_t* o_slave_strides0,
                            const Py_ssize_t* u_slave_shape1, const Py_ssize_t* u_slave_strides1,
                                  Py_ssize_t* o_slave_shape1,       Py_ssize_t* o_slave_strides1);

void _image_stack_median(const npy_float32* image_stack, const Py_ssize_t* image_stack_shape, const Py_ssize_t* image_stack_strides,
                         const npy_uint8* mask, const Py_ssize_t* mask_shape, const Py_ssize_t* mask_strides,
                         npy_float32* median, const Py_ssize_t* median_shape, const Py_ssize_t* median_strides);
