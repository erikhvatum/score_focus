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

#include "_image_stack_median.h"
#include <algorithm>
#include <vector>

void reorder_to_inner_outer(const Py_ssize_t* u_shape, const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,       Py_ssize_t* o_strides)
{
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
    }
}

void reorder_to_inner_outer(const Py_ssize_t* u_shape,       const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,             Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape, const Py_ssize_t* u_slave_strides,
                                  Py_ssize_t* o_slave_shape,       Py_ssize_t* o_slave_strides)
{
    // The u_strides[0] < u_strides[1] comparison controlling slave shape and striding reversal is not a typo: slave
    // striding and shape are reversed if non-slave striding and shape are reversed. 
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
        o_slave_strides[0] = u_slave_strides[0]; o_slave_strides[1] = u_slave_strides[1];
        o_slave_shape[0] = u_slave_shape[0]; o_slave_shape[1] = u_slave_shape[1];
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
        o_slave_strides[0] = u_slave_strides[1]; o_slave_strides[1] = u_slave_strides[0];
        o_slave_shape[0] = u_slave_shape[1]; o_slave_shape[1] = u_slave_shape[0];
    }
}

void reorder_to_inner_outer(const Py_ssize_t* u_shape,        const Py_ssize_t* u_strides,
                                  Py_ssize_t* o_shape,              Py_ssize_t* o_strides,
                            const Py_ssize_t* u_slave_shape0, const Py_ssize_t* u_slave_strides0,
                                  Py_ssize_t* o_slave_shape0,       Py_ssize_t* o_slave_strides0,
                            const Py_ssize_t* u_slave_shape1, const Py_ssize_t* u_slave_strides1,
                                  Py_ssize_t* o_slave_shape1,       Py_ssize_t* o_slave_strides1)
{
    // The u_strides[0] < u_strides[1] comparison controlling slave shape and striding reversal is not a typo: slave
    // striding and shape are reversed if non-slave striding and shape are reversed. 
    if(u_strides[0] >= u_strides[1])
    {
        o_strides[0] = u_strides[0]; o_strides[1] = u_strides[1];
        o_shape[0] = u_shape[0]; o_shape[1] = u_shape[1];
        o_slave_strides0[0] = u_slave_strides0[0]; o_slave_strides0[1] = u_slave_strides0[1];
        o_slave_shape0[0] = u_slave_shape0[0]; o_slave_shape0[1] = u_slave_shape0[1];
        o_slave_strides1[0] = u_slave_strides1[0]; o_slave_strides1[1] = u_slave_strides1[1];
        o_slave_shape1[0] = u_slave_shape1[0]; o_slave_shape1[1] = u_slave_shape1[1];
    }
    else
    {
        o_strides[0] = u_strides[1]; o_strides[1] = u_strides[0];
        o_shape[0] = u_shape[1]; o_shape[1] = u_shape[0];
        o_slave_strides0[0] = u_slave_strides0[1]; o_slave_strides0[1] = u_slave_strides0[0];
        o_slave_shape0[0] = u_slave_shape0[1]; o_slave_shape0[1] = u_slave_shape0[0];
        o_slave_strides1[0] = u_slave_strides1[1]; o_slave_strides1[1] = u_slave_strides1[0];
        o_slave_shape1[0] = u_slave_shape1[1]; o_slave_shape1[1] = u_slave_shape1[0];
    }
}

void _image_stack_median(const npy_float32* image_stack, const Py_ssize_t* _image_stack_shape, const Py_ssize_t* _image_stack_strides,
                         const npy_uint8* mask, const Py_ssize_t* _mask_shape, const Py_ssize_t* _mask_strides,
                         npy_float32* median, const Py_ssize_t* _median_shape, const Py_ssize_t* _median_strides)
{
    Py_ssize_t image_stack_shape[3], image_stack_strides[3], mask_shape[2], mask_strides[2], median_shape[2], median_strides[2];
    image_stack_shape[2] = _image_stack_shape[2];
    image_stack_strides[2] = _image_stack_strides[2];
    reorder_to_inner_outer(_image_stack_shape, _image_stack_strides,
                           image_stack_shape, image_stack_strides,
                           _mask_shape, _mask_strides,
                           mask_shape, mask_strides,
                           _median_shape, _median_strides,
                           median_shape, median_strides);
    const std::ptrdiff_t image_inner_end_offset = image_stack_shape[1] * image_stack_strides[1];
    const std::ptrdiff_t image_layer_end_offset = image_stack_shape[2] * image_stack_strides[2];
    const std::ptrdiff_t pixel_stack_median_offset{static_cast<std::ptrdiff_t>(image_stack_shape[2] / 2)};
    const std::ptrdiff_t outer_idx_count=image_stack_shape[0];
    #pragma omp parallel for
    for(std::ptrdiff_t outer_idx=0; outer_idx < outer_idx_count; ++outer_idx)
    {
        std::vector<npy_float32> pixel_stack_vector(image_stack_shape[2], 0);
        npy_float32 *pixel_stack{pixel_stack_vector.data()}, *pixel;
        npy_float32* pixel_stack_end{pixel_stack + image_stack_shape[2]};
        npy_float32* pixel_stack_median{pixel_stack + pixel_stack_median_offset};
        const npy_uint8* image_inner{reinterpret_cast<const npy_uint8*>(image_stack) + image_stack_strides[0] * outer_idx};
        const npy_uint8*const image_inner_end{image_inner + image_inner_end_offset};
        const npy_uint8* mask_inner{mask + mask_strides[0] * outer_idx};
        npy_uint8* median_inner{reinterpret_cast<npy_uint8*>(median) + median_strides[0] * outer_idx};
        const npy_uint8 *image_layer, *image_layer_end;
        for(;;)
        {
            if(*mask_inner != 0)
            {
                image_layer = image_inner;
                image_layer_end = image_layer + image_layer_end_offset;
                pixel = pixel_stack;
                for(;;)
                {
                    *pixel = *reinterpret_cast<const npy_float32*>(image_layer);
                    image_layer += image_stack_strides[2];
                    if(image_layer == image_layer_end) break;
                    ++pixel;
                }
                std::nth_element(pixel_stack, pixel_stack_median, pixel_stack_end);
                *reinterpret_cast<npy_float32*>(median_inner) = *pixel_stack_median;
            }
            else
            {
                *reinterpret_cast<npy_float32*>(median_inner) = 0;
            }
            image_inner += image_stack_strides[1];
            if(image_inner == image_inner_end) break;
            mask_inner += mask_strides[1];
            median_inner += median_strides[1];
        }
    }
}
