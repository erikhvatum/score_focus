# The MIT License (MIT)
#
# Copyright (c) 2014-2015 WUSTL ZPLAB
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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Willie Zhang

from concurrent import futures
import freeimage
import multiprocessing
import numpy
from zplib.image import mask as zplib_image_mask

MAX_WORKERS = multiprocessing.cpu_count()
pool = futures.ThreadPoolExecutor()

class WzBgs:
    def __init__(self, width, height, temporal_radius):
        assert all(int(d) == d and d > 0 for d in (width, height, temporal_radius)),\
            "WzBgs.__init__(self, width, height, temporal_radius): "\
            "width, height, temporal_radius must be positive integers"
        self.context_carousel = numpy.ndarray(
            (width, height, temporal_radius),
            strides=(temporal_radius*4, width*temporal_radius*4, 4),
            dtype=numpy.float32)
        # self.context_carousel_slices are views into self.context_carousel
        self.context_carousel_slices = [self.context_carousel[:, n:min(n+MAX_WORKERS,height), :] for n in range(0, height, MAX_WORKERS)]
        self.clear()

    def clear(self):
        self.context_id = 0
        self.context_idx = 0
        self.next_context_image_idx = 0
        self.model = None
        self.model_strips = None

    def _refresh_model(self):
        width, height = self.context_carousel.shape[:2]
        if self.model is None or self.model_strips is None:
            self.model = numpy.ndarray(
                (width, height),
                strides=(4, width*4),
                dtype=numpy.float32)
            self.model_strips = [self.model[:, n:min(n+MAX_WORKERS,height)] for n in range(0, height, MAX_WORKERS)]
        strip_futes = [pool.submit(numpy.median, slice, axis=2, out=strip) for strip, slice in zip(self.model_strips, self.context_carousel_slices)]
        for strip_fute in strip_futes:
            strip_fute.result()

    def updateModel(self, image, mask=None):
        temporal_radius = self.context_carousel.shape[2]
        assert image.shape == self.context_carousel.shape[:2]
        if self.context_id < temporal_radius:
            # Insufficient context was available for background model to be constructed.  Therefore, we cannot discern the foregound,
            # so we simply enter the input into the context.
            self.context_carousel[..., self.context_idx] = image
            self.context_id += 1
            self.context_idx += 1
            if self.context_id == temporal_radius:
                self.context_idx = 0
                # Entering the current input completed the context, permitting construction of a background model that future calls
                # will use for foreground discernment.
                self._refresh_model()
        else:
            if mask is None:
                mask = self.queryModelMask(image)
            self.context_carousel[..., self.context_idx] = image
            # Replace the region identified as the foreground with the corresponding region of the background model
            self.context_carousel[..., self.context_idx][mask] = self.model[mask]
            self.context_id += 1
            self.context_idx += 1
            if self.context_idx == temporal_radius:
                self.context_idx = 0
            self._refresh_model()
            return mask

    def queryModelDelta(self, image):
        if self.model is None:
            return
        assert image.shape == self.context_carousel.shape[:2]
        return self.model - image

    def queryModelMask(self, image, delta=None):
        if self.model is None:
            return
        if delta is None:
            delta = self.queryModelDelta(image)
        threshold = numpy.percentile(numpy.abs(delta), 97.5).astype(numpy.float32)
        mask = delta >= threshold
        mask[~zplib_image_mask.get_largest_object(mask)] = 0 # Remove dust from mask
        return mask

def processFlipbookPages(pages, temporal_radius=11):
    if not pages or not pages[0]:
        return []
    wzBgs = WzBgs(pages[0][0].size.width(), pages[0][0].size.height(), temporal_radius)
    vignette = freeimage.read('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/non-vignette.png') == 0
    ret = []
    try:
        for idx, page in enumerate(pages):
            image = page[0].data.astype(numpy.float32)
            image[vignette] = 0
            r = [image]
            mask = wzBgs.updateModel(image)
            if wzBgs.model is not None:
                r.append(wzBgs.model.astype(numpy.float32))
                if mask is not None:
                    r.append((mask*255).astype(numpy.uint8))
            ret.append(r)
            print('{} / {}'.format(idx+1, len(pages)))
    except KeyboardInterrupt:
        pass
    return ret

