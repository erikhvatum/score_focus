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

import freeimage
import math
import numpy
import time
from zplib.image import mask as zplib_image_mask

from _cppmod import image_stack_median

class WzBgs:
    def __init__(self, width, height, temporal_radius, input_mask=None):
        assert all(int(d) == d and d > 0 for d in (width, height, temporal_radius)),\
            "WzBgs.__init__(self, width, height, temporal_radius): "\
            "width, height, temporal_radius must be positive integers"
        assert input_mask is None or input_mask.shape[0] == width and input_mask.shape[1] == height
        self.context_carousel = numpy.ndarray(
            (width, height, temporal_radius),
            strides=(temporal_radius*4, width*temporal_radius*4, 4),
            dtype=numpy.float32)
        if input_mask is None:
            input_mask = numpy.ndarray((width, height), strides=(1, width), dtype=numpy.uint8)
            input_mask[:] = 255
        self.input_mask = input_mask
        self.clear()

    def clear(self):
        self.context_id = 0
        self.context_idx = 0
        self.next_context_image_idx = 0
        self.model = None

    def _refresh_model(self):
        width, height = self.context_carousel.shape[:2]
        if self.model is None:
            self.model = numpy.ndarray(
                (width, height),
                strides=(4, width*4),
                dtype=numpy.float32)
        image_stack_median(self.context_carousel, self.input_mask, self.model)

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
        return [], None
    non_vignette = freeimage.read('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/non-vignette.png')
    vignette = non_vignette == 0
    wzBgs = WzBgs(pages[0][0].size.width(), pages[0][0].size.height(), temporal_radius, non_vignette)
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
    return ret, wzBgs

def computeFocusMeasures():
    from rpc_acquisition.scope.device.autofocus import MultiBrenner
    import sqlite3
    db = sqlite3.connect('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/analysis/db.sqlite3')

if __name__ == '__main__':
    import sys
    from PyQt5 import Qt
    app = Qt.QApplication(sys.argv)
    from ris_widget.ris_widget import RisWidget
    rw = RisWidget()
    rw.show()
    rw.add_image_files_to_flipbook(('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1508 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1528 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1548 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1608 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1637 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1704 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1714 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1728 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1748 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1808 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1828 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1848 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1908 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t1937 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2004 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2014 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2028 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2048 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2108 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2128 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2148 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2208 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2236 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2304 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2314 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2328 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-13t2348 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0008 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0028 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0048 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0108 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0137 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0204 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0214 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0228 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0248 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0308 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0328 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0348 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0408 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0436 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0504 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0514 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0528 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0548 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0608 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0628 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0648 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0708 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0736 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0804 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0814 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0828 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0848 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0908 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0928 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t0948 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1008 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1036 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1104 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1114 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1128 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1148 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1208 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1228 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1248 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1308 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1336 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1404 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1414 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1428 bf_ffc.png', '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/14/2015-11-14t1448 bf_ffc.png'))
    def on_do_button_clicked():
        processFlipbookPages(rw.flipbook_pages)
    do_button = Qt.QPushButton('processFlipbookPages')
    do_button.clicked.connect(on_do_button_clicked)
    do_button.show()
    app.exec()
