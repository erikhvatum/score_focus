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
from pathlib import Path
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

from rpc_acquisition.scope.device.autofocus import MultiBrenner
import os.path
import wautofocuser
from zplib.image import fast_fft

FFTW_WISDOM = os.path.expanduser('~/fftw_wisdom')
if os.path.exists(FFTW_WISDOM):
    fast_fft.load_plan_hints(FFTW_WISDOM)

class MaskedAutofocusMetric:
    def __init__(self, shape):
        self.focus_scores = []

    def evaluate_image(self, image, mask):
        self.focus_scores.append(self.metric(image, mask))

    def metric(self, image, mask):
        raise NotImplementedError()

    def find_best_focus_index(self):
        best_i = numpy.argmax(self.focus_scores)
        focus_scores = self.focus_scores
        self.focus_scores = []
        return best_i, focus_scores

class MaskedBrenner(MaskedAutofocusMetric):
    def metric(self, image, mask):
        image = image.astype(numpy.float32) # otherwise can get overflow in the squaring and summation
        # Exclude unmasked regions from edge detection output lest the masked region's border register as a large, spurious edge that may
        # dominate the measure and make it a measure of mask region border length rather than information density within the masked region
        # (depending on image content and spatial filter preprocessing).
        x_diffs = (image[2:, :] - image[:-2, :])**2
        x_diffs[mask[:2558,:]] = 0
        y_diffs = (image[:, 2:] - image[:, :-2])**2
        y_diffs[mask[:,:2158]] = 0
        return x_diffs.sum() + y_diffs.sum()

class MaskedFilteredBrenner(MaskedBrenner):
    def __init__(self, shape):
        super().__init__(shape)
        t0 = time.time()
        self.filter = fast_fft.SpatialFilter(shape, self.PERIOD_RANGE, precision=32, threads=64, better_plan=False)
        if time.time() - t0 > 0.5:
            fast_fft.store_plan_hints(FFTW_WISDOM)

    def metric(self, image, measure_mask):
        filtered = self.filter.filter(image)
        return super().metric(filtered, measure_mask)

class MaskedHighpassBrenner(MaskedFilteredBrenner):
    PERIOD_RANGE = (None, 10)

class MaskedBandpassBrenner(MaskedFilteredBrenner):
    PERIOD_RANGE = (60, 100)

class MaskedMultiBrenner(MaskedAutofocusMetric):
    def __init__(self, shape):
        super().__init__(shape)
        self.hp = MaskedHighpassBrenner(shape)
        self.bp = MaskedBandpassBrenner(shape)

    def metric(self, image, measure_mask):
        return self.hp.metric(image, measure_mask), self.bp.metric(image, measure_mask)

DELTAS = []
MASKS = []

def _computeFocusMeasures(bgs, im_fpath, measure_mask):
    try:
        im = freeimage.read(str(im_fpath))
    except:
        return
    if bgs.model is not None:
        try:
            delta = numpy.abs(bgs.queryModelDelta(im))
            mask = bgs.queryModelMask(im, delta)
            DELTAS.append(delta)
            MASKS.append(mask)
        except:
            return
        focus_measures = {}
        focus_measures['whole_image_hp_brenner_sum_of_squares'], focus_measures['whole_image_bp_brenner_sum_of_squares'] = MultiBrenner((2560, 2160)).metric(im)
        model_delta_squares = delta.astype(numpy.float64)**2
        focus_measures['model_delta_sum_of_squares'] = model_delta_squares.sum()
        focus_measures['model_mask_count'] = mask.sum()
        model_delta_squares[measure_mask] = 0
        focus_measures['model_mask_region_delta_sum_of_squares'] = model_delta_squares.sum()
        focus_measures['model_mask_region_image_hp_brenner_sum_of_squares'], focus_measures['model_mask_region_image_bp_brenner_sum_of_squares'] = MaskedMultiBrenner((2560,2160)).metric(im, measure_mask)
        return focus_measures

def computeFocusMeasures(temporal_radius=11):
    import sqlite3
    with sqlite3.connect('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/analysis/db.sqlite3') as db:
        db.row_factory = sqlite3.Row
        non_vignette = freeimage.read('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/non-vignette.png')
        vignette = non_vignette == 0
        positions = [row['well_idx'] for row in db.execute('select well_idx from wells where did_hatch')]
        position_bgss = {position : WzBgs(2560, 2160, temporal_radius, non_vignette) for position in positions}
        time_points = [row['name'] for row in db.execute('select name from time_points')]
        for time_point in time_points:
            if time_point == '2015-11-13t1848':
                return
            print(time_point)
            for position in positions:
                print('', position)
                acquisition_names = [row['acquisition_name'] for row in db.execute('select * from images where well_idx=? and time_point=?', (position, time_point))]
                if acquisition_names:
                    bgs = position_bgss[position]
                    for acquisition_name in acquisition_names:
                        print(' ', acquisition_name)
                        im_fpath = Path('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/') / '{:02}'.format(position) / '{} {}_ffc.png'.format(time_point, acquisition_name)
                        focus_measures = _computeFocusMeasures(bgs, im_fpath, vignette)
                        if focus_measures is not None:
                            measure_names = sorted(focus_measures.keys())
                            q = 'update images set ' + ', '.join('{}=?'.format(measure_name) for measure_name in measure_names)
                            q+= ' where well_idx=? and time_point=? and acquisition_name=?'
                            v = [float(focus_measures[measure_name]) for measure_name in measure_names]
                            v.extend((position, time_point, acquisition_name))
                            list(db.execute(q, v))
                    try:
                        bgs.updateModel(freeimage.read(str(Path('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3/') / '{:02}'.format(position) / '{} bf_ffc.png'.format(time_point))))
                    except:
                        pass
                    db.commit()

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
