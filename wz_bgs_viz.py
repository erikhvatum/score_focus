# The MIT License (MIT)
#
# Copyright (c) 2014-2016 WUSTL ZPLAB
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

from concurrent.futures import ThreadPoolExecutor
import freeimage
import numpy
from pathlib import Path
import re
from ris_widget import om
from ris_widget.image import Image
from ris_widget.layer import Layer
import sqlite3
import sys
import time

DPATHSTR = {
    'darwin' : '/Volumes/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3'
}.get(sys.platform, '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3')
DPATH = Path(DPATHSTR)

from rpc_acquisition.scope.device.autofocus import MultiBrenner
import os.path
from zplib.image import fast_fft

FFTW_WISDOM = os.path.expanduser('~/fftw_wisdom')
if os.path.exists(FFTW_WISDOM):
    fast_fft.load_plan_hints(FFTW_WISDOM)

class AutofocusMetric:
    def __init__(self, shape):
        pass

    def metric(self, image):
        raise NotImplementedError()

class Brenner(AutofocusMetric):
    def metric(self, image):
        image = image.astype(numpy.float32) # otherwise can get overflow in the squaring and summation
        xd = image.astype(numpy.float32)
        yd = image.astype(numpy.float32)
        x_diffs = (image[2:, :] - image[:-2, :])**2
        xd[1:-1, :] = x_diffs
        y_diffs = (image[:, 2:] - image[:, :-2])**2
        yd[:,1:-1] = y_diffs
        return xd + yd

class FilteredBrenner(Brenner):
    def __init__(self, shape):
        super().__init__(shape)
        t0 = time.time()
        self.filter = fast_fft.SpatialFilter(shape, self.PERIOD_RANGE, precision=32, threads=64, better_plan=False)
        if time.time() - t0 > 0.5:
            fast_fft.store_plan_hints(FFTW_WISDOM)

    def metric(self, image):
        filtered = self.filter.filter(image)
        return super().metric(filtered)

class HighpassBrenner(FilteredBrenner):
    PERIOD_RANGE = (None, 10)

class BandpassBrenner(FilteredBrenner):
    PERIOD_RANGE = (60, 100)

class MultiBrenner(AutofocusMetric):
    def __init__(self, shape):
        super().__init__(shape)
        self.hp = HighpassBrenner(shape)
        self.bp = BandpassBrenner(shape)

    def metric(self, image):
        return self.hp.metric(image), self.bp.metric(image)

pool = ThreadPoolExecutor()

def makeWellViz(rw, wellIdx=14):
    non_vignette = freeimage.read(str(DPATH / 'non-vignette.png'))
    vignette = non_vignette == 0

    def insert_viz_images():
        def process_page(page):
            bf_ffc = page[0].data.astype(numpy.uint16)
            bf_ffc[vignette] = 0
            delta2 = (page[3].data**2).astype(numpy.float32)
            brenner_hp, brenner_bp = MultiBrenner((2560,2160)).metric(page[0].data)
            measure_antimask = page[2].data == 0
            m_delta = page[3].data.astype(numpy.float32)
            m_delta[measure_antimask] = 0
            m_delta2 = delta2.astype(numpy.float32)
            m_delta2[measure_antimask] = 0
            m_brenner_hp = brenner_hp.astype(numpy.float32)
            m_brenner_hp[measure_antimask] = 0
            m_brenner_bp = brenner_bp.astype(numpy.float32)
            m_brenner_bp[measure_antimask] = 0
            return (
                page,
                bf_ffc,
                delta2,
                brenner_hp,
                brenner_bp,
                m_delta,
                m_delta2,
                m_brenner_hp,
                m_brenner_bp)
        tasks = [pool.submit(process_page, page) for page in rw.flipbook.pages]
        for task in tasks:
            r = task.result()
            (
                page,
                bf_ffc,
                delta2,
                brenner_hp,
                brenner_bp,
                m_delta,
                m_delta2,
                m_brenner_hp,
                m_brenner_bp
            ) = r
            page[0].set_data(bf_ffc)
            page.extend((
                delta2,
                brenner_hp,
                brenner_bp,
                m_delta,
                m_delta2,
                m_brenner_hp,
                m_brenner_bp))
    rw.layers = [
        Layer(name='bf ffc'),
        Layer(name='model'),
        Layer(name='mask'),
        Layer(name='delta'),
        Layer(name='delta^2'),
        Layer(name='brenner hp'),
        Layer(name='brenner bp'),
        Layer(name='masked delta'),
        Layer(name='masked delta^2'),
        Layer(name='masked brenner hp'),
        Layer(name='masked brenner bp')]
    dpath = DPATH / '{:02}'.format(wellIdx)
    pages = []
    for delta_fpath in sorted(dpath.glob('*')):
        match = re.match(r'(201.* bf_ffc) wz_bgs_model_mask.png', delta_fpath.name)
        if match:
            prefix = match.group(1)
            bf_fpath = dpath / '{}.tiff'.format(prefix)
            model_fpath = dpath / '{} {}'.format(prefix, 'wz_bgs_model.tiff')
            mask_fpath = dpath / '{} {}'.format(prefix, 'wz_bgs_model_mask.png')
            delta_fpath = dpath / '{} {}'.format(prefix, 'wz_bgs_model_delta.tiff')
            pages.append([bf_fpath, model_fpath, mask_fpath, delta_fpath])
    if pages:
        rw.flipbook_pages = []
        rw.add_image_files_to_flipbook(pages, insert_viz_images)

def _apply_measure_transform(measure, im, measure_antimask, delta, mask):
    match = re.match(r'(model_mask_region_image|whole_image)_(hp_brenner_sum_of_squares|bp_brenner_sum_of_squares)_(max|min)', measure)
    delta[measure_antimask] = 0
    if not match:
        raise ValueError()
    if match.group(2) == 'hp_brenner_sum_of_squares':
        tim = HighpassBrenner(im.shape).metric(im)
    else:
        tim = BandpassBrenner(im.shape).metric(im)
    tim[measure_antimask] = 0
    if match.group(1) == 'model_mask_region_image':
        tim[mask==0] = 0
    return tim

class _NS:
    pass

def _makeMeasureInputSensitivityComparisonVizWorker(
        measure_antimask,
        measure_a,
        measure_b,
        bf_im_fpath,
        model_im_fpath,
        measure_a_im_fpath,
        measure_a_idx_delta,
        measure_a_delta_im_fpath,
        measure_a_mask_im_fpath,
        measure_b_im_fpath,
        measure_b_idx_delta,
        measure_b_delta_im_fpath,
        measure_b_mask_im_fpath
):
    r = _NS()
    r.bf_im, r.bf_im_fpath = freeimage.read(str(bf_im_fpath)), bf_im_fpath
    r.model_im = freeimage.read(str(model_im_fpath))
    r.measure_a_im, r.measure_a_im_fpath = freeimage.read(str(measure_a_im_fpath)), measure_a_im_fpath
    r.measure_a_idx_delta = measure_a_idx_delta
    r.measure_a_delta_im = freeimage.read(str(measure_a_delta_im_fpath))
    r.measure_a_mask_im = freeimage.read(str(measure_a_mask_im_fpath))
    r.measure_a_transformed_im = _apply_measure_transform(measure_a, r.measure_a_im, measure_antimask, r.measure_a_delta_im, r.measure_a_mask_im)
    r.measure_a_transformed_im_b = _apply_measure_transform(measure_b, r.measure_a_im, measure_antimask, r.measure_a_delta_im, r.measure_a_mask_im)
    r.measure_b_im, r.measure_b_im_fpath = freeimage.read(str(measure_b_im_fpath)), measure_b_im_fpath
    r.measure_b_idx_delta = measure_b_idx_delta
    r.measure_b_delta_im = freeimage.read(str(measure_b_delta_im_fpath))
    r.measure_b_mask_im = freeimage.read(str(measure_b_mask_im_fpath))
    r.measure_b_transformed_im = _apply_measure_transform(measure_b, r.measure_b_im, measure_antimask, r.measure_b_delta_im, r.measure_b_mask_im)
    r.measure_b_transformed_im_a = _apply_measure_transform(measure_a, r.measure_b_im, measure_antimask, r.measure_b_delta_im, r.measure_b_mask_im)
    return r

def makeMeasureInputSensitivityComparisonViz(rw, measure_a='model_mask_region_image_hp_brenner_sum_of_squares_max', measure_b='whole_image_hp_brenner_sum_of_squares_max'):
    measure_antimask = freeimage.read(str(DPATH / 'non-vignette.png')) == 0
    db = sqlite3.connect(str(DPATH / 'analysis/db.sqlite3'))
    rw.qt_object.layer_stack_flipbook.pages.append([Layer()])
    rw.qt_object.layer_stack_flipbook.pages.append([
        Layer(name='bf ffc'),
        Layer(name='model'),
        Layer(name='z stack image selected by {}'.format(measure_a)),
        Layer(name='z stack image selected by {} delta'.format(measure_a)),
        Layer(name='z stack image selected by {} mask'.format(measure_a)),
        Layer(name='z stack image selected and transformed by {}'.format(measure_a)),
        Layer(name='z stack image selected by {} and transformed by {}'.format(measure_a, measure_b)),
        Layer(name='z stack image selected by {}'.format(measure_b)),
        Layer(name='z stack image selected by {} delta'.format(measure_b)),
        Layer(name='z stack image selected by {} mask'.format(measure_b)),
        Layer(name='z stack image selected and transformed by {}'.format(measure_b)),
        Layer(name='z stack image selected by {} and transformed by {}'.format(measure_b, measure_a))])
    rw.qt_object.layer_stack_flipbook.pages[-1].name = "measure masking comparison: {} vs {}".format(measure_a, measure_b)
    rw.qt_object.layer_stack_flipbook.focused_page_idx = len(rw.qt_object.layer_stack_flipbook.pages) - 1
    rw.qt_object.layer_stack_flipbook_dock_widget.show()
    page_descs = []
    measure_a_extrema_fn = numpy.argmax if measure_a.endswith('max') else numpy.argmin
    measure_a_extrema_fn = lambda v, fn=measure_a_extrema_fn: int(fn(v))
    measure_b_extrema_fn = numpy.argmax if measure_b.endswith('max') else numpy.argmin
    measure_b_extrema_fn = lambda v, fn=measure_b_extrema_fn: int(fn(v))
    for time_point, well_idx, ma_idx_delta, mb_idx_delta in db.execute('select time_point, well_idx, {0}, {1} from focus_measure_vs_manual_idx_deltas order by {0} asc, {1} desc'.format(measure_a, measure_b)):
        z_stack_rows = list(
            db.execute(
                'select acquisition_name, is_focused, {0}, {1} from images where acquisition_name '
                'like "focus-__" and {0} is not NULL and {1} is not NULL and well_idx=? and time_point=? '
                'order by acquisition_name asc'.format(measure_a[:-4], measure_b[:-4]),
                (well_idx, time_point)
            )
        )
        focused_z_idx = int(numpy.argmax([z_stack_row[1] for z_stack_row in z_stack_rows]))
        measure_a_extrema_z_idx = measure_a_extrema_fn([z_stack_row[2] for z_stack_row in z_stack_rows])
        measure_b_extrema_z_idx = measure_b_extrema_fn([z_stack_row[3] for z_stack_row in z_stack_rows])
        page_descs.append(
            {
                'bf_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} bf_ffc.tiff'.format(time_point),
                'model_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} bf_ffc wz_bgs_model.tiff'.format(time_point),
                'measure_a_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc.tiff'.format(time_point, z_stack_rows[measure_a_extrema_z_idx][0]),
                'measure_a_idx_delta'
                    : int(numpy.abs(focused_z_idx - measure_a_extrema_z_idx)),
                'measure_a_delta_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc wz_bgs_model_delta.tiff'.format(time_point, z_stack_rows[measure_a_extrema_z_idx][0]),
                'measure_a_mask_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc wz_bgs_model_mask.png'.format(time_point, z_stack_rows[measure_a_extrema_z_idx][0]),
                'measure_b_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc.tiff'.format(time_point, z_stack_rows[measure_b_extrema_z_idx][0]),
                'measure_b_idx_delta'
                    : int(numpy.abs(focused_z_idx - measure_b_extrema_z_idx)),
                'measure_b_delta_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc wz_bgs_model_delta.tiff'.format(time_point, z_stack_rows[measure_b_extrema_z_idx][0]),
                'measure_b_mask_im_fpath'
                    : DPATH / '{:02}'.format(well_idx) / '{} {}_ffc wz_bgs_model_mask.png'.format(time_point, z_stack_rows[measure_b_extrema_z_idx][0])
            })
    tasks = [pool.submit(_makeMeasureInputSensitivityComparisonVizWorker, measure_antimask, measure_a, measure_b, **page_desc) for page_desc in page_descs]
    taskCount = len(tasks)
    taskN = 0
    pages = []
    page_names = []
    import gc
    while tasks:
        o = tasks.pop(0).result()
        pages.append(om.SignalingList([
            Image(o.bf_im, name=str(o.bf_im_fpath)),
            Image(o.model_im),
            Image(o.measure_a_im, name='{: 2} {}'.format(o.measure_a_idx_delta, o.measure_a_im_fpath)),
            Image(o.measure_a_delta_im),
            Image(o.measure_a_mask_im),
            Image(o.measure_a_transformed_im),
            Image(o.measure_a_transformed_im_b),
            Image(o.measure_b_im, name='{: 2} {}'.format(o.measure_b_idx_delta, o.measure_b_im_fpath)),
            Image(o.measure_b_delta_im),
            Image(o.measure_b_mask_im),
            Image(o.measure_b_transformed_im),
            Image(o.measure_b_transformed_im_a)
        ]))
        page_names.append('{: 2} | {: 2} ({})'.format(o.measure_a_idx_delta, o.measure_b_idx_delta, o.bf_im_fpath))
        taskN += 1
        print('{:%}'.format(taskN / taskCount))
        if taskN % 10 == 0:
            del o
            gc.collect()
    rw.flipbook_pages = []
    while pages:
        rw.flipbook_pages.extend(pages[:10])
        del pages[:10]
        gc.collect()
    for p, n in zip(reversed(rw.flipbook_pages), reversed(page_names)):
        p.name = n