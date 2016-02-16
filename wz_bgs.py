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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Willie Zhang

from concurrent.futures import ThreadPoolExecutor
import freeimage
import multiprocessing
import numpy
from pathlib import Path
import sqlite3
import sys
import threading
import time
from zplib.image import mask as zplib_image_mask

from _cppmod import image_stack_median

DPATHSTR = {
    'darwin' : '/Volumes/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3'
}.get(sys.platform, '/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3')
DPATH = Path(DPATHSTR)

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
            self.input_antimask = None
        else:
            self.input_antimask = input_mask == 0
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
        image_stack_median(self.context_carousel, self.model)

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
        delta = self.queryModelDelta(image) if delta is None else delta.astype(numpy.float32)
        if self.input_antimask is not None:
            delta[self.input_antimask] = 0
        threshold = numpy.percentile(numpy.abs(delta), 97.5).astype(numpy.float32)
        mask = delta >= threshold
        mask = zplib_image_mask.get_largest_object(mask)
#       mask[~zplib_image_mask.get_largest_object(mask)] = 0 # Remove dust from mask
        return mask

def processFlipbookPages(pages, temporal_radius=11):
    if not pages or not pages[0]:
        return [], None
    non_vignette = freeimage.read(str(DPATH / 'non-vignette.png'))
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

import os.path
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
        antimask = mask == 0
        image = image.astype(numpy.float32) # otherwise can get overflow in the squaring and summation
        # Exclude unmasked regions from edge detection output lest the masked region's border register as a large, spurious edge that may
        # dominate the measure and make it a measure of mask region border length rather than information density within the masked region
        # (depending on image content and spatial filter preprocessing).
        x_diffs = (image[2:, :] - image[:-2, :])**2
        x_diffs[antimask[:2558,:]] = 0
        y_diffs = (image[:, 2:] - image[:, :-2])**2
        y_diffs[antimask[:,:2158]] = 0
        return x_diffs.sum() + y_diffs.sum()

class MaskedFilteredBrenner(MaskedBrenner):
    def __init__(self, shape):
        super().__init__(shape)
        t0 = time.time()
        self.filter = fast_fft.SpatialFilter(shape, self.PERIOD_RANGE, precision=32, threads=multiprocessing.cpu_count(), better_plan=False)
        if time.time() - t0 > 0.5:
            fast_fft.store_plan_hints(FFTW_WISDOM)

    def metric(self, image, mask):
        filtered = self.filter.filter(image)
        return super().metric(filtered, mask)

class MaskedHighpassBrenner(MaskedFilteredBrenner):
    PERIOD_RANGE = (None, 10)

class MaskedBandpassBrenner(MaskedFilteredBrenner):
    PERIOD_RANGE = (60, 100)

class MaskedMultiBrenner(MaskedAutofocusMetric):
    def __init__(self, shape):
        super().__init__(shape)
        self.hp = MaskedHighpassBrenner(shape)
        self.bp = MaskedBandpassBrenner(shape)

    def metric(self, image, mask):
        return self.hp.metric(image, mask), self.bp.metric(image, mask)

def _computeFocusMeasures(bgs, im_fpath, measure_mask, compute_measures, write_models, write_deltas, write_masks):
    try:
        im = freeimage.read(str(im_fpath))
    except:
        return
    if bgs.model is not None:
        # NB: Model and delta are written as float32 tiffs
        try:
            if write_models:
                freeimage.write(
                    bgs.model,
                    str(im_fpath.parent / '{} wz_bgs_model.tiff'.format(im_fpath.stem)),
                    freeimage.IO_FLAGS.TIFF_DEFLATE)
            delta = numpy.abs(bgs.queryModelDelta(im))
            if write_deltas:
                freeimage.write(
                    delta,
                    str(im_fpath.parent / '{} wz_bgs_model_delta.tiff'.format(im_fpath.stem)),
                    freeimage.IO_FLAGS.TIFF_DEFLATE)
            mask = bgs.queryModelMask(im, delta)
            antimask = mask == 0
            if write_masks:
                freeimage.write(
                    (mask*255).astype(numpy.uint8),
                    str(im_fpath.parent / '{} wz_bgs_model_mask.png'.format(im_fpath.stem)))
        except:
            return
        if compute_measures:
            focus_measures = {}
            focus_measures['whole_image_hp_brenner_sum_of_squares'], focus_measures['whole_image_bp_brenner_sum_of_squares'] = MaskedMultiBrenner((2560, 2160)).metric(im, measure_mask)
            model_delta_squares = delta.astype(numpy.float64)**2
            model_delta_squares[~measure_mask] = 0
            focus_measures['model_delta_sum_of_squares'] = model_delta_squares.sum()
            focus_measures['model_mask_count'] = mask.sum()
            model_delta_squares[antimask] = 0
            focus_measures['model_mask_region_delta_sum_of_squares'] = model_delta_squares.sum()
            focus_measures['model_mask_region_image_hp_brenner_sum_of_squares'], focus_measures['model_mask_region_image_bp_brenner_sum_of_squares'] = MaskedMultiBrenner((2560,2160)).metric(im, mask)
            return focus_measures

pool = ThreadPoolExecutor()

def computeFocusMeasures(temporal_radius=11, update_db=True, write_models=False, write_deltas=False, write_masks=False):
    with sqlite3.connect(str(DPATH / 'analysis/db.sqlite3')) as db:
        db.row_factory = sqlite3.Row
        non_vignette = freeimage.read(str(DPATH / 'non-vignette.png')) != 0
        vignette = non_vignette == 0
        image_count = list(db.execute('select count() from images'))[0]['count()']
        positions = [row['well_idx'] for row in db.execute('select well_idx from wells where did_hatch')]
        position_bgss = {position : WzBgs(2560, 2160, temporal_radius, non_vignette) for position in positions}
        time_points = [row['name'] for row in db.execute('select name from time_points')]
        image_idx = 0
        for time_point in time_points:
            print(time_point)
            for position in positions:
                print('', position)
                acquisition_names = [row['acquisition_name'] for row in db.execute('select acquisition_name from images where well_idx=? and time_point=?', (position, time_point))]
                if acquisition_names:
                    bgs = position_bgss[position]
                    tasks = []
                    for acquisition_name in acquisition_names:
                        im_fpath = DPATH / '{:02}'.format(position) / '{} {}_ffc.png'.format(time_point, acquisition_name)
                        tasks.append(pool.submit(lambda an=acquisition_name, fn=_computeFocusMeasures, args=(bgs, im_fpath, non_vignette, update_db, write_models, write_deltas, write_masks): (an, fn(*args))))
                    for task in tasks:
                        acquisition_name, focus_measures = task.result()
                        if focus_measures is not None:
                            measure_names = sorted(focus_measures.keys())
                            q = 'update images set ' + ', '.join('{}=?'.format(measure_name) for measure_name in measure_names)
                            q+= ' where well_idx=? and time_point=? and acquisition_name=?'
                            v = [float(focus_measures[measure_name]) for measure_name in measure_names]
                            v.extend((position, time_point, acquisition_name))
                            list(db.execute(q, v))
                        image_idx += 1
                        print('  {:<10} {:%}'.format(acquisition_name, image_idx / image_count))
                    try:
                        im = freeimage.read(str(DPATH / '{:02}'.format(position) / '{} bf_ffc.png'.format(time_point)))
                        bgs.updateModel(im)
                    except:
                        pass
                    db.commit()

def computeFocusMeasureBestVsFocusedIdxDeltas():
    db = sqlite3.connect(str(DPATH / 'analysis/db.sqlite3'))
    measure_names = [d[0] for d in db.execute('select * from images').description][5:]
    time_point_well_idxs = list(db.execute(
        'select time_point, well_idx from ('
	    '   select time_point, well_idx, sum(is_focused) as sif from images where acquisition_name!="bf" group by time_point, well_idx'
        ') where sif == 1'))
    for time_point, well_idx in time_point_well_idxs:
        print(time_point, well_idx)
        for measure_name in measure_names:
            rows = list(db.execute('select is_focused, {} from images where time_point=? and well_idx=? and acquisition_name!="bf" order by acquisition_name'.format(measure_name), (time_point, well_idx)))
            if all(row[1] is not None for row in rows):
                # print(measure_name, time_point, well_idx, numpy.argmin([v[1] for v in rows]), numpy.argmax([v[0] for v in rows]))
                focused_idx = int(numpy.argmax([row[0] for row in rows]))
                measure_min_idx = int(numpy.argmin([row[1] for row in rows]))
                measure_min_idx_delta = abs(focused_idx - measure_min_idx)
                measure_max_idx = int(numpy.argmax([row[1] for row in rows]))
                measure_max_idx_delta = abs(focused_idx - measure_max_idx)
                if list(db.execute('select count() from focus_measure_vs_manual_idx_deltas where time_point=? and well_idx=?', (time_point, well_idx)))[0][0] == 0:
                    list(db.execute('insert into focus_measure_vs_manual_idx_deltas (time_point, well_idx) values(?, ?)', (time_point, well_idx)))
                list(db.execute('update focus_measure_vs_manual_idx_deltas set {0}_min=?, {0}_max=? where time_point=? and well_idx=?'.format(measure_name), (measure_min_idx_delta, measure_max_idx_delta, time_point, well_idx)))
    db.commit()

def makeFocusMeasureBestVsFocusedIdxDeltaHistograms():
    import matplotlib.pyplot as plt
    db = sqlite3.connect(str(DPATH / 'analysis/db.sqlite3'))
    dbq = db.execute('select * from focus_measure_vs_manual_idx_deltas')
    measure_names = [dbqd[0] for dbqd in dbq.description][2:]
    data = numpy.array(list(dbqr[2:] for dbqr in dbq))
    plt.ion()
    fig = plt.figure()
    fig_idx = fig.number
    fig_rowcount = 2
    fig_colcount = len(measure_names) / fig_rowcount
    for measure_idx in range(len(measure_names)):
        plt.subplot(fig_rowcount, fig_colcount, measure_idx+1)
        plt.hist(data[:, measure_idx], bins=5)
        label = measure_names[measure_idx]
        # if len(xlabel) > 20:
        #     xlabel = xlabel[:20] + '\n' + xlabel[20:]
        plt.ylabel(label)

class WeightedMaskedAutofocusMetric:
    def __init__(self, shape):
        pass

    def metric(self, image, antimask, weights):
        raise NotImplementedError()

class WeightedMaskedBrenner(WeightedMaskedAutofocusMetric):
    def metric(self, image, antimask, weights):
        if image.dtype.type is not numpy.float64:
            image = image.astype(numpy.float64) # otherwise can get overflow when applying logistic_sigmoid weights
        # Exclude unmasked regions from edge detection output lest the masked region's border register as a large, spurious edge that may
        # dominate the measure and make it a measure of mask region border length rather than information density within the masked region
        # (depending on image content and spatial filter preprocessing).
        x_diffs = (image[2:, :] - image[:-2, :])**2
        x_diffs *= weights[:2558,:]
        x_diffs[antimask[:2558,:]] = 0
        y_diffs = (image[:, 2:] - image[:, :-2])**2
        y_diffs *= weights[:,:2158]
        y_diffs[antimask[:,:2158]] = 0
        return x_diffs.sum() + y_diffs.sum()

class WeightedMaskedFilteredBrenner(WeightedMaskedBrenner):
    def __init__(self, shape):
        super().__init__(shape)
        t0 = time.time()
        self.filter = fast_fft.SpatialFilter(shape, self.PERIOD_RANGE, precision=32, threads=multiprocessing.cpu_count(), better_plan=False)
        if time.time() - t0 > 0.5:
            fast_fft.store_plan_hints(FFTW_WISDOM)

    def metric(self, image, antimask, weights):
        filtered = self.filter.filter(image)
        return super().metric(filtered, antimask, weights)

class WeightedMaskedHighpassBrenner(WeightedMaskedFilteredBrenner):
    PERIOD_RANGE = (None, 10)

__weighted_masked_highpass_brenner_instances = dict()
def getWeightedMaskedHighpassBrennerInstance(shape):
    if shape in __weighted_masked_highpass_brenner_instances:
        return __weighted_masked_highpass_brenner_instances[shape]
    else:
        __weighted_masked_highpass_brenner_instances[shape] = inst = WeightedMaskedHighpassBrenner(shape)
        return inst

class WeightedMaskedBandpassBrenner(WeightedMaskedFilteredBrenner):
    PERIOD_RANGE = (60, 100)

class WeightedMaskedMultiBrenner(WeightedMaskedAutofocusMetric):
    def __init__(self, shape):
        super().__init__(shape)
        self.hp = WeightedMaskedHighpassBrenner(shape)
        self.bp = WeightedMaskedBandpassBrenner(shape)

    def metric(self, image, antimask, weights):
        return self.hp.metric(image, antimask, weights), self.bp.metric(image, antimask, weights)

def logistic_sigmoid(x, x0=0, k=1, L=1):
    return (1 / (1 + numpy.exp(-k * (x - x0) ) )).astype(numpy.float64)

def _computeSigmoidWeightedMeasures(db_lock, update_db, measure_antimask, x0, k, L):
    weightedMaskedHighpassBrenner = getWeightedMaskedHighpassBrennerInstance((2560, 1600))
    with db_lock, sqlite3.connect(str(DPATH / 'analysis/db.sqlite3')) as db:
        column_name = 'x0:{},k:{}'.format(x0, k)
        time_point_well_idxs = list(db.execute(
            'select time_point, well_idx from ('
            '   select time_point, well_idx, sum(is_focused) as sif from images where acquisition_name!="bf" group by time_point, well_idx'
            ') where sif == 1'))
    for time_point, well_idx in time_point_well_idxs:
        with db_lock, sqlite3.connect(str(DPATH / 'analysis/db.sqlite3')) as db:
            acquisition_names = [row[0] for row in db.execute('select acquisition_name from images where well_idx=? and time_point=?', (well_idx, time_point))]
        results = []
        for acquisition_name in acquisition_names:
            delta_fpath = DPATH / '{:02}'.format(well_idx) / '{} {}_ffc wz_bgs_model_delta.tiff'.format(time_point, acquisition_name)
            if not delta_fpath.exists():
                continue
            image = freeimage.read(str(DPATH / '{:02}'.format(well_idx) / '{} {}_ffc.png'.format(time_point, acquisition_name)))
            delta = freeimage.read(str(delta_fpath))
            weights = logistic_sigmoid(delta, x0, k, L)
            results.append(float(weightedMaskedHighpassBrenner.metric(image, measure_antimask, weights)))
        if not results or not update_db:
            continue
        with db_lock, sqlite3.connect(str(DPATH / 'analysis/db.sqlite3')) as db:
            for result, acquisition_name in zip(results, acquisition_names):
                image_id = list(db.execute(
                    'select image_id from images where time_point=? and well_idx=? and acquisition_name=?',
                    (time_point, well_idx, acquisition_name)))[0][0]
                if list(db.execute('select count(*) from sigmoids where image_id=?', (image_id,)))[0][0] == 0:
                    list(db.execute('insert into sigmoids (image_id, "{}") values (?, ?)'.format(column_name), (image_id, result)))
                else:
                    list(db.execute('update sigmoids set "{}"=? where image_id=?'.format(column_name), (result, image_id)))
            db.commit()

def computeSigmoidWeightedMeasures(update_db=True, logistic_sigmoid_parameter_sets=None):
    db_lock = threading.Lock()
    measure_antimask = freeimage.read(str(DPATH / 'non-vignette.png')) == 0
    with db_lock, sqlite3.connect(str(DPATH / 'analysis/db.sqlite3')) as db:
        if logistic_sigmoid_parameter_sets is None:
            logistic_sigmoid_parameter_sets = [
                {"x0":x0, "k":k, "L":1, "measure_antimask":measure_antimask, "db_lock":db_lock, "update_db":update_db}
                for x0 in numpy.linspace(2000, 20000, 20, dtype=numpy.float32) for k in numpy.linspace(1/100, 1/1000, 20, dtype=numpy.float32)
            ]
        if update_db:
            # If sigmoids table exists, drop it
            if list(db.execute('select name from sqlite_master where type="table" and name="sigmoids"')):
                list(db.execute('drop table sigmoids'))
            # Create sigmoids table with desired columns
            sigmoid_columns = ['"image_id" integer not null']
            sigmoid_columns += ['"x0:{},k:{}" REAL'.format(x0, k) for x0, k in ((p["x0"], p["k"]) for p in logistic_sigmoid_parameter_sets)]
            sigmoid_columns.append('FOREIGN KEY(`image_id`) REFERENCES images ( image_id )')
            list(db.execute('create table "sigmoids" ({})'.format(',\n'.join(sigmoid_columns))))
            db.commit()
    # _computeSigmoidWeightedMeasures(**logistic_sigmoid_parameter_sets[0])
    tasks = [pool.submit(lambda a: _computeSigmoidWeightedMeasures(**a), p) for p in logistic_sigmoid_parameter_sets]
    for task in tasks:
        try:
            r = task.result()
        except Exception as e:
            print(e)

def _getLogisticSigmoidParameterSets(db):
    sig_row_0 = list(db.execute('select * from sigmoids'))[0][1:]
    sig_col_names = [d[0] for d in db.execute('select * from sigmoids').description][1:]
    # discard empty columns
    sig_col_names = [sig_col_name for sig_col_name, sig_col_val in zip(sig_col_names, sig_row_0) if sig_col_val is not None]
    # parse sigmoid parameters out of column names
    return [{'x0':float(x0[1]), 'k':float(k[1])} for x0, k in ((vv.split(':') for vv in v.split(',')) for v in sig_col_names)], sig_col_names

def computeSigmoidWeightedMeasuresVsFocusedIdxDeltas():
    db = sqlite3.connect(str(DPATH / 'analysis/db_.sqlite3'))
    logistic_sigmoid_parameter_sets, sigmoid_idx_delta_column_names = _getLogisticSigmoidParameterSets(db)
    if list(db.execute('select name from sqlite_master where type="table" and name="sigmoid_weighted_focus_measure_vs_manual_idx_deltas"')):
        list(db.execute('drop table sigmoid_weighted_focus_measure_vs_manual_idx_deltas'))
    sigmoid_idx_delta_columns = ['"well_idx" integer not null']
    sigmoid_idx_delta_columns+= ['"time_point" text not null']
    sigmoid_idx_delta_columns+= ['"{0}_min" integer,\n"{0}_max" integer'.format(sigmoid_idx_delta_column_name) for sigmoid_idx_delta_column_name in sigmoid_idx_delta_column_names]
    sigmoid_idx_delta_columns+= ['foreign key("well_idx") references images ( well_idx )']
    sigmoid_idx_delta_columns+= ['foreign key("time_point") references images ( time_point )']
    list(db.execute('create table "sigmoid_weighted_focus_measure_vs_manual_idx_deltas" ({})'.format(',\n'.join(sigmoid_idx_delta_columns))))
    db.commit()
    time_point_well_idxs = list(db.execute(
        'select time_point, well_idx from ('
	    '   select time_point, well_idx, sum(is_focused) as sif from images where acquisition_name!="bf" group by time_point, well_idx'
        ') where sif == 1'))
    for time_point, well_idx in time_point_well_idxs:
        print(time_point, well_idx)
        image_ids, is_focuseds = zip(*list(db.execute(
            'select image_id, is_focused from images where time_point=? and well_idx=? and acquisition_name!="bf" order by acquisition_name',
            (time_point, well_idx))))
        focused_idx = int(numpy.argmax(is_focuseds))
        for params, name in zip(logistic_sigmoid_parameter_sets, sigmoid_idx_delta_column_names):
            sigmoid_rows = [list(db.execute('select "{}" from sigmoids where image_id=?'.format(name), (image_id,))) for image_id in image_ids]
            if all(sigmoid_row and sigmoid_row[0][0] is not None for sigmoid_row in sigmoid_rows):
                measure_values = [sigmoid_row[0][0] for sigmoid_row in sigmoid_rows]
                # print(measure_values)
                measure_min_idx = int(numpy.argmin(measure_values))
                measure_min_idx_delta = abs(focused_idx - measure_min_idx)
                measure_max_idx = int(numpy.argmax(measure_values))
                measure_max_idx_delta = abs(focused_idx - measure_max_idx)
                # print(name, measure_min_idx_delta, measure_max_idx_delta)
                if list(db.execute('select count() from sigmoid_weighted_focus_measure_vs_manual_idx_deltas where time_point=? and well_idx=?', (time_point, well_idx)))[0][0] == 0:
                    list(db.execute('insert into sigmoid_weighted_focus_measure_vs_manual_idx_deltas (time_point, well_idx) values(?, ?)', (time_point, well_idx)))
                list(db.execute(
                    'update sigmoid_weighted_focus_measure_vs_manual_idx_deltas set "{0}_min"=?, "{0}_max"=? where time_point=? and well_idx=?'.format(name),
                    (measure_min_idx_delta, measure_max_idx_delta, time_point, well_idx)))
        db.commit()

def makeSigmoidWeightedFocusMeasureBestVsFocusedIdxDeltaHistograms():
    import matplotlib.pyplot as plt
    db = sqlite3.connect(str(DPATH / 'analysis/db_.sqlite3'))
    dbq = db.execute('select * from sigmoid_weighted_focus_measure_vs_manual_idx_deltas')
    names = [dbqd[0] for dbqd in dbq.description][2:]
    dbq = db.execute('select * from sigmoid_weighted_focus_measure_vs_manual_idx_deltas')
    data = numpy.array(list(dbqr[2:] for dbqr in dbq))
    plt.ioff()
    fig = plt.figure()
    dpi = fig.get_dpi()
    fig.set_size_inches(400/dpi, 500/dpi)
    for idx, name in enumerate(names):
        if sum(v is not None for v in data[:, idx]) < 2:
            continue
        plt.hist([v for v in data[:, idx] if v is not None], bins=5)
        label = names[idx]
        # if len(xlabel) > 20:
        #     xlabel = xlabel[:20] + '\n' + xlabel[20:]
        plt.ylabel(label)
        fig.savefig('{}.png'.format(name.replace(':', '_').replace(',', '-')))
        fig.clear()

# def combineSigmoidWeightedFocusMeasureBestVsFocusedIdxDeltaHistograms():

if __name__ == '__main__':
    computeSigmoidWeightedMeasures()