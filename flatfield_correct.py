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

import argparse
import freeimage
import json
import numpy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def flatfield_correct(im_fpath, ff_fpath, ffc_fpath):
    if not im_fpath.exists():
        return False, 'skipping "{}" (file not found)'.format(str(im_fpath))
    if not ff_fpath.exists():
        return False, 'skipping "{}" (flatfield reference image file "{}" not found)'.format(str(ff_fpath))
    try:
        im = freeimage.read(str(im_fpath))
        ff = freeimage.read(str(ff_fpath))
        ffc = im.astype(numpy.float32) * ff
        ffc *= 65535.0 / float(numpy.percentile(ffc, 90))
        ffc[ffc < 0] = 0
        ffc[ffc > 65535] = 65535
        freeimage.write(ffc.astype(numpy.uint16), str(ffc_fpath), freeimage.IO_FLAGS.PNG_Z_BEST_SPEED)
    except Exceptions as e:
        return False, 'exception while correcting "{}": {}'.format(str(im_fpath), e)
    return True, '{} done'.format(str(im_fpath))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Apply flatfield correction to acquired images and store the result alongside the originals with "_ffc" appended to the filename.')
    parser.add_argument(
        '--experiment_dpath',
        type=Path,
        help='Directory containing experiment_metadata.json.  Default: "/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3"',
        default=Path('/mnt/bulkdata/Sinha_Drew/2015.11.13_ZPL8Prelim3'),
        required=False)
    parser.add_argument(
        '--positions',
        type=int,
        nargs='*',
        help='Indexes of positions with images to correct.  If an empty string or whitespace is supplied for this argument, or if the argument is not supplied, '
             'flatfield_correct.py operates on all positions.',
        required=False)
    parser.add_argument(
        '--suffixes',
        type=str,
        nargs='+',
        help='The suffixes of the images that are to be corrected.  ex: "bf" "gfp"',
        required=True)
    args = parser.parse_args()
    experiment_dpath = Path(args.experiment_dpath)
    assert experiment_dpath.exists()
    experiment_metadata_fpath = experiment_dpath / 'experiment_metadata.json'
    assert experiment_metadata_fpath.exists()
    with experiment_metadata_fpath.open() as f:
        experiment_metadata = json.load(f)
    assert "positions" in experiment_metadata
    all_positions = sorted(int(pstr) for pstr in experiment_metadata["positions"].keys())
    positions = args.positions
    if positions is None:
        positions = all_positions
    else:
        assert all(p in all_positions for p in positions)
    suffixes = args.suffixes
    assert "timepoints" in experiment_metadata
    timepoints = experiment_metadata['timepoints']

    with ProcessPoolExecutor() as process_pool:
        futes = \
        [
            process_pool.submit(
                flatfield_correct,
                experiment_dpath / '{:02}'.format(position) / '{} {}.png'.format(timepoint, suffix),
                experiment_dpath / 'calibrations' / '{} {}_flatfield.tiff'.format(timepoint, suffix if suffix=='bf' else 'fl'),
                experiment_dpath / '{:02}'.format(position) / '{} {}_ffc.png'.format(timepoint, suffix))
            for timepoint in timepoints
            for position in positions
            for suffix in suffixes
        ]

        for idx, (result, msg) in enumerate(fute.result() for fute in futes):
            print('{}% {}'.format(int(1600*(idx+1)/len(futes))/16, msg))
