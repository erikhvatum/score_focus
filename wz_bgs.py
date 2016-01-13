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

def image_stack_median(image_stack):
    r = numpy.median(image_stack, axis=2)
    if r.dtype.type is not numpy.float32:
        r = r.astype(numpy.float32)
    return r

class WzBgs:
    def __init__(self, width, height, temporal_radius):
        assert all(int(d) == d and d > 0 for d in (width, height, temporal_radius)),\
            "WzBgs.__init__(self, width, height, temporal_radius): "\
            "width, height, temporal_radius must be positive integers"
        self.context_carrousel = numpy.ndarray(
            (width, height, temporal_radius),
            strides=(temporal_radius*4, width*temporal_radius*4, 4),
            dtype=numpy.float32)
        self.clear()

    def clear(self):
        self.context_id = 0
        self.context_idx = 0
        self.next_context_image_idx = 0
        self.model = None

    def updateModel(self, image, mask=None):
        temporal_radius = self.context_carrousel.shape[2]
        assert image.shape == self.context_carrousel.shape[:2]
        if self.context_id < temporal_radius:
            # Insufficient context was available for background model to be constructed.  Therefore, we cannot discern the foregound,
            # so we simply enter the input into the context.
            self.context_carrousel[..., self.context_idx] = image
            self.context_id += 1
            self.context_idx += 1
            if self.context_id == temporal_radius:
                self.context_idx = 0
                # Entering the current input completed the context, permitting construction of a background model that future calls
                # will use for foreground discernment.
                self.model = image_stack_median(self.context_carrousel)
        else:
            if mask is None:
                mask = self.queryModelMask(image)
            self.context_carrousel[..., self.context_idx] = image
            # Replace the region identified as the foreground with the corresponding region of the background model
            self.context_carrousel[..., self.context_idx][mask] = self.model[mask]
            self.context_id += 1
            self.context_idx += 1
            if self.context_idx == temporal_radius:
                self.context_idx = 0
            self.model = image_stack_median(self.context_carrousel)
            return mask

    def queryModelDelta(self, image):
        if self.model is None:
            return
        assert image.shape == self.context_carrousel.shape[:2]
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
                r.append(wzBgs.model)
                if mask is not None:
                    r.append((mask*255).astype(numpy.uint8))
            ret.append(r)
            print('{} / {}'.format(idx+1, len(pages)))
    except KeyboardInterrupt:
        pass
    return ret

#def overallBackgroundSubtract(data_dpath, match_glob, temporal_radius, save_dpath, save_dpath2 = '', save_dpath3 = '', demonstration_mode = False):
#    '''
#    Do background subtraction to find worms. This uses only past data, masking out the worms to create
#    a background that won't disappear once the worm stops  moving.
#    '''
#    data_dpath = Path(data_dpath)
#    save_dpath = Path(save_dpath)
#    if save_dpath2:
#        save_dpath2 = Path(save_dpath2)
#    if save_dpath3:
#        save_dpath3 = Path(save_dpath3)
#    my_file_fpaths = sorted(data_dpath.glob(match_glob))
#
#    # Initialize my special background context.
#    temp_dpath = save_dpath / 'temp'
#    if not temp_dpath.exists():
#        temp_dpath.mkdir(parents=True)
#    for i in range(0, temporal_radius):
#        shutil.copy(str(my_file_fpaths[i]), str(temp_dpath / my_file_fpaths[i].name))
#
#    # Run the actual simple subtraction, saving out masked files.
#    context_files = [freeimage.read(str(my_file_fpaths[j])) for j in range(0, temporal_radius)]
#    for i in range(temporal_radius, len(my_file_fpaths)):
#        real_raw_file = freeimage.read(str(my_file_fpaths[i]))
#        raw_file = real_raw_file.copy()
#        context_files.append(raw_file)
#        (foreground_file, background_file) = simple_running_median_subtraction(raw_file, context_files)
#
#        thresholded_mask = percentile_floor(foreground_file, threshold_proportion = 0.975)
#        final_mask = clean_dust_and_holes(thresholded_mask)
#
#        raw_file[final_mask.astype('bool')] = background_file[final_mask.astype('bool')]
#
#        if demonstration_mode:
#            freeimage.write(real_raw_file, str(save_dpath / my_file_fpaths[i].name))
#            freeimage.write(background_file, str(save_dpath2 / my_file_fpaths[i].name))
#            freeimage.write(final_mask, str(save_dpath3 / my_file_fpaths[i].name))
#
#        if not demonstration_mode:
#            freeimage.write(raw_file, str(temp_dpath / my_file_fpaths[i].name))
#            freeimage.write(final_mask,str(save_dpath / my_file_fpaths[i].name))
#
#        context_files = context_files[1:]
#
#    return
#
#def clean_dust_and_holes(dusty_pic):
#    '''
#    Picks out the largest object in dusty_mask and fills in its holes, returning cleaned_mask.
#    '''
#    my_dtype = dusty_pic.dtype
#    dust_mask = numpy.invert(zplib_image_mask.get_largest_object(dusty_pic))
#    dusty_pic[dust_mask] = 0
#    cleaned_mask = simple_floor(zplib_image_mask.fill_small_area_holes(dusty_pic, 90000).astype(my_dtype), 1)
#    return cleaned_mask
#
#def simple_floor(focal_image, threshold_value):
#    '''
#    Takes a grayscale focal image (in the form of a numpy array), and sets to zero (black) all values below the threshold value.
#    '''
#    max_value = -1
#    binary_image = focal_image.copy()
#    binary_image[binary_image < threshold_value] = 0
#    binary_image[binary_image >= threshold_value] = max_value
#    return binary_image
#
#def percentile_floor(focal_image, threshold_proportion):
#    '''
#    Takes a grayscale focal image (in the form of a numpy array), and sets to zero (black) all values below the percentile indicated by threshold_proportion.
#    '''
#    max_value = -1
#    binary_image = focal_image.copy()
#    threshold_value = int(numpy.percentile(binary_image, threshold_proportion*100))
#    binary_image[binary_image < threshold_value] = 0
#    binary_image[binary_image >= threshold_value] = max_value
#    return binary_image
#
#def simple_running_median_subtraction(focal_image, background_images):
#    '''
#    Takes a focal image and a list of background images (grayscale, in the form of numpy arrays), and returns the focal image with the background subtracted. This simply takes the median value of each pixel to construct a background.
#    '''
#    median_image = median_image_from_list(background_images)
#    foreground_only = abs(focal_image.astype('int16') - median_image.astype('int16')).astype('uint16')
#    return (foreground_only, median_image)
#
#def median_image_from_list(background_images):
#    '''
#    Takes a list of background images (grayscale, in the form of numpy arrays), and returns an image constructed by taking the median value of each pixel.
#    '''
#    big_array = numpy.array([background_image for background_image in background_images])
#    median_image = numpy.median(big_array, axis = 0).astype('uint16')
#    return median_image
