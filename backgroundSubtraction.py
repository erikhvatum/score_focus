# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:06:30 2015

@author: Willie
"""

import freeimage
import numpy as np
from pathlib import Path
import shutil
from zplib.image import mask as zplib_image_mask


def overallBackgroundSubtract(data_dpath, match_glob, temporal_radius, save_dpath, save_dpath2 = '', save_dpath3 = '', demonstration_mode = False):
    '''
    Do background subtraction to find worms. This uses only past data, masking out the worms to create a background that won't disappear once the worm stops  moving.
    '''
    data_dpath = Path(data_dpath)
    save_dpath = Path(save_dpath)
    if save_dpath2:
        save_dpath2 = Path(save_dpath2)
    if save_dpath3:
        save_dpath3 = Path(save_dpath3)
    my_file_fpaths = sorted(data_dpath.glob(match_glob))

    # Initialize my special background context.
    temp_dpath = save_dpath / 'temp'
    if not temp_dpath.exists():
        temp_dpath.mkdir(parents=True)
    for i in range(0, temporal_radius):
        shutil.copy(str(my_file_fpaths[i]), str(temp_dpath / my_file_fpaths[i].name))

    # Run the actual simple subtraction, saving out masked files.
    context_files = [freeimage.read(str(my_file_fpaths[j])) for j in range(0, temporal_radius)]
    for i in range(temporal_radius, len(my_file_fpaths)):
        real_raw_file = freeimage.read(str(my_file_fpaths[i]))
        raw_file = real_raw_file.copy()     
        context_files.append(raw_file)
        (foreground_file, background_file) = simple_running_median_subtraction(raw_file, context_files)
        
        thresholded_mask = percentile_floor(foreground_file, threshold_proportion = 0.975)
        final_mask = clean_dust_and_holes(thresholded_mask)

        raw_file[final_mask.astype('bool')] = background_file[final_mask.astype('bool')]        

        if demonstration_mode:
            freeimage.write(real_raw_file, str(save_dpath / my_file_fpaths[i].name))
            freeimage.write(background_file, str(save_dpath2 / my_file_fpaths[i].name))
            freeimage.write(final_mask, str(save_dpath3 / my_file_fpaths[i].name))

        if not demonstration_mode:
            freeimage.write(raw_file, str(temp_dpath / my_file_fpaths[i].name))
            freeimage.write(final_mask,str(save_dpath / my_file_fpaths[i].name))
    
        context_files = context_files[1:]

    return

def OLDoverallBackgroundSubtract(data_dir, match_string, temporal_radius, save_dir):
    '''
    Do background subtraction to find worms.
    '''
    my_files = sorted(os.listdir(data_dir))
    my_files = [a_file for a_file in my_files if match_string == a_file.split('_')[-1]]

    # Intialize my special background context.
    temp_folder = save_dir + '\\' + 'temp'
    try:
        os.stat(temp_folder)
    except: 
        os.mkdir(temp_folder)
    for i in range(0, temporal_radius):
        shutil.copy(data_dir + '\\' + my_files[i], temp_folder + '\\' + my_files[i])
        
    # Run the actual simple subtraction, saving out masked files.
    for i in range(temporal_radius, len(my_files)-temporal_radius):
        #context_files = [freeimage.read(data_dir + '\\' + my_files[j]) for j in range(i-temporal_radius, i+temporal_radius+1)]
        context_files = [freeimage.read(data_dir + '\\' + my_files[j]) for j in range(i-temporal_radius, i+1)]
        raw_file = freeimage.read(data_dir + '\\' + my_files[i])
        (simple_foreground_file, background_file) = simple_running_median_subtraction(raw_file, context_files)
        
        thresholded_mask = percentile_floor(simple_foreground_file, threshold_proportion = 0.975)
        final_mask = clean_dust_and_holes(thresholded_mask)

        raw_file[final_mask.astype('bool')] = background_file[final_mask.astype('bool')]        
        freeimage.write(raw_file, temp_folder + '\\' + my_files[i])

    # Fill in remaining tail files.
    for i in range(len(my_files)-temporal_radius, len(my_files)):
        shutil.copy(data_dir + '\\' + my_files[i], temp_folder + '\\' + my_files[i])

    # Now let's do it for real!
    for i in range(temporal_radius, len(my_files)-temporal_radius):
        context_files = [freeimage.read(temp_folder + '\\' + my_files[j]) for j in range(i-temporal_radius, i+temporal_radius+1)]
        raw_file = freeimage.read(data_dir + '\\' + my_files[i])
        (simple_foreground_file, background_file) = simple_running_median_subtraction(raw_file, context_files)
        
        thresholded_pic = percentile_floor(simple_foreground_file, threshold_proportion = 0.975)
        final_mask = clean_dust_and_holes(thresholded_pic)

        freeimage.write(final_mask, save_dir + '\\' + my_files[i])


    return


def clean_dust_and_holes(dusty_pic):
    '''
    Picks out the largest object in dusty_mask and fills in its holes, returning cleaned_mask.
    '''
    my_dtype = dusty_pic.dtype
    dust_mask = np.invert(zplib_image_mask.get_largest_object(dusty_pic))       
    dusty_pic[dust_mask] = 0
    cleaned_mask = simple_floor(zplib_image_mask.fill_small_area_holes(dusty_pic, 90000).astype(my_dtype), 1)
    return cleaned_mask

def simple_floor(focal_image, threshold_value):
    ''' 
    Takes a grayscale focal image (in the form of a numpy array), and sets to zero (black) all values below the threshold value.
    '''
    max_value = -1
    binary_image = focal_image.copy()
    binary_image[binary_image < threshold_value] = 0 
    binary_image[binary_image >= threshold_value] = max_value 
    return binary_image
    
def percentile_floor(focal_image, threshold_proportion):
    '''
    Takes a grayscale focal image (in the form of a numpy array), and sets to zero (black) all values below the percentile indicated by threshold_proportion.
    '''
    max_value = -1
    binary_image = focal_image.copy()
    threshold_value = int(np.percentile(binary_image, threshold_proportion*100))
    binary_image[binary_image < threshold_value] = 0 
    binary_image[binary_image >= threshold_value] = max_value   
    return binary_image

def simple_running_median_subtraction(focal_image, background_images):
    ''' 
    Takes a focal image and a list of background images (grayscale, in the form of numpy arrays), and returns the focal image with the background subtracted. This simply takes the median value of each pixel to construct a background.   
    '''
    median_image = median_image_from_list(background_images)
    foreground_only = abs(focal_image.astype('int16') - median_image.astype('int16')).astype('uint16')
    return (foreground_only, median_image)

def median_image_from_list(background_images):
    ''' 
    Takes a list of background images (grayscale, in the form of numpy arrays), and returns an image constructed by taking the median value of each pixel.
    '''
    big_array = np.array([background_image for background_image in background_images])
    median_image = np.median(big_array, axis = 0).astype('uint16')
    return median_image


def main():
    return

if __name__ == "__main__":
    main()
