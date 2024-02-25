################################################
# Nearest-neighbor
# Bilinear
# Bicubic
# Lanczos
# Pixel area relation
# https://www.semanticscholar.org/paper/Comparison-of-Commonly-Used-Image-Interpolation-Han/47408d4854044f900d6ee757ca322de983d702ce
# https://stackoverflow.com/questions/3112364/how-do-i-choose-an-image-interpolation-method-emgu-opencv
# https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
################################################

################################################
# Dataset from Nick Yue
# Special thanks to nick :)
################################################

################################################
#
# EENG30009 Individual Research Project 3
# TOPIC: Single-image Super-resolution
#
# Yumu Xie
# University of Bristol
#
################################################

################################################
# Programming environment: WSL Linux subsystem of Windows
# Python version: 3.7.9
# Python setup is based on 'pyenv' (Simple Python version management)
# https://github.com/pyenv/pyenv
#
# OpenCV, Matplotlib, Pandas and Numpy packages are required in this script
# Instruction of installing packages:
# pip install opencv-python
# pip install matplotlib
# pip install pandas
# pip install numpy
#
# The scale factor is 2 and 4 respectively
# For example, an input image has resolution (x, y)
# After executing this script, (x/2, y/2) and (x/4, y/4) images will be produced
#
# The purpose of this script is to execute pre-processing interpolation methods for downscaling images / downsampling images
# [input_image] -> [downsampling/resampling] -> [output_image]
################################################

################################################
# Please ensure that the directory which contains input images is named as "input"
# Please create 10 folders before executing this script, otherwise output images would not be saved
# These folders should have exact same name below:
# "nearest_neighbor_half"
# "nearest_neighbor_quarter"
# "bilinear_half"
# "bilinear_quarter"
# "bicubic_half"
# "bicubic_quarter"
# "lanczos_half"
# "lanczos_quarter"
# "pixel_area_relation_half"
# "pixel_area_relation_quarter"
################################################

# python downsampling_resampling.py

################################################
# Downsampling + Resampling
import cv2
import logging
import time
################################################

################################################
# Table
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
################################################

input_image_files = ['input/001.jpg',
                     'input/002.jpg',
                     'input/003.jpg',
                     'input/004.jpg',
                     'input/005.jpg',
                     'input/006.jpg',
                     'input/007.jpg',
                     'input/008.jpg',
                     'input/009.jpg',
                     'input/010.jpg',
                     'input/011.jpg',
                     'input/012.jpg',
                     'input/013.jpg',
                     'input/014.jpg',
                     'input/015.jpg',
                     'input/016.jpg',
                     'input/017.jpg',
                     'input/018.jpg',
                     'input/019.jpg',
                     'input/020.jpg',
                     'input/021.jpg',
                     'input/022.jpg',
                     'input/023.jpg',
                     'input/024.jpg',
                     'input/025.jpg']

nearest_neighbor_half_image_files = ['nearest_neighbor_half/001_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/002_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/003_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/004_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/005_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/006_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/007_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/008_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/009_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/010_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/011_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/012_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/013_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/014_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/015_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/016_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/017_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/018_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/019_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/020_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/021_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/022_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/023_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/024_nearest_neighbor_half.jpg',
                                     'nearest_neighbor_half/025_nearest_neighbor_half.jpg']

nearest_neighbor_quarter_image_files = ['nearest_neighbor_quarter/001_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/002_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/003_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/004_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/005_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/006_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/007_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/008_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/009_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/010_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/011_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/012_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/013_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/014_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/015_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/016_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/017_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/018_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/019_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/020_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/021_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/022_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/023_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/024_nearest_neighbor_quarter.jpg',
                                        'nearest_neighbor_quarter/025_nearest_neighbor_quarter.jpg']

bilinear_half_image_files = ['bilinear_half/001_bilinear_half.jpg',
                             'bilinear_half/002_bilinear_half.jpg',
                             'bilinear_half/003_bilinear_half.jpg',
                             'bilinear_half/004_bilinear_half.jpg',
                             'bilinear_half/005_bilinear_half.jpg',
                             'bilinear_half/006_bilinear_half.jpg',
                             'bilinear_half/007_bilinear_half.jpg',
                             'bilinear_half/008_bilinear_half.jpg',
                             'bilinear_half/009_bilinear_half.jpg',
                             'bilinear_half/010_bilinear_half.jpg',
                             'bilinear_half/011_bilinear_half.jpg',
                             'bilinear_half/012_bilinear_half.jpg',
                             'bilinear_half/013_bilinear_half.jpg',
                             'bilinear_half/014_bilinear_half.jpg',
                             'bilinear_half/015_bilinear_half.jpg',
                             'bilinear_half/016_bilinear_half.jpg',
                             'bilinear_half/017_bilinear_half.jpg',
                             'bilinear_half/018_bilinear_half.jpg',
                             'bilinear_half/019_bilinear_half.jpg',
                             'bilinear_half/020_bilinear_half.jpg',
                             'bilinear_half/021_bilinear_half.jpg',
                             'bilinear_half/022_bilinear_half.jpg',
                             'bilinear_half/023_bilinear_half.jpg',
                             'bilinear_half/024_bilinear_half.jpg',
                             'bilinear_half/025_bilinear_half.jpg']

bilinear_quarter_image_files = ['bilinear_quarter/001_bilinear_quarter.jpg',
                                'bilinear_quarter/002_bilinear_quarter.jpg',
                                'bilinear_quarter/003_bilinear_quarter.jpg',
                                'bilinear_quarter/004_bilinear_quarter.jpg',
                                'bilinear_quarter/005_bilinear_quarter.jpg',
                                'bilinear_quarter/006_bilinear_quarter.jpg',
                                'bilinear_quarter/007_bilinear_quarter.jpg',
                                'bilinear_quarter/008_bilinear_quarter.jpg',
                                'bilinear_quarter/009_bilinear_quarter.jpg',
                                'bilinear_quarter/010_bilinear_quarter.jpg',
                                'bilinear_quarter/011_bilinear_quarter.jpg',
                                'bilinear_quarter/012_bilinear_quarter.jpg',
                                'bilinear_quarter/013_bilinear_quarter.jpg',
                                'bilinear_quarter/014_bilinear_quarter.jpg',
                                'bilinear_quarter/015_bilinear_quarter.jpg',
                                'bilinear_quarter/016_bilinear_quarter.jpg',
                                'bilinear_quarter/017_bilinear_quarter.jpg',
                                'bilinear_quarter/018_bilinear_quarter.jpg',
                                'bilinear_quarter/019_bilinear_quarter.jpg',
                                'bilinear_quarter/020_bilinear_quarter.jpg',
                                'bilinear_quarter/021_bilinear_quarter.jpg',
                                'bilinear_quarter/022_bilinear_quarter.jpg',
                                'bilinear_quarter/023_bilinear_quarter.jpg',
                                'bilinear_quarter/024_bilinear_quarter.jpg',
                                'bilinear_quarter/025_bilinear_quarter.jpg']

bicubic_half_image_files = ['bicubic_half/001_bicubic_half.jpg',
                            'bicubic_half/002_bicubic_half.jpg',
                            'bicubic_half/003_bicubic_half.jpg',
                            'bicubic_half/004_bicubic_half.jpg',
                            'bicubic_half/005_bicubic_half.jpg',
                            'bicubic_half/006_bicubic_half.jpg',
                            'bicubic_half/007_bicubic_half.jpg',
                            'bicubic_half/008_bicubic_half.jpg',
                            'bicubic_half/009_bicubic_half.jpg',
                            'bicubic_half/010_bicubic_half.jpg',
                            'bicubic_half/011_bicubic_half.jpg',
                            'bicubic_half/012_bicubic_half.jpg',
                            'bicubic_half/013_bicubic_half.jpg',
                            'bicubic_half/014_bicubic_half.jpg',
                            'bicubic_half/015_bicubic_half.jpg',
                            'bicubic_half/016_bicubic_half.jpg',
                            'bicubic_half/017_bicubic_half.jpg',
                            'bicubic_half/018_bicubic_half.jpg',
                            'bicubic_half/019_bicubic_half.jpg',
                            'bicubic_half/020_bicubic_half.jpg',
                            'bicubic_half/021_bicubic_half.jpg',
                            'bicubic_half/022_bicubic_half.jpg',
                            'bicubic_half/023_bicubic_half.jpg',
                            'bicubic_half/024_bicubic_half.jpg',
                            'bicubic_half/025_bicubic_half.jpg']

bicubic_quarter_image_files = ['bicubic_quarter/001_bicubic_quarter.jpg',
                               'bicubic_quarter/002_bicubic_quarter.jpg',
                               'bicubic_quarter/003_bicubic_quarter.jpg',
                               'bicubic_quarter/004_bicubic_quarter.jpg',
                               'bicubic_quarter/005_bicubic_quarter.jpg',
                               'bicubic_quarter/006_bicubic_quarter.jpg',
                               'bicubic_quarter/007_bicubic_quarter.jpg',
                               'bicubic_quarter/008_bicubic_quarter.jpg',
                               'bicubic_quarter/009_bicubic_quarter.jpg',
                               'bicubic_quarter/010_bicubic_quarter.jpg',
                               'bicubic_quarter/011_bicubic_quarter.jpg',
                               'bicubic_quarter/012_bicubic_quarter.jpg',
                               'bicubic_quarter/013_bicubic_quarter.jpg',
                               'bicubic_quarter/014_bicubic_quarter.jpg',
                               'bicubic_quarter/015_bicubic_quarter.jpg',
                               'bicubic_quarter/016_bicubic_quarter.jpg',
                               'bicubic_quarter/017_bicubic_quarter.jpg',
                               'bicubic_quarter/018_bicubic_quarter.jpg',
                               'bicubic_quarter/019_bicubic_quarter.jpg',
                               'bicubic_quarter/020_bicubic_quarter.jpg',
                               'bicubic_quarter/021_bicubic_quarter.jpg',
                               'bicubic_quarter/022_bicubic_quarter.jpg',
                               'bicubic_quarter/023_bicubic_quarter.jpg',
                               'bicubic_quarter/024_bicubic_quarter.jpg',
                               'bicubic_quarter/025_bicubic_quarter.jpg']

lanczos_half_image_files = ['lanczos_half/001_lanczos_half.jpg',
                            'lanczos_half/002_lanczos_half.jpg',
                            'lanczos_half/003_lanczos_half.jpg',
                            'lanczos_half/004_lanczos_half.jpg',
                            'lanczos_half/005_lanczos_half.jpg',
                            'lanczos_half/006_lanczos_half.jpg',
                            'lanczos_half/007_lanczos_half.jpg',
                            'lanczos_half/008_lanczos_half.jpg',
                            'lanczos_half/009_lanczos_half.jpg',
                            'lanczos_half/010_lanczos_half.jpg',
                            'lanczos_half/011_lanczos_half.jpg',
                            'lanczos_half/012_lanczos_half.jpg',
                            'lanczos_half/013_lanczos_half.jpg',
                            'lanczos_half/014_lanczos_half.jpg',
                            'lanczos_half/015_lanczos_half.jpg',
                            'lanczos_half/016_lanczos_half.jpg',
                            'lanczos_half/017_lanczos_half.jpg',
                            'lanczos_half/018_lanczos_half.jpg',
                            'lanczos_half/019_lanczos_half.jpg',
                            'lanczos_half/020_lanczos_half.jpg',
                            'lanczos_half/021_lanczos_half.jpg',
                            'lanczos_half/022_lanczos_half.jpg',
                            'lanczos_half/023_lanczos_half.jpg',
                            'lanczos_half/024_lanczos_half.jpg',
                            'lanczos_half/025_lanczos_half.jpg']

lanczos_quarter_image_files = ['lanczos_quarter/001_lanczos_quarter.jpg',
                               'lanczos_quarter/002_lanczos_quarter.jpg',
                               'lanczos_quarter/003_lanczos_quarter.jpg',
                               'lanczos_quarter/004_lanczos_quarter.jpg',
                               'lanczos_quarter/005_lanczos_quarter.jpg',
                               'lanczos_quarter/006_lanczos_quarter.jpg',
                               'lanczos_quarter/007_lanczos_quarter.jpg',
                               'lanczos_quarter/008_lanczos_quarter.jpg',
                               'lanczos_quarter/009_lanczos_quarter.jpg',
                               'lanczos_quarter/010_lanczos_quarter.jpg',
                               'lanczos_quarter/011_lanczos_quarter.jpg',
                               'lanczos_quarter/012_lanczos_quarter.jpg',
                               'lanczos_quarter/013_lanczos_quarter.jpg',
                               'lanczos_quarter/014_lanczos_quarter.jpg',
                               'lanczos_quarter/015_lanczos_quarter.jpg',
                               'lanczos_quarter/016_lanczos_quarter.jpg',
                               'lanczos_quarter/017_lanczos_quarter.jpg',
                               'lanczos_quarter/018_lanczos_quarter.jpg',
                               'lanczos_quarter/019_lanczos_quarter.jpg',
                               'lanczos_quarter/020_lanczos_quarter.jpg',
                               'lanczos_quarter/021_lanczos_quarter.jpg',
                               'lanczos_quarter/022_lanczos_quarter.jpg',
                               'lanczos_quarter/023_lanczos_quarter.jpg',
                               'lanczos_quarter/024_lanczos_quarter.jpg',
                               'lanczos_quarter/025_lanczos_quarter.jpg']

pixel_area_relation_half_image_files = ['pixel_area_relation_half/001_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/002_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/003_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/004_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/005_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/006_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/007_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/008_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/009_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/010_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/011_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/012_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/013_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/014_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/015_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/016_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/017_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/018_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/019_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/020_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/021_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/022_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/023_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/024_pixel_area_relation_half.jpg',
                                        'pixel_area_relation_half/025_pixel_area_relation_half.jpg']

pixel_area_relation_quarter_image_files = ['pixel_area_relation_quarter/001_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/002_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/003_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/004_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/005_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/006_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/007_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/008_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/009_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/010_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/011_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/012_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/013_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/014_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/015_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/016_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/017_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/018_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/019_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/020_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/021_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/022_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/023_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/024_pixel_area_relation_quarter.jpg',
                                           'pixel_area_relation_quarter/025_pixel_area_relation_quarter.jpg']

################################################
# Downsampling + Resampling script (Downscaling factor: 2 and 4)
################################################

################################################
# 4 interpolation methods for downsampling images: (half + quarter)
# Nearest-neighbor, Bilinear, Bicubic, Lanczos
################################################

################################################
# 1 interpolation method for resampling images: (half + quarter)
# Pixel area relation
# Downscaling images
################################################

################################################
# Downsampling
# The principle of read, resize and write images:
# Read:
# image = cv2.imread(input_file)
#   [0]    [1]     [2]    (index of image.shape)
# height, width, channels = image.shape
# Resize based on different interpolation methods:
# resized_image = cv2.resize(image, (width, height), interpolation=...)
# Write:
# output_file = cv2.imwrite(output_file, resized_image)
################################################

################################################
# Resampling
# The principle of read, resize and write images:
# Read:
# image = cv2.imread(input_file)
#   [0]    [1]     [2]    (index of image.shape)
# height, width, channels = image.shape
# Resize:
# resized_image = cv2.resize(image, (width, height), interpolation=...)
# Write:
# output_file = cv2.imwrite(output_file, resized_image)
################################################

################################################
# Pre-processing downsampling and resampling functions
################################################

################################################
# Nearest-neighbor interpolation
################################################
def nearest_neighbor_interpolation_half(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_NEAREST)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def nearest_neighbor_interpolation_quarter(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_NEAREST)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
################################################

################################################
# Bilinear interpolation
################################################
def bilinear_interpolation_half(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def bilinear_interpolation_quarter(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_LINEAR)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
################################################

################################################
# Bicubic interpolation over 4x4 pixel neighborhood
################################################
def bicubic_interpolation_half(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def bicubic_interpolation_quarter(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
################################################

################################################
# Lanczos interpolation over 8x8 pixel neighborhood
################################################
def lanczos_interpolation_half(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_LANCZOS4)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def lanczos_interpolation_quarter(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_LANCZOS4)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
################################################
        
################################################
# Resampling using pixel area relation
# It is the best method for downscaling images
################################################
def pixel_area_relation_interpolation_half(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_AREA)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def pixel_area_relation_interpolation_quarter(input_file, output_file):
    try:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        resized_image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation=cv2.INTER_AREA)
        output_file = cv2.imwrite(output_file, resized_image)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
################################################

################################################
# Main function
################################################ 
if __name__ == "__main__":

    # Configure logging to save to a file
    log_filename = 'downsampling-resampling-log.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    ####################
    # Downsampling part:
    ####################
    print("Downsampling part is starting")

    # Log the separation line
    logging.info(f"################################################################################################")

    ################################################
    # Half
    ################################################
    # Nearest-neighbor
    nearest_neighbor_half_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = nearest_neighbor_half_image_files[i]
            nearest_neighbor_interpolation_half(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    nearest_neighbor_half_end_time = time.time()
    nearest_neighbor_half_elapsed_time = nearest_neighbor_half_end_time - nearest_neighbor_half_start_time
    # Bilinear
    bilinear_half_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = bilinear_half_image_files[i]
            bilinear_interpolation_half(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    bilinear_half_end_time = time.time()
    bilinear_half_elapsed_time = bilinear_half_end_time - bilinear_half_start_time
    # Bicubic
    bicubic_half_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = bicubic_half_image_files[i]
            bicubic_interpolation_half(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    bicubic_half_end_time = time.time()
    bicubic_half_elapsed_time = bicubic_half_end_time - bicubic_half_start_time
    # Lanczos
    lanczos_half_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = lanczos_half_image_files[i]
            lanczos_interpolation_half(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    lanczos_half_end_time = time.time()
    lanczos_half_elapsed_time = lanczos_half_end_time - lanczos_half_start_time

    # Log the half processing time
    logging.info(f"Downsampling | nearest-neighbor half processing time: {nearest_neighbor_half_elapsed_time} seconds")
    logging.info(f"Downsampling | bilinear half processing time: {bilinear_half_elapsed_time} seconds")
    logging.info(f"Downsampling | bicubic half processing time: {bicubic_half_elapsed_time} seconds")
    logging.info(f"Downsampling | lanczos half processing time: {lanczos_half_elapsed_time} seconds")

    # Display the half processing time
    print(f"Downsampling | nearest-neighbor half processing time: {nearest_neighbor_half_elapsed_time} seconds")
    print(f"Downsampling | bilinear half processing time: {bilinear_half_elapsed_time} seconds")
    print(f"Downsampling | bicubic half processing time: {bicubic_half_elapsed_time} seconds")
    print(f"Downsampling | lanczos half processing time: {lanczos_half_elapsed_time} seconds")

    # Log the separation line
    logging.info(f"################################################################################################")

    ################################################
    # Quarter
    ################################################
    # Nearest-neighbor
    nearest_neighbor_quarter_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = nearest_neighbor_quarter_image_files[i]
            nearest_neighbor_interpolation_quarter(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    nearest_neighbor_quarter_end_time = time.time()
    nearest_neighbor_quarter_elapsed_time = nearest_neighbor_quarter_end_time - nearest_neighbor_quarter_start_time
    # Bilinear
    bilinear_quarter_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = bilinear_quarter_image_files[i]
            bilinear_interpolation_quarter(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    bilinear_quarter_end_time = time.time()
    bilinear_quarter_elapsed_time = bilinear_quarter_end_time - bilinear_quarter_start_time
    # Bicubic
    bicubic_quarter_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = bicubic_quarter_image_files[i]
            bicubic_interpolation_quarter(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    bicubic_quarter_end_time = time.time()
    bicubic_quarter_elapsed_time = bicubic_quarter_end_time - bicubic_quarter_start_time
    # Lanczos
    lanczos_quarter_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = lanczos_quarter_image_files[i]
            lanczos_interpolation_quarter(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    lanczos_quarter_end_time = time.time()
    lanczos_quarter_elapsed_time = lanczos_quarter_end_time - lanczos_quarter_start_time

    # Log the quarter processing time
    logging.info(f"Downsampling | nearest-neighbor quarter processing time: {nearest_neighbor_quarter_elapsed_time} seconds")
    logging.info(f"Downsampling | bilinear quarter processing time: {bilinear_quarter_elapsed_time} seconds")
    logging.info(f"Downsampling | bicubic quarter processing time: {bicubic_quarter_elapsed_time} seconds")
    logging.info(f"Downsampling | lanczos quarter processing time: {lanczos_quarter_elapsed_time} seconds")

    # Display the quarter processing time
    print(f"Downsampling | nearest-neighbor quarter processing time: {nearest_neighbor_quarter_elapsed_time} seconds")
    print(f"Downsampling | bilinear quarter processing time: {bilinear_quarter_elapsed_time} seconds")
    print(f"Downsampling | bicubic quarter processing time: {bicubic_quarter_elapsed_time} seconds")
    print(f"Downsampling | lanczos quarter processing time: {lanczos_quarter_elapsed_time} seconds")

    # Log the separation line
    logging.info(f"################################################################################################")

    # Indication of downsampling completed
    print(f"Downsampling | nearest-neighbor, bilinear, bicubic and lanczos completed")

    ####################
    # Resampling part: downscaling
    ####################
    print("Resampling part is starting")

    # Log the separation line
    logging.info(f"################################################################################################")

    ################################################
    # Half
    ################################################
    # Pixel area relation
    pixel_area_relation_half_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = pixel_area_relation_half_image_files[i]
            pixel_area_relation_interpolation_half(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    pixel_area_relation_half_end_time = time.time()
    pixel_area_relation_half_elapsed_time = pixel_area_relation_half_end_time - pixel_area_relation_half_start_time
    
    ################################################
    # Quarter
    ################################################
    # Pixel area relation
    pixel_area_relation_quarter_start_time = time.time()
    for i in range(25):
        try:
            input_image = input_image_files[i]
            if input_image is None:
                raise FileNotFoundError(f"Error: Unable to read input image at {input_image}")
            output_image = pixel_area_relation_quarter_image_files[i]
            pixel_area_relation_interpolation_quarter(input_image, output_image)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")
    pixel_area_relation_quarter_end_time = time.time()
    pixel_area_relation_quarter_elapsed_time = pixel_area_relation_quarter_end_time - pixel_area_relation_quarter_start_time

    # Log the processing time
    logging.info(f"Resampling | pixel area relation half processing time: {pixel_area_relation_half_elapsed_time} seconds")
    logging.info(f"Resampling | pixel area relation quarter processing time: {pixel_area_relation_quarter_elapsed_time} seconds")

    # Display the processsing time
    print(f"Resampling | pixel area relation half processing time: {pixel_area_relation_half_elapsed_time} seconds")
    print(f"Resampling | pixel area relation quarter processing time: {pixel_area_relation_quarter_elapsed_time} seconds")

    # Log the separation line
    logging.info(f"################################################################################################")

    # Indication of resampling completed
    print(f"Resampling | pixel area relation completed")

    ####################
    # Plotting diagram part: table
    ####################

    # Data to be used to plot table
    data = {
        'Pre-processing metrics' : ['Nearest-neighbor (Half)',
                                   'Nearest-neighbor (Quarter)',
                                   'Bilinear (Half)',
                                   'Bilinear (Quarter)',
                                   'Bicubic (Half)',
                                   'Bicubic (Quarter)',
                                   'Lanczos (Half)',
                                   'Lanczos (Quarter)',
                                   'Pixel area relation (Half)',
                                   'Pixel area relation (Quarter)'],
        'Scale' : ['2',
                   '4',
                   '2',
                   '4',
                   '2',
                   '4',
                   '2',
                   '4',
                   '2',
                   '4'],
        'Processing time' : [nearest_neighbor_half_elapsed_time,
                             nearest_neighbor_quarter_elapsed_time,
                             bilinear_half_elapsed_time,
                             bilinear_quarter_elapsed_time,
                             bicubic_half_elapsed_time,
                             bicubic_quarter_elapsed_time,
                             lanczos_half_elapsed_time,
                             lanczos_quarter_elapsed_time,
                             pixel_area_relation_half_elapsed_time,
                             pixel_area_relation_quarter_elapsed_time],
        'Average processing time' : [nearest_neighbor_half_elapsed_time/25,
                                     nearest_neighbor_quarter_elapsed_time/25,
                                     bilinear_half_elapsed_time/25,
                                     bilinear_quarter_elapsed_time/25,
                                     bicubic_half_elapsed_time/25,
                                     bicubic_quarter_elapsed_time/25,
                                     lanczos_half_elapsed_time/25,
                                     lanczos_quarter_elapsed_time/25,
                                     pixel_area_relation_half_elapsed_time/25,
                                     pixel_area_relation_quarter_elapsed_time/25]
    }

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    df.fillna('-', inplace=True) # Replacing missing values with '-'

    # Create a figure to plot the table
    fig, ax = plt.subplots(figsize=(12, 8)) # Size should be adjusted to match the aspect ratio of your data
    ax.axis('off') # Hide the axes

    #############################
    # Table properties
    #############################

    #############################
    # Loose version of table
    # Table will take more space
    #############################
    # tab = table(ax, df, loc='center', cellLoc='center', colLoc='center')
    # tab.auto_set_font_size(False)
    # tab.set_fontsize(10)
    # tab.scale(1.2, 1.2)  # Scale table size

    #############################
    # Plot the table
    #############################
    # Compact version of table
    # Table will take less space
    #############################
    tab = table(ax, df, loc='center', cellLoc='center', colLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10) # Adjust to match the font size in your image
    tab.auto_set_column_width(col=list(range(len(df.columns)))) # Adjust column widths

    # Style the table
    for (i, j), val in np.ndenumerate(df.values):
        if j == 0: # First column
            tab[(i+1, j)].set_facecolor('#dddddd') # Shade the first column
        if j == -1 or j == 0: # Header or first column
            tab[(i+1, j)].set_text_props(weight='bold') # Bold the font of first column

    # Add the title
    plt.suptitle('TABLE\nThe processing time of downsampling and resampling on dataset', fontsize=14, weight='bold')

    # Save the figure as a PDF
    pdf_file = 'table.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    # Save the figure as an image (PNG)
    png_file = 'table.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)

    # Close the matplotlib figure
    plt.close(fig)

    # Indication of plotting table completed
    print("Plotting table completed")
