################################################
# RDN: psnr-large
# https://github.com/idealo/image-super-resolution/
# @misc{cardinale2018isr,
#   title={ISR},
#   author={Francesco Cardinale et al.},
#   year={2018},
#   howpublished={\url{https://github.com/idealo/image-super-resolution}},
# }
################################################

################################################
# Dataset from Nick Yue
# Special thanks to nick :)
# Pre-processing images produced by Yumu Xie
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
# ISR, OpenCV, Matplotlib, Pandas and Numpy packages are required in this script
# Instruction of installing packages:
# pip install ISR
# pip install opencv-python
# pip install matplotlib
# pip install pandas
# pip install numpy
#
# The scale factor of RDN psnr-large model is 2
# For example, an input image has resolution (x, y)
# After RDN psnr-large model, the output image will have resolution (2x, 2y)
#
# The purpose of this script is to execute super-resolution of RDN psnr-large model
# [input_image] -> [RDN(psnr-large)] -> [output_image]
################################################

################################################
# Please ensure that the folders contain input images are named as following:
# "nearest_neighbor_half"
# "bilinear_half"
# "bicubic_half"
# "lanczos_half"
# "pixel_area_relation_half"
# Please create folders to contain output images:
# "nearest_neighbor_psnr_large"
# "bilinear_psnr_large"
# "bicubic_psnr_large"
# "lanczos_psnr_large"
# "pixel_area_relation_psnr_large"
################################################

################################################
# RDN psnr-large
import cv2
from ISR.models import RDN
import time
################################################

################################################
# Table
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
################################################

################################################
# Input images
################################################
# Nearest-neighbor
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
# Bilinear
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
# Bicubic
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
# Lanczos
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
# Pixel area relation
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

################################################
# Input images
################################################
# Nearest-neighbor
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
# Bilinear
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
# Bicubic
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
# Lanczos
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
# Pixel area relation
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
# Output images
################################################
# Nearest-neighbor (psnr-large)
nearest_neighbor_psnr_large_image_files = [   'nearest_neighbor_psnr_large/001_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/002_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/003_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/004_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/005_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/006_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/007_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/008_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/009_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/010_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/011_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/012_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/013_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/014_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/015_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/016_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/017_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/018_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/019_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/020_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/021_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/022_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/023_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/024_nearest_neighbor_psnr_large.jpg',
                                        'nearest_neighbor_psnr_large/025_nearest_neighbor_psnr_large.jpg']
# Bilinear (psnr-large)
bilinear_psnr_large_image_files = [   'bilinear_psnr_large/001_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/002_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/003_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/004_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/005_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/006_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/007_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/008_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/009_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/010_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/011_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/012_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/013_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/014_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/015_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/016_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/017_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/018_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/019_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/020_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/021_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/022_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/023_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/024_bilinear_psnr_large.jpg',
                                'bilinear_psnr_large/025_bilinear_psnr_large.jpg']
# Bicubic (psnr-large)
bicubic_psnr_large_image_files = [   'bicubic_psnr_large/001_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/002_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/003_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/004_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/005_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/006_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/007_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/008_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/009_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/010_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/011_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/012_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/013_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/014_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/015_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/016_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/017_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/018_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/019_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/020_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/021_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/022_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/023_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/024_bicubic_psnr_large.jpg',
                               'bicubic_psnr_large/025_bicubic_psnr_large.jpg']
# Lanczos (psnr-large)
lanczos_psnr_large_image_files = [   'lanczos_psnr_large/001_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/002_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/003_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/004_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/005_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/006_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/007_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/008_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/009_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/010_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/011_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/012_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/013_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/014_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/015_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/016_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/017_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/018_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/019_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/020_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/021_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/022_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/023_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/024_lanczos_psnr_large.jpg',
                               'lanczos_psnr_large/025_lanczos_psnr_large.jpg']
# Pixel area relation (psnr-large)
pixel_area_relation_psnr_large_image_files = [   'pixel_area_relation_psnr_large/001_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/002_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/003_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/004_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/005_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/006_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/007_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/008_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/009_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/010_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/011_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/012_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/013_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/014_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/015_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/016_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/017_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/018_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/019_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/020_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/021_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/022_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/023_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/024_pixel_area_relation_psnr_large.jpg',
                                           'pixel_area_relation_psnr_large/025_pixel_area_relation_psnr_large.jpg']

# super_resolve_image function
def super_resolve_image(input_path, output_path, model_path = 'psnr-large'):
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")       
        # Initialise the RDN model
        model = RDN(weights=model_path)
        # Perform super-resolution of 'psnr-large' model
        sr_img = model.predict(img)
        # Save the output image
        cv2.imwrite(output_path, sr_img)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

# main
if __name__ == "__main__":

    ##########
    # half image files are not used in RDN
    # quarter image files will be utilised
    # full reference image files will take half image files 
    # to use instead of original image files
    # this is due to different pixel dimensions
    # RDN has issues related to process half image files
    # Debugging:
    # 1. Script memory allocation [x]
    # 2. RDN processing limitation (very large pixel dimensions may not lead RDN working) [o]
    ##########

    print("RDN psnr-large part is starting")

    # Start time
    start_time = time.time()

    # Nearest-neighbor
    # Load data set and perform super-resolution
    nearest_neighbor_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = nearest_neighbor_quarter_image_files[i]
        # input_file = nearest_neighbor_half_image_files[i]
        output_file = nearest_neighbor_psnr_large_image_files[i]
        super_resolve_image(input_file, output_file)
    nearest_neighbor_end_time = time.time()
    nearest_neighbor_elapsed_time = nearest_neighbor_end_time - nearest_neighbor_start_time
    nearest_neighbor_average_time = nearest_neighbor_elapsed_time/25

    # Bilinear
    # Load data set and perform super-resolution
    bilinear_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = bilinear_quarter_image_files[i]
        # input_file = bilinear_half_image_files[i]
        output_file = bilinear_psnr_large_image_files[i]
        super_resolve_image(input_file, output_file)
    bilinear_end_time = time.time()
    bilinear_elapsed_time = bilinear_end_time - bilinear_start_time
    bilinear_average_time = bilinear_elapsed_time/25

    # Bicubic
    # Load data set and perform super-resolution
    bicubic_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = bicubic_quarter_image_files[i]
        # input_file = bicubic_half_image_files[i]
        output_file = bicubic_psnr_large_image_files[i]
        super_resolve_image(input_file, output_file)
    bicubic_end_time = time.time()
    bicubic_elapsed_time = bicubic_end_time - bicubic_start_time
    bicubic_average_time = bicubic_elapsed_time/25

    # Lanczos
    # Load data set and perform super-resolution
    lanczos_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = lanczos_quarter_image_files[i]
        # input_file = lanczos_half_image_files[i]
        output_file = lanczos_psnr_large_image_files[i]
        super_resolve_image(input_file, output_file)
    lanczos_end_time = time.time()
    lanczos_elapsed_time = lanczos_end_time - lanczos_start_time
    lanczos_average_time = lanczos_elapsed_time/25

    # Pixel area relation
    # Load data set and perform super-resolution
    pixel_area_relation_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = pixel_area_relation_quarter_image_files[i]
        # input_file = pixel_area_relation_half_image_files[i]
        output_file = pixel_area_relation_psnr_large_image_files[i]
        super_resolve_image(input_file, output_file)
    pixel_area_relation_end_time = time.time()
    pixel_area_relation_elapsed_time = pixel_area_relation_end_time - pixel_area_relation_start_time
    pixel_area_relation_average_time = pixel_area_relation_elapsed_time/25

    # End time
    end_time = time.time()

    # Total processing time
    elapsed_time = end_time - start_time

    # Average processing time
    average_time = elapsed_time/125

    # Display the processsing time
    print(f"RDN: psnr-large | total processing time: {elapsed_time} seconds")
    print(f"RDN: psnr-large | average processing time: {average_time} seconds")

    # Indication of super-resolution completed
    print(f"RDN: psnr-large | Super-resolution completed")

    ####################
    # Plotting diagram part: table
    ####################

    # Data to be used to plot table
    data = {
        'Processing metrics' : ['Nearest-neighbor + psnr-large',
                                'Bilinear + psnr-large',
                                'Bicubic + psnr-large',
                                'Lanczos + psnr-large',
                                'Pixel area relation + psnr-large',
                                'Total'
                                ],
        'Scale' : ['2',
                   '2',
                   '2',
                   '2',
                   '2',
                   '2'
                   ],
        'Processing time' : [nearest_neighbor_elapsed_time,
                             bilinear_elapsed_time,
                             bicubic_elapsed_time,
                             lanczos_elapsed_time,
                             pixel_area_relation_elapsed_time,
                             elapsed_time
                             ],
        'Average processing time' : [nearest_neighbor_average_time,
                                     bilinear_average_time,
                                     bicubic_average_time,
                                     lanczos_average_time,
                                     pixel_area_relation_average_time,
                                     average_time]
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
    plt.suptitle('TABLE\nThe processing time of RDN psnr-large model', fontsize=14, weight='bold')

    # Save the figure as a PDF
    pdf_file = 'psnr-large.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    # Save the figure as an image (PNG)
    png_file = 'psnr-large.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)

    # Close the matplotlib figure
    plt.close(fig)

    # Indication of plotting table completed
    print("Plotting table completed")
