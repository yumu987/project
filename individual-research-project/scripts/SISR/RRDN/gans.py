################################################
# RRDN: gans
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
# The scale factor of RRDN gans model is 4
# For example, an input image has resolution (x, y)
# After RRDN gans model, the output image will have resolution (4x, 4y)
#
# The purpose of this script is to execute super-resolution of RRDN gans model
# [input_image] -> [RRDN(gans)] -> [output_image]
################################################

################################################
# Please ensure that the folders contain input images are named as following:
# "nearest_neighbor_quarter"
# "bilinear_quarter"
# "bicubic_quarter"
# "lanczos_quarter"
# "pixel_area_relation_quarter"
# Please create folders to contain output images:
# "nearest_neighbor_gans"
# "bilinear_gans"
# "bicubic_gans"
# "lanczos_gans"
# "pixel_area_relation_gans"
################################################

################################################
# RRDN gans
import cv2
from ISR.models import RRDN
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
# Nearest-neighbor (GANs)
nearest_neighbor_gans_image_files = [   'nearest_neighbor_gans/001_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/002_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/003_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/004_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/005_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/006_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/007_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/008_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/009_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/010_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/011_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/012_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/013_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/014_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/015_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/016_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/017_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/018_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/019_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/020_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/021_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/022_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/023_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/024_nearest_neighbor_gans.jpg',
                                        'nearest_neighbor_gans/025_nearest_neighbor_gans.jpg']
# Bilinear (GANs)
bilinear_gans_image_files = [   'bilinear_gans/001_bilinear_gans.jpg',
                                'bilinear_gans/002_bilinear_gans.jpg',
                                'bilinear_gans/003_bilinear_gans.jpg',
                                'bilinear_gans/004_bilinear_gans.jpg',
                                'bilinear_gans/005_bilinear_gans.jpg',
                                'bilinear_gans/006_bilinear_gans.jpg',
                                'bilinear_gans/007_bilinear_gans.jpg',
                                'bilinear_gans/008_bilinear_gans.jpg',
                                'bilinear_gans/009_bilinear_gans.jpg',
                                'bilinear_gans/010_bilinear_gans.jpg',
                                'bilinear_gans/011_bilinear_gans.jpg',
                                'bilinear_gans/012_bilinear_gans.jpg',
                                'bilinear_gans/013_bilinear_gans.jpg',
                                'bilinear_gans/014_bilinear_gans.jpg',
                                'bilinear_gans/015_bilinear_gans.jpg',
                                'bilinear_gans/016_bilinear_gans.jpg',
                                'bilinear_gans/017_bilinear_gans.jpg',
                                'bilinear_gans/018_bilinear_gans.jpg',
                                'bilinear_gans/019_bilinear_gans.jpg',
                                'bilinear_gans/020_bilinear_gans.jpg',
                                'bilinear_gans/021_bilinear_gans.jpg',
                                'bilinear_gans/022_bilinear_gans.jpg',
                                'bilinear_gans/023_bilinear_gans.jpg',
                                'bilinear_gans/024_bilinear_gans.jpg',
                                'bilinear_gans/025_bilinear_gans.jpg']
# Bicubic (GANs)
bicubic_gans_image_files = [   'bicubic_gans/001_bicubic_gans.jpg',
                               'bicubic_gans/002_bicubic_gans.jpg',
                               'bicubic_gans/003_bicubic_gans.jpg',
                               'bicubic_gans/004_bicubic_gans.jpg',
                               'bicubic_gans/005_bicubic_gans.jpg',
                               'bicubic_gans/006_bicubic_gans.jpg',
                               'bicubic_gans/007_bicubic_gans.jpg',
                               'bicubic_gans/008_bicubic_gans.jpg',
                               'bicubic_gans/009_bicubic_gans.jpg',
                               'bicubic_gans/010_bicubic_gans.jpg',
                               'bicubic_gans/011_bicubic_gans.jpg',
                               'bicubic_gans/012_bicubic_gans.jpg',
                               'bicubic_gans/013_bicubic_gans.jpg',
                               'bicubic_gans/014_bicubic_gans.jpg',
                               'bicubic_gans/015_bicubic_gans.jpg',
                               'bicubic_gans/016_bicubic_gans.jpg',
                               'bicubic_gans/017_bicubic_gans.jpg',
                               'bicubic_gans/018_bicubic_gans.jpg',
                               'bicubic_gans/019_bicubic_gans.jpg',
                               'bicubic_gans/020_bicubic_gans.jpg',
                               'bicubic_gans/021_bicubic_gans.jpg',
                               'bicubic_gans/022_bicubic_gans.jpg',
                               'bicubic_gans/023_bicubic_gans.jpg',
                               'bicubic_gans/024_bicubic_gans.jpg',
                               'bicubic_gans/025_bicubic_gans.jpg']
# Lanczos (GANs)
lanczos_gans_image_files = [   'lanczos_gans/001_lanczos_gans.jpg',
                               'lanczos_gans/002_lanczos_gans.jpg',
                               'lanczos_gans/003_lanczos_gans.jpg',
                               'lanczos_gans/004_lanczos_gans.jpg',
                               'lanczos_gans/005_lanczos_gans.jpg',
                               'lanczos_gans/006_lanczos_gans.jpg',
                               'lanczos_gans/007_lanczos_gans.jpg',
                               'lanczos_gans/008_lanczos_gans.jpg',
                               'lanczos_gans/009_lanczos_gans.jpg',
                               'lanczos_gans/010_lanczos_gans.jpg',
                               'lanczos_gans/011_lanczos_gans.jpg',
                               'lanczos_gans/012_lanczos_gans.jpg',
                               'lanczos_gans/013_lanczos_gans.jpg',
                               'lanczos_gans/014_lanczos_gans.jpg',
                               'lanczos_gans/015_lanczos_gans.jpg',
                               'lanczos_gans/016_lanczos_gans.jpg',
                               'lanczos_gans/017_lanczos_gans.jpg',
                               'lanczos_gans/018_lanczos_gans.jpg',
                               'lanczos_gans/019_lanczos_gans.jpg',
                               'lanczos_gans/020_lanczos_gans.jpg',
                               'lanczos_gans/021_lanczos_gans.jpg',
                               'lanczos_gans/022_lanczos_gans.jpg',
                               'lanczos_gans/023_lanczos_gans.jpg',
                               'lanczos_gans/024_lanczos_gans.jpg',
                               'lanczos_gans/025_lanczos_gans.jpg']
# Pixel area relation (GANs)
pixel_area_relation_gans_image_files = [   'pixel_area_relation_gans/001_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/002_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/003_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/004_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/005_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/006_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/007_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/008_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/009_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/010_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/011_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/012_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/013_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/014_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/015_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/016_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/017_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/018_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/019_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/020_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/021_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/022_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/023_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/024_pixel_area_relation_gans.jpg',
                                           'pixel_area_relation_gans/025_pixel_area_relation_gans.jpg']

# super_resolve_image function
def super_resolve_image(input_path, output_path, model_path = 'gans'):
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Initialise the RRDN model
        model = RRDN(weights=model_path)
        # Perform super-resolution of 'gans' model
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

    print("RRDN gans part is starting")

    # Start time
    start_time = time.time()

    # Nearest-neighbor
    # Load data set and perform super-resolution
    nearest_neighbor_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = nearest_neighbor_quarter_image_files[i]
        output_file = nearest_neighbor_gans_image_files[i]
        super_resolve_image(input_file, output_file)
    nearest_neighbor_end_time = time.time()
    nearest_neighbor_elapsed_time = nearest_neighbor_end_time - nearest_neighbor_start_time
    nearest_neighbor_average_time = nearest_neighbor_elapsed_time/25

    # Bilinear
    # Load data set and perform super-resolution
    bilinear_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = bilinear_quarter_image_files[i]
        output_file = bilinear_gans_image_files[i]
        super_resolve_image(input_file, output_file)
    bilinear_end_time = time.time()
    bilinear_elapsed_time = bilinear_end_time - bilinear_start_time
    bilinear_average_time = bilinear_elapsed_time/25

    # Bicubic
    # Load data set and perform super-resolution
    bicubic_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = bicubic_quarter_image_files[i]
        output_file = bicubic_gans_image_files[i]
        super_resolve_image(input_file, output_file)
    bicubic_end_time = time.time()
    bicubic_elapsed_time = bicubic_end_time - bicubic_start_time
    bicubic_average_time = bicubic_elapsed_time/25

    # Lanczos
    # Load data set and perform super-resolution
    lanczos_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = lanczos_quarter_image_files[i]
        output_file = lanczos_gans_image_files[i]
        super_resolve_image(input_file, output_file)
    lanczos_end_time = time.time()
    lanczos_elapsed_time = lanczos_end_time - lanczos_start_time
    lanczos_average_time = lanczos_elapsed_time/25

    # Pixel area relation
    # Load data set and perform super-resolution
    pixel_area_relation_start_time = time.time()
    for i in range(25): # len(input_image_files) / len(output_image_files
        input_file = pixel_area_relation_quarter_image_files[i]
        output_file = pixel_area_relation_gans_image_files[i]
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
    print(f"RRDN: gans | total processing time: {elapsed_time} seconds")
    print(f"RRDN: gans | average processing time: {average_time} seconds")

    # Indication of super-resolution completed
    print(f"RRDN: gans | Super-resolution completed")

    ####################
    # Plotting diagram part: table
    ####################

    # Data to be used to plot table
    data = {
        'Processing metrics' : ['Nearest-neighbor + GANs',
                                'Bilinear + GANs',
                                'Bicubic + GANs',
                                'Lanczos + GANs',
                                'Pixel area relation + GANs',
                                'Total'
                                ],
        'Scale' : ['4',
                   '4',
                   '4',
                   '4',
                   '4',
                   '4'
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
    plt.suptitle('TABLE\nThe processing time of RRDN GANs model', fontsize=14, weight='bold')

    # Save the figure as a PDF
    pdf_file = 'gans.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    # Save the figure as an image (PNG)
    png_file = 'gans.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)

    # Close the matplotlib figure
    plt.close(fig)

    # Indication of plotting table completed
    print("Plotting table completed")
