################################################
# Dimension
################################################

################################################
# Final result:
# Equal: 004, 005, 006, 015, 016, 017, 018, 019, 020, 021, 024, 025
# Not Equal: 001, 002, 003, 007, 008, 009, 010, 011, 012, 013, 014, 022, 023
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

##############################################
# Programming environment: WSL Linux subsystem of Windows
# Python version: 3.7.9
# Python setup is based on 'pyenv' (Simple Python version management)
# https://github.com/pyenv/pyenv
#
# OpenCV package are required in this script
# Instruction of installing package:
# pip install opencv-python
#
# The purpose of this script is to compare the dimension of two images
# Same dimension -> No need to do bilinear interpolation upsampling
# Different dimension -> Need to do bilinear interpolation upsampling
################################################

################################################
# Dimension checking
import cv2
################################################

################################################
# Reference images
################################################
# Dataset
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

################################################
# Distorted images
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

def main():
    # Nearest-neighbor
    print("--------------------------------------------------")
    print("Nearest-neighbor")
    print("--------------------------------------------------")
    for i in range(25):
        ref_img = cv2.imread(input_image_files[i])
        dis_img = cv2.imread(nearest_neighbor_gans_image_files[i])
        if ref_img.shape != dis_img.shape:
            print("##############################")
            print(f"Not equal at {nearest_neighbor_gans_image_files[i]}")
            print("Bilinear interpolation will be used for upsampling (resizing)")
            print("##############################")
        if ref_img.shape == dis_img.shape:
            print("##############################")
            print(f"Equal at {nearest_neighbor_gans_image_files[i]}")
            print("##############################")
    print("--------------------------------------------------")
    print()
    # Bilinear
    print("--------------------------------------------------")
    print("Bilinear")
    print("--------------------------------------------------")
    for i in range(25):
        ref_img = cv2.imread(input_image_files[i])
        dis_img = cv2.imread(bilinear_gans_image_files[i])
        if ref_img.shape != dis_img.shape:
            print("##############################")
            print(f"Not equal at {bilinear_gans_image_files[i]}")
            print("Bilinear interpolation will be used for upsampling (resizing)")
            print("##############################")
        if ref_img.shape == dis_img.shape:
            print("##############################")
            print(f"Equal at {bilinear_gans_image_files[i]}")
            print("##############################")
    print("--------------------------------------------------")
    print()
    # Bicubic
    print("--------------------------------------------------")
    print("Bicubic")
    print("--------------------------------------------------")
    for i in range(25):
        ref_img = cv2.imread(input_image_files[i])
        dis_img = cv2.imread(bicubic_gans_image_files[i])
        if ref_img.shape != dis_img.shape:
            print("##############################")
            print(f"Not equal at {bicubic_gans_image_files[i]}")
            print("Bilinear interpolation will be used for upsampling (resizing)")
            print("##############################")
        if ref_img.shape == dis_img.shape:
            print("##############################")
            print(f"Equal at {bicubic_gans_image_files[i]}")
            print("##############################")
    print("--------------------------------------------------")
    print()
    # Lanczos
    print("--------------------------------------------------")
    print("Lanczos")
    print("--------------------------------------------------")
    for i in range(25):
        ref_img = cv2.imread(input_image_files[i])
        dis_img = cv2.imread(lanczos_gans_image_files[i])
        if ref_img.shape != dis_img.shape:
            print("##############################")
            print(f"Not equal at {lanczos_gans_image_files[i]}")
            print("Bilinear interpolation will be used for upsampling (resizing)")
            print("##############################")
        if ref_img.shape == dis_img.shape:
            print("##############################")
            print(f"Equal at {lanczos_gans_image_files[i]}")
            print("##############################")
    print("--------------------------------------------------")
    print()
    # Pixel area relation
    print("--------------------------------------------------")
    print("Pixel area relation")
    print("--------------------------------------------------")
    for i in range(25):
        ref_img = cv2.imread(input_image_files[i])
        dis_img = cv2.imread(pixel_area_relation_gans_image_files[i])
        if ref_img.shape != dis_img.shape:
            print("##############################")
            print(f"Not equal at {pixel_area_relation_gans_image_files[i]}")
            print("Bilinear interpolation will be used for upsampling (resizing)")
            print("##############################")
        if ref_img.shape == dis_img.shape:
            print("##############################")
            print(f"Equal at {pixel_area_relation_gans_image_files[i]}")
            print("##############################")
    print("--------------------------------------------------")
    print()

if __name__ == "__main__":
    main()
