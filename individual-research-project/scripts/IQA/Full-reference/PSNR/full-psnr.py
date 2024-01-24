################################################
# Peak-signal-to-noise-ratio (PSNR)
# https://ieeexplore.ieee.org/document/8980171
# PSNR vs. SSIM
# https://ieeexplore.ieee.org/document/5596999
################################################

################################################
# Set 5 dataset
# https://github.com/jbhuang0604/SelfExSR
# https://paperswithcode.com/dataset/set5
# @inproceedings{Huang-CVPR-2015,
#     title={Single Image Super-Resolution From Transformed Self-Exemplars},
#     Author = {Huang, Jia-Bin and Singh, Abhishek and Ahuja, Narendra},
#     booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#     pages={5197--5206},
#     Year = {2015}
# }
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
# NumPy and OpenCV packages are required in this script
# Instruction of installing Numpy and OpenCV:
# pip install numpy
# pip install opencv-python
#
# Peak-signal-to-noise-ratio (PSNR) is usually measured by computing distortion 
# in terms of mean-squared-error (MSE) between the reference image and its
# distorted version. Mean-squared-error (MSE) is measured over the entire image by giving 
# equal weight to each corresponding pixel difference of the reference and distorted image.
#
# The PSNR value approaches infinity as the MSE approaches
# zero; this shows that a higher PSNR value provides a higher
# image quality. At the other end of the scale, a small value of
# the PSNR implies high numerical differences between images.
#
# Small PSNR -> High numerical differences
# High PSNR -> Small numerical differences
#
# LaTeX formula:
# Mean-squared-error (MSE)
# \begin{equation}
# MSE = \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} (I_r(i,j) - I_d(i,j))^2
# \end{equation}
# Peak-signal-to-noise-ratio (PSNR)
# \begin{equation}
# PSNR = 10 \log_{10} \left(\frac{255^2}{MSE}\right)
# \end{equation}
#
# Proposed Idea:
# [Reference Image]
#           \
#            \
#             ---> [Blockwise MSE] -> [Maximum MSE] -> [PSNR] -> [PSNR-MDR]
#            /
#           /
# [Distorted Image]
#
# The purpose of this script is to execute PSNR IQA metric
# [MSE] -> [PSNR]
#
# Please create a 'txt' file called 'full-psnr-output.txt' under the same directory (folder)
# before running this script. Otherwise, it may have some potential issues.
################################################

import cv2
import numpy as np

# reference_image_files
# total: 35
reference_image_files = [# RDN psnr-large first loop
                         'Set5_SR/Set5/image_SRF_2/img_001_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_002_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_003_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_004_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_005_SRF_2_HR.png',
                         # RDN psnr-small second loop
                         'Set5_SR/Set5/image_SRF_2/img_001_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_002_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_003_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_004_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_005_SRF_2_HR.png',
                         # RDN noise-cancel third loop
                         'Set5_SR/Set5/image_SRF_2/img_001_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_002_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_003_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_004_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_005_SRF_2_HR.png',
                         # RRDN gans
                         'Set5_SR/Set5/image_SRF_4/img_001_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_002_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_003_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_004_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_005_SRF_4_HR.png',
                         # Noise cancel artefact: psnr-large, psnr-small, gans
                         # psnr-large
                         'Set5_SR/Set5/image_SRF_2/img_001_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_002_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_003_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_004_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_005_SRF_2_HR.png',
                         # psnr-small
                         'Set5_SR/Set5/image_SRF_2/img_001_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_002_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_003_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_004_SRF_2_HR.png',
                         'Set5_SR/Set5/image_SRF_2/img_005_SRF_2_HR.png',
                         # gans
                         'Set5_SR/Set5/image_SRF_4/img_001_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_002_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_003_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_004_SRF_4_HR.png',
                         'Set5_SR/Set5/image_SRF_4/img_005_SRF_4_HR.png']

# super_resolution_image_files
# total: 35
super_resolution_image_files = [# RDN psnr-large
                                'output/RDN/img_001_SRF_2_RDN_psnr-large.png',
                                'output/RDN/img_002_SRF_2_RDN_psnr-large.png',
                                'output/RDN/img_003_SRF_2_RDN_psnr-large.png',
                                'output/RDN/img_004_SRF_2_RDN_psnr-large.png',
                                'output/RDN/img_005_SRF_2_RDN_psnr-large.png',
                                # RDN psnr-small
                                'output/RDN/img_001_SRF_2_RDN_psnr-small.png',
                                'output/RDN/img_002_SRF_2_RDN_psnr-small.png',
                                'output/RDN/img_003_SRF_2_RDN_psnr-small.png',
                                'output/RDN/img_004_SRF_2_RDN_psnr-small.png',
                                'output/RDN/img_005_SRF_2_RDN_psnr-small.png',
                                # RDN noise-cancel
                                'output/RDN/img_001_SRF_2_RDN_noise-cancel.png',
                                'output/RDN/img_002_SRF_2_RDN_noise-cancel.png',
                                'output/RDN/img_003_SRF_2_RDN_noise-cancel.png',
                                'output/RDN/img_004_SRF_2_RDN_noise-cancel.png',
                                'output/RDN/img_005_SRF_2_RDN_noise-cancel.png',
                                # RRDN gans
                                'output/RRDN_gans/img_001_SRF_4_RRDN_gans.png',
                                'output/RRDN_gans/img_002_SRF_4_RRDN_gans.png',
                                'output/RRDN_gans/img_003_SRF_4_RRDN_gans.png',
                                'output/RRDN_gans/img_004_SRF_4_RRDN_gans.png',
                                'output/RRDN_gans/img_005_SRF_4_RRDN_gans.png',
                                # Noise cancel artefact: psnr-large, psnr-small, gans
                                # psnr-large
                                'output/Noise/img_001_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_002_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_003_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_004_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_005_SRF_2_RDN_psnr-large_noise_cancel.png',
                                # psnr-small
                                'output/Noise/img_001_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_002_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_003_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_004_SRF_2_RDN_psnr-large_noise_cancel.png',
                                'output/Noise/img_005_SRF_2_RDN_psnr-large_noise_cancel.png',
                                # gans
                                'output/Noise/img_001_SRF_4_RRDN_gans_noise_cancel.png',
                                'output/Noise/img_002_SRF_4_RRDN_gans_noise_cancel.png',
                                'output/Noise/img_003_SRF_4_RRDN_gans_noise_cancel.png',
                                'output/Noise/img_004_SRF_4_RRDN_gans_noise_cancel.png',
                                'output/Noise/img_005_SRF_4_RRDN_gans_noise_cancel.png']

# file_path
file_path = "full-psnr-output.txt"

def calculate_psnr(original, distorted):
    # Ensure images have the same shape and type
    # if original.shape != distorted.shape or original.dtype != distorted.dtype:
    if distorted.shape != original.shape or distorted.dtype != original.dtype:
        raise ValueError("Distorted images must have the same shape and type as reference images!")

    # Convert images to float32 for accurate calculations
    original = original.astype(np.float32)
    distorted = distorted.astype(np.float32)

    # Calculate mean-squared-error (MSE)
    mse = np.mean((original - distorted) ** 2)

    # Maximum pixel value: 255
    max_pixel_value = 255.0

    # Calculate peak-signal-to-noise-ratio (PSNR)
    # psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    ####################
    # log10(a ** b) = b * log10(a)
    # log10(a / b) = log10(a) - log10(b)
    ####################
    psnr = (20 * np.log10(max_pixel_value)) - (10 * np.log10(mse))

    return psnr

if __name__ == "__main__":

    for i in range(35):
        # Load original images and distorted images
        # Reference images: the corresponding high-resolution images
        # Super-resolution images: the images executed by super-resolution methods
        original_image = cv2.imread(reference_image_files[i]) # reference images
        distorted_image = cv2.imread(super_resolution_image_files[i]) # super-resolution images

        # Check image properties
        print(f"Original Image Shape: {original_image.shape}, Data Type: {original_image.dtype}")
        print(f"Distorted Image Shape: {distorted_image.shape}, Data Type: {distorted_image.dtype}")

        # Resize distorted image to match the size of the original image
        distorted_image = cv2.resize(distorted_image, (original_image.shape[1], original_image.shape[0]))

        # Check resized image properties
        print(f"Resized Distorted Image Shape: {distorted_image.shape}, Data Type: {distorted_image.dtype}")

        # Calculate PSNR
        psnr_value = calculate_psnr(original_image, distorted_image)

        print(f"PSNR: {psnr_value:.2f} dB")

        # Open the file as append mode
        with open(file_path, 'a') as file:
            # Append value to the file
            file.write("Index is: ")
            file.write(str(i)) # Convert integer to string
            file.write(" ")
            file.write("Reference image: ")
            file.write(reference_image_files[i])
            file.write(" ")
            file.write("Super-resolution image: ")
            file.write(super_resolution_image_files[i])
            file.write(" ")
            file.write("PSNR: ")
            file.write(str(psnr_value)) # PSNR value
            file.write("\n\r") # line feed and carriage return
