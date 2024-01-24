################################################
# Structural Similarity (SSIM)
# https://www.cns.nyu.edu/~lcv/ssim/
# https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
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
# Skimage and OpenCV packages are required in this script
# Instruction of installing Skimage and OpenCV:
# pip install scikit-image
# pip install opencv-python
#
# The SSIM is a well-known quality metric used to
# measure the similarity between two images. It was developed
# by Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
# “Image quality assessment: from error visibility to structural
# similarity”, IEEE Transactions on Image Processing, vol. 13,
# no. 4, pp. 600-612, 2004., and is considered to be correlated with
# the quality perception of the human visual system (HVS).
# Instead of using traditional error summation methods,
# the SSIM is designed by modeling any image distortion as a
# combination of three factors that are loss of correlation,
# luminance distortion and contrast distortion.
#
# SSIM = {
# 1.loss of correlation
# 2.luminance distortion
# 3.contrast distortion
# }
#
# S(x, y) = f(l(x, y), c(x, y), s(x, y)).
# 1. Symmetry: S(x, y) = S(y, x);
# 2. Boundedness: S(x, y) ≤ 1;
# 3. Unique maximum: S(x, y) = 1 if and only if x = y (in
# discrete representations, xi = yi for all i = 1, 2, · · · , N);
#
# LaTeX formula:
# Structural Similarity (SSIM)
# \text{SSIM}(f, g) = l(f, g) \cdot c(f, g) \cdot s(f, g)
# \begin{align*}
# l(f, g) &= \frac{2\mu_f\mu_g + C_1}{\mu_f^2 + \mu_g^2 + C_1} \\
# c(f, g) &= \frac{2\sigma_f\sigma_g + C_2}{\sigma_f^2 + \sigma_g^2 + C_2} \\
# s(f, g) &= \frac{\sigma_{fg} + C_3}{\sigma_f \sigma_g + C_3}
# \end{align*}
#
# Proposed Idea:
# [Signal x]  ->  [Luminance Measurement]
#         [+]    [Contrast Measurement]
#             [\]                                     [Luminance Comparison]          |
#                                                                                     v
#                                                     [Contrast Comparison]   ->  [Combination]   ->  Similarity Measure
#                                                                                     ^
#                                                     [Structure Comparison]          |
# [Signal y]  ->  [Luminance Measurement]
#         [+]    [Contrast Measurement]
#             [\]
#
# The purpose of this script is to execute SSIM IQA metric
# [SSIM]
#
# Please create a 'txt' file called 'full-ssim-output.txt' under the same directory (folder)
# before running this script. Otherwise, it may have some potential issues.
################################################

import cv2
from skimage.metrics import structural_similarity as ssim

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
file_path = "full-ssim-output.txt"

if __name__ == "__main__":

    for i in range(35):
        # Load reference and comparison images
        # Reference images: the corresponding high-resolution images
        # Super-resolution images: the images executed by super-resolution methods
        ref_image = cv2.imread(reference_image_files[i])
        comp_image = cv2.imread(super_resolution_image_files[i])

        # Check image properties
        print(f"Reference Image Shape: {ref_image.shape}, Data Type: {ref_image.dtype}")
        print(f"Comparison Image Shape: {comp_image.shape}, Data Type: {comp_image.dtype}")

        # Resize images
        comp_image = cv2.resize(comp_image, (ref_image.shape[1], ref_image.shape[0]))

        # Check resized image properties
        print(f"Resized Comparison Image Shape: {comp_image.shape}, Data Type: {comp_image.dtype}")

        # Convert images to grayscale
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim_score, _ = ssim(ref_gray, comp_gray, full=True)

        print(f"SSIM Score: {ssim_score}")

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
            file.write("SSIM: ")
            file.write(str(ssim_score)) # SSIM score
            file.write("\n\r") # line feed and carriage return
