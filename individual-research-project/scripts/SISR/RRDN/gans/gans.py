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
# ISR and OpenCV packages are required in this script
# Instruction of installing ISR and OpenCV:
# pip install ISR
# pip install opencv-python
#
# The scale factor of RRDN gans model is 4
# For example, an input image has resolution (x, y)
# After RRDN gans model, the output image will have resolution (4x, 4y)
#
# The purpose of this script is to execute super-resolution of RRDN gans model
# [input_image] -> [RRDN(gans)] -> [output_image]
################################################

import cv2
from ISR.models import RRDN
import time
import logging

# input_image_files
input_image_files = ['Set5_SR/Set5/image_SRF_4/img_001_SRF_4_LR.png', 
                     'Set5_SR/Set5/image_SRF_4/img_002_SRF_4_LR.png', 
                     'Set5_SR/Set5/image_SRF_4/img_003_SRF_4_LR.png', 
                     'Set5_SR/Set5/image_SRF_4/img_004_SRF_4_LR.png', 
                     'Set5_SR/Set5/image_SRF_4/img_005_SRF_4_LR.png']

# output_image_files
output_image_files = ['output_image/img_001_SRF_4_RRDN_gans.png', 
                      'output_image/img_002_SRF_4_RRDN_gans.png', 
                      'output_image/img_003_SRF_4_RRDN_gans.png', 
                      'output_image/img_004_SRF_4_RRDN_gans.png', 
                      'output_image/img_005_SRF_4_RRDN_gans.png']

# super_resolve_image function
def super_resolve_image(input_path, output_path, model_path = 'gans'):

    try:
        # Read the input image
        img = cv2.imread(input_path)

        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_file}")
        
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

    # Configure logging to save to a file
    log_filename = 'log.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    # Log the separation line
    logging.info(f"################################################################################################")

    # Record the all five experiments start time
    start_time_total = time.time()

    # Load set 5 data set and perform super-resolution
    for i in range(5): # len(input_image_files) / len(output_image_files)
        # Record the start time
        start_time_experiment = time.time()
        input_file = input_image_files[i]
        output_file = output_image_files[i]
        super_resolve_image(input_file, output_file)
        # Record the end time
        end_time_experiment = time.time()
        # Calculate the elapsed time
        elapsed_time_experiment = end_time_experiment - start_time_experiment
        # Log the processing time
        logging.info(f"RRDN: gans | Single experiment processing time: {elapsed_time_experiment} seconds")
        # Log the separation line
        logging.info(f"################################################################################################")

    # Record the all five experiments end time
    end_time_total = time.time()

    # Calculate the total elapsed time
    elapsed_time_total = end_time_total - start_time_total

    # Calculate the mean value of total elapsed time
    elapsed_time_mean = elapsed_time_total / 5 # len(input_image_files) / len(output_image_files)

    # Log the separation line
    logging.info(f"################################################################################################")

    # Log the total processing time
    logging.info(f"RRDN: gans | Total processing time: {elapsed_time_total} seconds")

    # Log the mean processing time
    logging.info(f"RRDN: gans | Mean processing time: {elapsed_time_mean} seconds")

    # Log the separation line
    logging.info(f"################################################################################################")

    # Indication of processing time of RRDN gans model
    print(f"RRDN: gans | Total processing time: {elapsed_time_total} seconds")

    # Indication of mean processing time of RRDN gans model
    print(f"RRDN: gans | Mean processing time: {elapsed_time_mean} seconds")

    # Indication of super-resolution completed
    print(f"RRDN: gans | Super-resolution completed")
