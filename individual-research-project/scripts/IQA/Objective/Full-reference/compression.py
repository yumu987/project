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
################################################

################################################
import cv2
import numpy as np
import mse
import psnr
from skimage.metrics import structural_similarity as ssim
################################################

################################################
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
################################################

input_compressed_1_mse_array = []
input_compressed_1_psnr_array = []
input_compressed_1_ssim_array = []

input_compressed_25_mse_array = []
input_compressed_25_psnr_array = []
input_compressed_25_ssim_array = []

input_compressed_50_mse_array = []
input_compressed_50_psnr_array = []
input_compressed_50_ssim_array = []

input_compressed_75_mse_array = []
input_compressed_75_psnr_array = []
input_compressed_75_ssim_array = []

input_compressed_100_mse_array = []
input_compressed_100_psnr_array = []
input_compressed_100_ssim_array = []

input_encoded_mse_array = []
input_encoded_psnr_array = []
input_encoded_ssim_array = []

input_decoded_mse_array = []
input_decoded_psnr_array = []
input_decoded_ssim_array = []

def compute_data(reference_image, test_image):
    mse_value = mse.mse(reference_image, test_image)
    psnr_value = psnr.psnr(reference_image, test_image)
    ssim_value, _ = ssim(reference_image, test_image, full=True)
    print(f"The image has MSE:{mse_value}, PSNR:{psnr_value}, SSIM:{ssim_value}")
    return mse_value, psnr_value, ssim_value

def plot_mse_bar_chart(mean_compressed_1, 
                       mean_compressed_25, 
                       mean_compressed_50, 
                       mean_compressed_75, 
                       mean_compressed_100,
                       mean_compressed_encoded,
                       mean_compressed_decoded):
    # Sample data
    sample_name_array = [
        'Compressed 1', 
        'Compressed 25', 
        'Compressed 50', 
        'Compressed 75', 
        'Compressed 100',
        'Encoded',
        'Decoded'
        ]
    sample_data_array = [
        mean_compressed_1, 
        mean_compressed_25, 
        mean_compressed_50, 
        mean_compressed_75, 
        mean_compressed_100,
        mean_compressed_encoded,
        mean_compressed_decoded
        ]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Average MSE in 25 images')
    # plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Mean Squared Error')
    # Save the figure
    plt.savefig('Compression_MSE_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("Compression MSE: Bar chart has been drawn")

def plot_psnr_bar_chart(mean_compressed_1, 
                       mean_compressed_25, 
                       mean_compressed_50, 
                       mean_compressed_75, 
                       mean_compressed_100,
                       mean_compressed_encoded,
                       mean_compressed_decoded):
    # Sample data
    sample_name_array = [
        'Compressed 1', 
        'Compressed 25', 
        'Compressed 50', 
        'Compressed 75', 
        'Compressed 100',
        'Encoded',
        'Decoded'
        ]
    sample_data_array = [
        mean_compressed_1, 
        mean_compressed_25, 
        mean_compressed_50, 
        mean_compressed_75, 
        mean_compressed_100,
        mean_compressed_encoded,
        mean_compressed_decoded
        ]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Average PSNR in 25 images')
    # plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Peak-signal-to-noise-ratio')
    # Save the figure
    plt.savefig('Compression_PSNR_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("Compression PSNR: Bar chart has been drawn")

def plot_ssim_bar_chart(mean_compressed_1, 
                       mean_compressed_25, 
                       mean_compressed_50, 
                       mean_compressed_75, 
                       mean_compressed_100,
                       mean_compressed_encoded,
                       mean_compressed_decoded):
    # Sample data
    sample_name_array = [
        'Compressed 1', 
        'Compressed 25', 
        'Compressed 50', 
        'Compressed 75', 
        'Compressed 100',
        'Encoded',
        'Decoded'
        ]
    sample_data_array = [
        mean_compressed_1, 
        mean_compressed_25, 
        mean_compressed_50, 
        mean_compressed_75, 
        mean_compressed_100,
        mean_compressed_encoded,
        mean_compressed_decoded
        ]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Average SSIM in 25 images')
    # plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Structural Similarity Index')
    # Save the figure
    plt.savefig('Compression_SSIM_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("Compression SSIM: Bar chart has been drawn")

def main():
    # Reference images
    input_files = ['input/{}.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]

    # Test images
    input_compressed_1_files = ['input_compressed_1/{}_compressed_1.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_1_files), 1):
        input_img = cv2.imread(input_files[i])
        compressed_1_img = cv2.imread(input_compressed_1_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        compressed_1_grayscale = cv2.cvtColor(compressed_1_img, cv2.COLOR_BGR2GRAY)
        compressed_1_mse, compressed_1_psnr, compressed_1_ssim = compute_data(input_grayscale, compressed_1_grayscale)
        input_compressed_1_mse_array.append(compressed_1_mse)
        input_compressed_1_psnr_array.append(compressed_1_psnr)
        input_compressed_1_ssim_array.append(compressed_1_ssim)
    tmp_input_compressed_1_mse = 0.0
    tmp_input_compressed_1_psnr = 0.0
    tmp_input_compressed_1_ssim = 0.0
    for i in range(25):
        tmp_input_compressed_1_mse = tmp_input_compressed_1_mse + input_compressed_1_mse_array[i]
        tmp_input_compressed_1_psnr = tmp_input_compressed_1_psnr + input_compressed_1_psnr_array[i]
        tmp_input_compressed_1_ssim = tmp_input_compressed_1_ssim + input_compressed_1_ssim_array[i]
    mean_compressed_1_mse = tmp_input_compressed_1_mse/25
    mean_compressed_1_psnr = tmp_input_compressed_1_psnr/25
    mean_compressed_1_ssim = tmp_input_compressed_1_ssim/25

    # Test images
    input_compressed_25_files = ['input_compressed_25/{}_compressed_25.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_25_files), 1):
        input_img = cv2.imread(input_files[i])
        compressed_25_img = cv2.imread(input_compressed_25_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        compressed_25_grayscale = cv2.cvtColor(compressed_25_img, cv2.COLOR_BGR2GRAY)
        compressed_25_mse, compressed_25_psnr, compressed_25_ssim = compute_data(input_grayscale, compressed_25_grayscale)
        input_compressed_25_mse_array.append(compressed_25_mse)
        input_compressed_25_psnr_array.append(compressed_25_psnr)
        input_compressed_25_ssim_array.append(compressed_25_ssim)
    tmp_input_compressed_25_mse = 0.0
    tmp_input_compressed_25_psnr = 0.0
    tmp_input_compressed_25_ssim = 0.0
    for i in range(25):
        tmp_input_compressed_25_mse = tmp_input_compressed_25_mse + input_compressed_25_mse_array[i]
        tmp_input_compressed_25_psnr = tmp_input_compressed_25_psnr + input_compressed_25_psnr_array[i]
        tmp_input_compressed_25_ssim = tmp_input_compressed_25_ssim + input_compressed_25_ssim_array[i]
    mean_compressed_25_mse = tmp_input_compressed_25_mse/25
    mean_compressed_25_psnr = tmp_input_compressed_25_psnr/25
    mean_compressed_25_ssim = tmp_input_compressed_25_ssim/25

    # Test images
    input_compressed_50_files = ['input_compressed_50/{}_compressed_50.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_50_files), 1):
        input_img = cv2.imread(input_files[i])
        compressed_50_img = cv2.imread(input_compressed_50_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        compressed_50_grayscale = cv2.cvtColor(compressed_50_img, cv2.COLOR_BGR2GRAY)
        compressed_50_mse, compressed_50_psnr, compressed_50_ssim = compute_data(input_grayscale, compressed_50_grayscale)
        input_compressed_50_mse_array.append(compressed_50_mse)
        input_compressed_50_psnr_array.append(compressed_50_psnr)
        input_compressed_50_ssim_array.append(compressed_50_ssim)
    tmp_input_compressed_50_mse = 0.0
    tmp_input_compressed_50_psnr = 0.0
    tmp_input_compressed_50_ssim = 0.0
    for i in range(25):
        tmp_input_compressed_50_mse = tmp_input_compressed_50_mse + input_compressed_50_mse_array[i]
        tmp_input_compressed_50_psnr = tmp_input_compressed_50_psnr + input_compressed_50_psnr_array[i]
        tmp_input_compressed_50_ssim = tmp_input_compressed_50_ssim + input_compressed_50_ssim_array[i]
    mean_compressed_50_mse = tmp_input_compressed_50_mse/25
    mean_compressed_50_psnr = tmp_input_compressed_50_psnr/25
    mean_compressed_50_ssim = tmp_input_compressed_50_ssim/25

    # Test images
    input_compressed_75_files = ['input_compressed_75/{}_compressed_75.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_75_files), 1):
        input_img = cv2.imread(input_files[i])
        compressed_75_img = cv2.imread(input_compressed_75_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        compressed_75_grayscale = cv2.cvtColor(compressed_75_img, cv2.COLOR_BGR2GRAY)
        compressed_75_mse, compressed_75_psnr, compressed_75_ssim = compute_data(input_grayscale, compressed_75_grayscale)
        input_compressed_75_mse_array.append(compressed_75_mse)
        input_compressed_75_psnr_array.append(compressed_75_psnr)
        input_compressed_75_ssim_array.append(compressed_75_ssim)
    tmp_input_compressed_75_mse = 0.0
    tmp_input_compressed_75_psnr = 0.0
    tmp_input_compressed_75_ssim = 0.0
    for i in range(25):
        tmp_input_compressed_75_mse = tmp_input_compressed_75_mse + input_compressed_75_mse_array[i]
        tmp_input_compressed_75_psnr = tmp_input_compressed_75_psnr + input_compressed_75_psnr_array[i]
        tmp_input_compressed_75_ssim = tmp_input_compressed_75_ssim + input_compressed_75_ssim_array[i]
    mean_compressed_75_mse = tmp_input_compressed_75_mse/25
    mean_compressed_75_psnr = tmp_input_compressed_75_psnr/25
    mean_compressed_75_ssim = tmp_input_compressed_75_ssim/25

    # Test images
    input_compressed_100_files = ['input_compressed_100/{}_compressed_100.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_100_files), 1):
        input_img = cv2.imread(input_files[i])
        compressed_100_img = cv2.imread(input_compressed_100_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        compressed_100_grayscale = cv2.cvtColor(compressed_100_img, cv2.COLOR_BGR2GRAY)
        compressed_100_mse, compressed_100_psnr, compressed_100_ssim = compute_data(input_grayscale, compressed_100_grayscale)
        input_compressed_100_mse_array.append(compressed_100_mse)
        input_compressed_100_psnr_array.append(compressed_100_psnr)
        input_compressed_100_ssim_array.append(compressed_100_ssim)
    tmp_input_compressed_100_mse = 0.0
    tmp_input_compressed_100_psnr = 0.0
    tmp_input_compressed_100_ssim = 0.0
    for i in range(25):
        tmp_input_compressed_100_mse = tmp_input_compressed_100_mse + input_compressed_100_mse_array[i]
        tmp_input_compressed_100_psnr = tmp_input_compressed_100_psnr + input_compressed_100_psnr_array[i]
        tmp_input_compressed_100_ssim = tmp_input_compressed_100_ssim + input_compressed_100_ssim_array[i]
    mean_compressed_100_mse = tmp_input_compressed_100_mse/25
    mean_compressed_100_psnr = tmp_input_compressed_100_psnr/25
    mean_compressed_100_ssim = tmp_input_compressed_100_ssim/25

    # Test images
    input_encoded_files = ['input_encoded/{}_encoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_encoded_files), 1):
        input_img = cv2.imread(input_files[i])
        encoded_img = cv2.imread(input_encoded_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        encoded_grayscale = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2GRAY)
        encoded_mse, encoded_psnr, encoded_ssim = compute_data(input_grayscale, encoded_grayscale)
        input_encoded_mse_array.append(encoded_mse)
        input_encoded_psnr_array.append(encoded_psnr)
        input_encoded_ssim_array.append(encoded_ssim)
    tmp_input_encoded_mse = 0.0
    tmp_input_encoded_psnr = 0.0
    tmp_input_encoded_ssim = 0.0
    for i in range(25):
        tmp_input_encoded_mse = tmp_input_encoded_mse + input_encoded_mse_array[i]
        tmp_input_encoded_psnr = tmp_input_encoded_psnr + input_encoded_psnr_array[i]
        tmp_input_encoded_ssim = tmp_input_encoded_ssim + input_encoded_ssim_array[i]
    mean_encoded_mse = tmp_input_encoded_mse/25
    mean_encoded_psnr = tmp_input_encoded_psnr/25
    mean_encoded_ssim = tmp_input_encoded_ssim/25

    # Test images
    input_decoded_files = ['input_decoded/{}_decoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_decoded_files), 1):
        input_img = cv2.imread(input_files[i])
        decoded_img = cv2.imread(input_decoded_files[i])
        input_grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        decoded_grayscale = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
        decoded_mse, decoded_psnr, decoded_ssim = compute_data(input_grayscale, decoded_grayscale)
        input_decoded_mse_array.append(decoded_mse)
        input_decoded_psnr_array.append(decoded_psnr)
        input_decoded_ssim_array.append(decoded_ssim)
    tmp_input_decoded_mse = 0.0
    tmp_input_decoded_psnr = 0.0
    tmp_input_decoded_ssim = 0.0
    for i in range(25):
        tmp_input_decoded_mse = tmp_input_decoded_mse + input_decoded_mse_array[i]
        tmp_input_decoded_psnr = tmp_input_decoded_psnr + input_decoded_psnr_array[i]
        tmp_input_decoded_ssim = tmp_input_decoded_ssim + input_decoded_ssim_array[i]
    mean_decoded_mse = tmp_input_decoded_mse/25
    mean_decoded_psnr = tmp_input_decoded_psnr/25
    mean_decoded_ssim = tmp_input_decoded_ssim/25

    plot_mse_bar_chart(
        mean_compressed_1_mse,
        mean_compressed_25_mse,
        mean_compressed_50_mse,
        mean_compressed_75_mse,
        mean_compressed_100_mse,
        mean_encoded_mse,
        mean_decoded_mse
    )

    plot_psnr_bar_chart(
        mean_compressed_1_psnr,
        mean_compressed_25_psnr,
        mean_compressed_50_psnr,
        mean_compressed_75_psnr,
        mean_compressed_100_psnr,
        mean_encoded_psnr,
        mean_decoded_psnr
    )

    plot_ssim_bar_chart(
        mean_compressed_1_ssim,
        mean_compressed_25_ssim,
        mean_compressed_50_ssim,
        mean_compressed_75_ssim,
        mean_compressed_100_ssim,
        mean_encoded_ssim,
        mean_decoded_ssim
    )

if __name__ == "__main__":
    main()
