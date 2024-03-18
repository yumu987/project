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
# Bitrate (Data rate)
# Static bitrate of the image itself
# Without considering transmission time, channel effect, modulation effect and demodulation effect
# Without considering other unexpected factors
# Considering real image size as image depth
# Basic algorithm:
# total pixels = width * height
# bits per pixel = bit depth (image size) * 8 * channels
# img.dtype.itemsize will only output one because the size of one element of the data type of image in bytes is one
# bitrate = bits per pixel * total pixels
# bitrate = (bit depth * 8 * channels) * (width * height)
# The unit is bit/frame (kbit/frame) (mbit/frame) instead of bit/s (kbit/s) (mbit/s)
# The final unit is bits (kbits) (mbits)
# Algorithm source:
# https://knowledge.ni.com/KnowledgeArticleDetails?id=kA03q000000YI5bCAG&l=en-US#:~:text=1%20Width%20x%20Height%20%3D%20Resolution%20%28in%20pixels%29,Bytes%2Fframe%203%20Bytes%2Fframe%20x%20Frame%20Rate%20%3D%20Bytes%2Fsecond
# https://datacarpentry.org/image-processing/02-image-basics
# https://faceconverter.com/how-image-size-is-calculated/#:~:text=Here%E2%80%99s%20a%20quick%20summary%20of%20the%20steps%20to,1024%20to%20ascertain%20the%20image%20size%20in%20megabytes
################################################

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import os

input_bitrate_array = []
input_grayscale_bitrate_array = []
input_compressed_1_bitrate_array = []
input_compressed_25_bitrate_array = []
input_compressed_50_bitrate_array = []
input_compressed_75_bitrate_array = []
input_compressed_100_bitrate_array = []
input_encoded_bitrate_array = []
input_decoded_bitrate_array = []
nearest_neighbor_quarter_bitrate_array = []
bilinear_quarter_bitrate_array = []
bicubic_quarter_bitrate_array = []
lanczos_quarter_bitrate_array = []
pixel_area_relation_quarter_bitrate_array = []

input_size_array = []
input_grayscale_size_array = []
input_compressed_1_size_array = []
input_compressed_25_size_array = []
input_compressed_50_size_array = []
input_compressed_75_size_array = []
input_compressed_100_size_array = []
input_encoded_size_array = []
input_decoded_size_array = []
nearest_neighbor_quarter_size_array = []
bilinear_quarter_size_array = []
bicubic_quarter_size_array = []
lanczos_quarter_size_array = []
pixel_area_relation_quarter_size_array = []

def calculate_bitrate(image_path):
    try:
        ##################################################
        # Read input image
        img = cv2.imread(image_path)
        image = Image.open(image_path)
        # Check the channel property of image
        mode = image.mode
        # Print out the mode (type) of channels
        print("The mode of channels (mode) in image is:", mode)
        # Print out the number of channels
        print("The length of channels (mode) in image is:", len(mode))
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {image_path}")
        # Debugging
        # print("The type of image:")
        # print(type(img))
        ##################################################
        # Extract height, width and number of channels
        height, width, channels = img.shape
        # Based on the property of channels to define the number of channels
        channels = len(mode)
        # image.dtype: the data type of image
        # itemsize: the size of one element of the data type in bytes
        # Debugging
        # print("img.dtype.itemsize:")
        # print(img.dtype.itemsize)
        # bit_depth = img.dtype.itemsize * 8 # Convert byte to bits (1 byte = 8 bits)
        # Get the file size (data size) of the image
        # Consider the real file size (data size) of the image
        # Debugging
        print(f"The size of image in bytes is {os.path.getsize(image_path)}")
        bit_depth = os.path.getsize(image_path) * 8 # Convert bytes to bits (1 byte = 8 bits)
        # Each pixel in the image consists of multiple channels
        if channels is None: # Exception
            raise ValueError(f"Error: Unable to read {channels}")
        elif channels == 1: # Grayscale mode
            bits_per_pixel = bit_depth # Without considering colour channels
        else: # Colourful mode
            bits_per_pixel = bit_depth * channels # Consider colour channels
        ##################################################
        total_pixels = width * height # height * width
        bitrate = bits_per_pixel * total_pixels # colour depth * (width * height)
        ##################################################
        return bitrate
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
        return None
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")
        return None

def calculate_size(image_path):
    try:
        size = os.path.getsize(image_path)
        bits_size = size * 8 # Convert byte to bits (1 byte = 8 bits)
        mega_size = bits_size / (1024 * 1024) # Convert bits to megabits
        return mega_size
    except Exception as e:
        # Handle exceptions
        print(f"An exception occurred: {e}")
        return None

def plot_table(data):
    df = pd.DataFrame(data)
    df.fillna('-', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center', colLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(len(df.columns))))

    for (i, j), val in np.ndenumerate(df.values):
        if j == 0: # First column
            tab[(i+1, j)].set_facecolor('#dddddd') # Shade the first column
        if j == -1 or j == 0: # Header or first column
            tab[(i+1, j)].set_text_props(weight='bold') # Bold the font of the first column

    plt.suptitle('TABLE\nThe bitrate table', fontsize=14, weight='bold')

    pdf_file = 'bitrate.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    png_file = 'bitrate.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

    print("Computing bitrate and plotting table completed")

def plot_bar_chart(mean_input,
                   mean_input_grayscale,
                   mean_input_compressed_1,
                   mean_input_compressed_25,
                   mean_input_compressed_50,
                   mean_input_compressed_75,
                   mean_input_compressed_100,
                   mean_input_encoded,
                   mean_input_decoded,
                   mean_nearest_neighbor, 
                   mean_bilinear, 
                   mean_bicubic, 
                   mean_lanczos, 
                   mean_pixel_area_relation):
    # Sample data
    sample_name_array = [
        'Input', 
        'Grayscale',
        'Compressed 1',
        'Compressed 25',
        'Compressed 50',
        'Compressed 75',
        'Compressed 100',
        'Encoded',
        'Decoded',
        'Nearest-neighbor', 
        'Bilinear', 
        'Bicubic', 
        'Lanczos', 
        'Pixel area relation'
        ]
    sample_data_array = [
        mean_input, 
        mean_input_grayscale,
        mean_input_compressed_1,
        mean_input_compressed_25,
        mean_input_compressed_50,
        mean_input_compressed_75,
        mean_input_compressed_100,
        mean_input_encoded,
        mean_input_decoded,
        mean_nearest_neighbor, 
        mean_bilinear, 
        mean_bicubic, 
        mean_lanczos, 
        mean_pixel_area_relation
        ]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor", fontsize=8.2)
    # Title and label the bar chart
    plt.title('Average bitrate in 25 images')
    # plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Bitrate: Megabits')
    # Save the figure
    plt.savefig('Bitrate_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Debugging
    print("--------------------------------------------------")
    print(f"The mean bitrate of input in megabits is: {mean_input}")
    print(f"The mean bitrate of grayscale in megabits is: {mean_input_grayscale}")
    print(f"The mean bitrate of compressed 1 in megabits is: {mean_input_compressed_1}")
    print(f"The mean bitrate of compressed 25 in megabits is: {mean_input_compressed_25}")
    print(f"The mean bitrate of compressed 50 in megabits is: {mean_input_compressed_50}")
    print(f"The mean bitrate of compressed 75 in megabits is: {mean_input_compressed_75}")
    print(f"The mean bitrate of compressed 100 in megabits is: {mean_input_compressed_100}")
    print(f"The mean bitrate of encoded in megabits is: {mean_input_encoded}")
    print(f"The mean bitrate of decoded in megabits is: {mean_input_decoded}")
    print(f"The mean bitrate of nearest-neighbor in megabits is: {mean_nearest_neighbor}")
    print(f"The mean bitrate of bilinear in megabits is: {mean_bilinear}")
    print(f"The mean bitrate of bicubic in megabits is: {mean_bicubic}")
    print(f"The mean bitrate of lanczos in megabits is: {mean_lanczos}")
    print(f"The mean bitrate of pixel area relation in megabits is: {mean_pixel_area_relation}")
    print("--------------------------------------------------")
    # Indication of plotting bar chart completed
    print("Bitrate: Bar chart has been drawn")

def plot_size_bar_chart(mean_input,
                   mean_input_grayscale,
                   mean_input_compressed_1,
                   mean_input_compressed_25,
                   mean_input_compressed_50,
                   mean_input_compressed_75,
                   mean_input_compressed_100,
                   mean_input_encoded,
                   mean_input_decoded,
                   mean_nearest_neighbor, 
                   mean_bilinear, 
                   mean_bicubic, 
                   mean_lanczos, 
                   mean_pixel_area_relation):
    # Sample data
    sample_name_array = [
        'Input', 
        'Grayscale',
        'Compressed 1',
        'Compressed 25',
        'Compressed 50',
        'Compressed 75',
        'Compressed 100',
        'Encoded',
        'Decoded',
        'Nearest-neighbor', 
        'Bilinear', 
        'Bicubic', 
        'Lanczos', 
        'Pixel area relation'
        ]
    sample_data_array = [
        mean_input, 
        mean_input_grayscale,
        mean_input_compressed_1,
        mean_input_compressed_25,
        mean_input_compressed_50,
        mean_input_compressed_75,
        mean_input_compressed_100,
        mean_input_encoded,
        mean_input_decoded,
        mean_nearest_neighbor, 
        mean_bilinear, 
        mean_bicubic, 
        mean_lanczos, 
        mean_pixel_area_relation
        ]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor", fontsize=8.2)
    # Title and label the bar chart
    plt.title('Average size in 25 images')
    # plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Size: Megabits')
    # Save the figure
    plt.savefig('Size_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("Size: Bar chart has been drawn")

def main():
    # Unit conversion
    # x bits = x/1024 kbits = (x/1024)/1024 mbits ([x/(1024 * 1024)] mbits)

    input_files = ['input/{}.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i]
        input_bitrate = calculate_bitrate(input_batch)
        input_bitrate = input_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_bitrate_array.append(input_bitrate)
        input_size = calculate_size(input_batch)
        input_size_array.append(input_size)
    tmp_input = 0.0
    tmp_input_size = 0.0
    for i in range(25):
        tmp_input = tmp_input + input_bitrate_array[i]
        tmp_input_size = tmp_input_size + input_size_array[i]
    mean_input = tmp_input/25 # Megabits/frame
    mean_input_size = tmp_input_size/25

    input_grayscale_files = ['input_grayscale/{}_grayscale.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_grayscale_files), 1):
        input_grayscale_batch = input_grayscale_files[i]
        input_grayscale_bitrate = calculate_bitrate(input_grayscale_batch)
        input_grayscale_bitrate = input_grayscale_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_grayscale_bitrate_array.append(input_grayscale_bitrate)
        input_grayscale_size = calculate_size(input_grayscale_batch)
        input_grayscale_size_array.append(input_grayscale_size)
    tmp_input_grayscale = 0.0
    tmp_input_grayscale_size = 0.0
    for i in range(25):
        tmp_input_grayscale = tmp_input_grayscale + input_grayscale_bitrate_array[i]
        tmp_input_grayscale_size = tmp_input_grayscale_size + input_grayscale_size_array[i]
    mean_input_grayscale = tmp_input_grayscale/25 # Megabits/frame
    mean_input_grayscale_size = tmp_input_grayscale_size/25

    input_compressed_1_files = ['input_compressed_1/{}_compressed_1.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_1_files), 1):
        input_compressed_1_batch = input_compressed_1_files[i]
        input_compressed_1_bitrate = calculate_bitrate(input_compressed_1_batch)
        input_compressed_1_bitrate = input_compressed_1_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_compressed_1_bitrate_array.append(input_compressed_1_bitrate)
        input_compressed_1_size = calculate_size(input_compressed_1_batch)
        input_compressed_1_size_array.append(input_compressed_1_size)
    tmp_input_compressed_1 = 0.0
    tmp_input_compressed_1_size = 0.0
    for i in range(25):
        tmp_input_compressed_1 = tmp_input_compressed_1 + input_compressed_1_bitrate_array[i]
        tmp_input_compressed_1_size = tmp_input_compressed_1_size + input_compressed_1_size_array[i]
    mean_input_compressed_1 = tmp_input_compressed_1/25 # Megabits/frame
    mean_input_compressed_1_size = tmp_input_compressed_1_size/25

    input_compressed_25_files = ['input_compressed_25/{}_compressed_25.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_25_files), 1):
        input_compressed_25_batch = input_compressed_25_files[i]
        input_compressed_25_bitrate = calculate_bitrate(input_compressed_25_batch)
        input_compressed_25_bitrate = input_compressed_25_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_compressed_25_bitrate_array.append(input_compressed_25_bitrate)
        input_compressed_25_size = calculate_size(input_compressed_25_batch)
        input_compressed_25_size_array.append(input_compressed_25_size)
    tmp_input_compressed_25 = 0.0
    tmp_input_compressed_25_size = 0.0
    for i in range(25):
        tmp_input_compressed_25 = tmp_input_compressed_25 + input_compressed_25_bitrate_array[i]
        tmp_input_compressed_25_size = tmp_input_compressed_25_size + input_compressed_25_size_array[i]
    mean_input_compressed_25 = tmp_input_compressed_25/25 # Megabits/frame
    mean_input_compressed_25_size = tmp_input_compressed_25_size/25

    input_compressed_50_files = ['input_compressed_50/{}_compressed_50.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_50_files), 1):
        input_compressed_50_batch = input_compressed_50_files[i]
        input_compressed_50_bitrate = calculate_bitrate(input_compressed_50_batch)
        input_compressed_50_bitrate = input_compressed_50_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_compressed_50_bitrate_array.append(input_compressed_50_bitrate)
        input_compressed_50_size = calculate_size(input_compressed_50_batch)
        input_compressed_50_size_array.append(input_compressed_50_size)
    tmp_input_compressed_50 = 0.0
    tmp_input_compressed_50_size = 0.0
    for i in range(25):
        tmp_input_compressed_50 = tmp_input_compressed_50 + input_compressed_50_bitrate_array[i]
        tmp_input_compressed_50_size = tmp_input_compressed_50_size + input_compressed_50_size_array[i]
    mean_input_compressed_50 = tmp_input_compressed_50/25 # Megabits/frame
    mean_input_compressed_50_size = tmp_input_compressed_50_size/25

    input_compressed_75_files = ['input_compressed_75/{}_compressed_75.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_75_files), 1):
        input_compressed_75_batch = input_compressed_75_files[i]
        input_compressed_75_bitrate = calculate_bitrate(input_compressed_75_batch)
        input_compressed_75_bitrate = input_compressed_75_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_compressed_75_bitrate_array.append(input_compressed_75_bitrate)
        input_compressed_75_size = calculate_size(input_compressed_75_batch)
        input_compressed_75_size_array.append(input_compressed_75_size)
    tmp_input_compressed_75 = 0.0
    tmp_input_compressed_75_size = 0.0
    for i in range(25):
        tmp_input_compressed_75 = tmp_input_compressed_75 + input_compressed_75_bitrate_array[i]
        tmp_input_compressed_75_size = tmp_input_compressed_75_size + input_compressed_75_size_array[i]
    mean_input_compressed_75 = tmp_input_compressed_75/25 # Megabits/frame
    mean_input_compressed_75_size = tmp_input_compressed_75_size/25

    input_compressed_100_files = ['input_compressed_100/{}_compressed_100.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_compressed_100_files), 1):
        input_compressed_100_batch = input_compressed_100_files[i]
        input_compressed_100_bitrate = calculate_bitrate(input_compressed_100_batch)
        input_compressed_100_bitrate = input_compressed_100_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_compressed_100_bitrate_array.append(input_compressed_100_bitrate)
        input_compressed_100_size = calculate_size(input_compressed_100_batch)
        input_compressed_100_size_array.append(input_compressed_100_size)
    tmp_input_compressed_100 = 0.0
    tmp_input_compressed_100_size = 0.0
    for i in range(25):
        tmp_input_compressed_100 = tmp_input_compressed_100 + input_compressed_100_bitrate_array[i]
        tmp_input_compressed_100_size = tmp_input_compressed_100_size + input_compressed_100_size_array[i]
    mean_input_compressed_100 = tmp_input_compressed_100/25 # Megabits/frame
    mean_input_compressed_100_size = tmp_input_compressed_100_size/25

    input_encoded_files = ['input_encoded/{}_encoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_encoded_files), 1):
        input_encoded_batch = input_encoded_files[i]
        input_encoded_bitrate = calculate_bitrate(input_encoded_batch)
        input_encoded_bitrate = input_encoded_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_encoded_bitrate_array.append(input_encoded_bitrate)
        input_encoded_size = calculate_size(input_encoded_batch)
        input_encoded_size_array.append(input_encoded_size)
    tmp_input_encoded = 0.0
    tmp_input_encoded_size = 0.0
    for i in range(25):
        tmp_input_encoded = tmp_input_encoded + input_encoded_bitrate_array[i]
        tmp_input_encoded_size = tmp_input_encoded_size + input_encoded_size_array[i]
    mean_input_encoded = tmp_input_encoded/25 # Megabits/frame
    mean_input_encoded_size = tmp_input_encoded_size/25

    input_decoded_files = ['input_decoded/{}_decoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_decoded_files), 1):
        input_decoded_batch = input_decoded_files[i]
        input_decoded_bitrate = calculate_bitrate(input_decoded_batch)
        input_decoded_bitrate = input_decoded_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        input_decoded_bitrate_array.append(input_decoded_bitrate)
        input_decoded_size = calculate_size(input_decoded_batch)
        input_decoded_size_array.append(input_decoded_size)
    tmp_input_decoded = 0.0
    tmp_input_decoded_size = 0.0
    for i in range(25):
        tmp_input_decoded = tmp_input_decoded + input_decoded_bitrate_array[i]
        tmp_input_decoded_size = tmp_input_decoded_size + input_decoded_size_array[i]
    mean_input_decoded = tmp_input_decoded/25 # Megabits/frame
    mean_input_decoded_size = tmp_input_decoded_size/25

    nearest_neighbor_files = ['nearest_neighbor_quarter/{}_nearest_neighbor_quarter.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(nearest_neighbor_files), 1):
        nearest_neighbor_batch = nearest_neighbor_files[i]
        nearest_neighbor_bitrate = calculate_bitrate(nearest_neighbor_batch)
        nearest_neighbor_bitrate = nearest_neighbor_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        nearest_neighbor_quarter_bitrate_array.append(nearest_neighbor_bitrate)
        nearest_neighbor_size = calculate_size(nearest_neighbor_batch)
        nearest_neighbor_quarter_size_array.append(nearest_neighbor_size)
    tmp_nearest_neighbor = 0.0
    tmp_nearest_neighbor_size = 0.0
    for i in range(25):
        tmp_nearest_neighbor = tmp_nearest_neighbor + nearest_neighbor_quarter_bitrate_array[i]
        tmp_nearest_neighbor_size = tmp_nearest_neighbor_size + nearest_neighbor_quarter_size_array[i]
    mean_nearest_neighbor = tmp_nearest_neighbor/25 # Megabits/frame
    mean_nearest_neighbor_size = tmp_nearest_neighbor_size/25

    bilinear_files = ['bilinear_quarter/{}_bilinear_quarter.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(bilinear_files), 1):
        bilinear_batch = bilinear_files[i]
        bilinear_bitrate = calculate_bitrate(bilinear_batch)
        bilinear_bitrate = bilinear_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        bilinear_quarter_bitrate_array.append(bilinear_bitrate)
        bilinear_size = calculate_size(bilinear_batch)
        bilinear_quarter_size_array.append(bilinear_size)
    tmp_bilinear = 0.0
    tmp_bilinear_size = 0.0
    for i in range(25):
        tmp_bilinear = tmp_bilinear + bilinear_quarter_bitrate_array[i]
        tmp_bilinear_size = tmp_bilinear_size + bilinear_quarter_size_array[i]
    mean_bilinear = tmp_bilinear/25 # Megabits/frame
    mean_bilinear_size = tmp_bilinear_size/25

    bicubic_files = ['bicubic_quarter/{}_bicubic_quarter.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(bicubic_files), 1):
        bicubic_batch = bicubic_files[i]
        bicubic_bitrate = calculate_bitrate(bicubic_batch)
        bicubic_bitrate = bicubic_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        bicubic_quarter_bitrate_array.append(bicubic_bitrate)
        bicubic_size = calculate_size(bicubic_batch)
        bicubic_quarter_size_array.append(bicubic_size)
    tmp_bicubic = 0.0
    tmp_bicubic_size = 0.0
    for i in range(25):
        tmp_bicubic = tmp_bicubic + bicubic_quarter_bitrate_array[i]
        tmp_bicubic_size = tmp_bicubic_size + bicubic_quarter_size_array[i]
    mean_bicubic = tmp_bicubic/25 # Megabits/frame
    mean_bicubic_size = tmp_bicubic_size/25

    lanczos_files = ['lanczos_quarter/{}_lanczos_quarter.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(lanczos_files), 1):
        lanczos_batch = lanczos_files[i]
        lanczos_bitrate = calculate_bitrate(lanczos_batch)
        lanczos_bitrate = lanczos_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        lanczos_quarter_bitrate_array.append(lanczos_bitrate)
        lanczos_size = calculate_size(lanczos_batch)
        lanczos_quarter_size_array.append(lanczos_size)
    tmp_lanczos = 0.0
    tmp_lanczos_size = 0.0
    for i in range(25):
        tmp_lanczos = tmp_lanczos + lanczos_quarter_bitrate_array[i]
        tmp_lanczos_size = tmp_lanczos_size + lanczos_quarter_size_array[i]
    mean_lanczos = tmp_lanczos/25 # Megabits/frame
    mean_lanczos_size = tmp_lanczos_size/25

    pixel_area_relation_files = ['pixel_area_relation_quarter/{}_pixel_area_relation_quarter.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(pixel_area_relation_files), 1):
        pixel_area_relation_batch = pixel_area_relation_files[i]
        pixel_area_relation_bitrate = calculate_bitrate(pixel_area_relation_batch)
        pixel_area_relation_bitrate = pixel_area_relation_bitrate/(1024 * 1024) # Convert bits/frame to megabits/frame
        pixel_area_relation_quarter_bitrate_array.append(pixel_area_relation_bitrate)
        pixel_area_relation_size = calculate_size(pixel_area_relation_batch)
        pixel_area_relation_quarter_size_array.append(pixel_area_relation_size)
    tmp_pixel_area_relation = 0.0
    tmp_pixel_area_relation_size = 0.0
    for i in range(25):
        tmp_pixel_area_relation = tmp_pixel_area_relation + pixel_area_relation_quarter_bitrate_array[i]
        tmp_pixel_area_relation_size = tmp_pixel_area_relation_size + pixel_area_relation_quarter_size_array[i]
    mean_pixel_area_relation = tmp_pixel_area_relation/25 # Megabits/frame
    mean_pixel_area_relation_size = tmp_pixel_area_relation_size/25

    plot_bar_chart(
        mean_input,
        mean_input_grayscale,
        mean_input_compressed_1,
        mean_input_compressed_25,
        mean_input_compressed_50,
        mean_input_compressed_75,
        mean_input_compressed_100,
        mean_input_encoded,
        mean_input_decoded,
        mean_nearest_neighbor,
        mean_bilinear,
        mean_bicubic,
        mean_lanczos,
        mean_pixel_area_relation
    )

    plot_size_bar_chart(
        mean_input_size,
        mean_input_grayscale_size,
        mean_input_compressed_1_size,
        mean_input_compressed_25_size,
        mean_input_compressed_50_size,
        mean_input_compressed_75_size,
        mean_input_compressed_100_size,
        mean_input_encoded_size,
        mean_input_decoded_size,
        mean_nearest_neighbor_size,
        mean_bilinear_size,
        mean_bicubic_size,
        mean_lanczos_size,
        mean_pixel_area_relation_size
    )

    # image_path = input("Enter the path to the image file: ")
    # bitrate = calculate_bitrate(image_path)
    # print("Bitrate:", bitrate, "bits/frame")
    # print("Bitrate:", bitrate/1024, "kilobits/frame")
    # print("Bitrate:", bitrate/(1024 * 1024), "megabits/frame")

if __name__ == "__main__":
    main()
