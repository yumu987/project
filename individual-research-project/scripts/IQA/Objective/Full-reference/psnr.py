################################################
# Peak-signal-to-noise-ratio (PSNR)
# https://arxiv.org/abs/1406.7799
# https://ieeexplore.ieee.org/document/8980171
# https://ieeexplore.ieee.org/document/5596999
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
# OpenCV, Matplotlib, Pandas and Numpy packages are required in this script
# Instruction of installing packages:
# pip install opencv-python
# pip install matplotlib
# pip install pandas
# pip install numpy
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
################################################

################################################
# Peak-Signal-to-Noise-Ratio
import cv2
import numpy as np
import mse
################################################

################################################
# Table
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
################################################

# GANs

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

nearest_neighbor_psnr_array = []
nearest_neighbor_name_array = []

bilinear_psnr_array = []
bilinear_name_array = []

bicubic_psnr_array = []
bicubic_name_array = []

lanczos_psnr_array = []
lanczos_name_array = []

pixel_area_relation_psnr_array = []
pixel_area_relation_name_array = []

def psnr(reference_image, test_image):
    """
    PSNR is the ratio of maximum 
    possible power of a signal and power of distortion
    """
    # Calculate Mean Squared Error
    mean_squared_error = mse.mse(reference_image, test_image)
    """
    where D denotes the dynamic range of pixel intensities, 
    e.g., for an 8 bits/pixel image we have D = 255
    """
    d = 255.0
    # log10(a ** b) = b * log10(a)
    # log10(a / b) = log10(a) - log10(b)
    # peak_signal_to_noise_ratio = 10 * np.log10((d ** 2) / mean_squared_error)
    peak_signal_to_noise_ratio = (20 * np.log10(d)) - (10 * np.log10(mean_squared_error))
    return peak_signal_to_noise_ratio

def plot_bar_chart(mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation):
    # Sample data
    sample_name_array = ['Nearest-neighbor', 'Bilinear', 'Bicubic', 'Lanczos', 'Pixel area relation']
    sample_data_array = [mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Average PSNR in 25 images')
    plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Peak-signal-to-noise-ratio')
    # Save the figure
    plt.savefig('PSNR_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("PSNR: Bar chart has been drawn")

def plot_table(data):
    # Data to be used to plot table
    # data = {}
    # data = data

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    # Replacing missing values with '-'
    df.fillna('-', inplace=True)

    # Create a figure to plot the table
    # Size should be adjusted to match the aspect ratio of your data
    fig, ax = plt.subplots(figsize=(12, 8))
    # Hide the axes
    ax.axis('off')

    tab = table(ax, df, loc='center', cellLoc='center', colLoc='center')
    tab.auto_set_font_size(False)
    # Adjust to match the font size in your image
    tab.set_fontsize(10)
    # Adjust column widths
    tab.auto_set_column_width(col=list(range(len(df.columns))))

    # Style the table
    for (i, j), val in np.ndenumerate(df.values):
        # First column
        if j == 0:
            # Shade the first column
            tab[(i+1, j)].set_facecolor('#dddddd')
        # Header or first column
        if j == -1 or j == 0:
            # Bold the font of first column
            tab[(i+1, j)].set_text_props(weight='bold')

    # Add the title
    # plt.suptitle('TABLE\nPeak-Signal-to-Noise-Ratio', fontsize=14, weight='bold')
    # Save the figure as a PDF
    pdf_file = 'PSNR.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    # Save the figure as an image (PNG)
    png_file = 'PSNR.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    # Close the matplotlib figure
    plt.close(fig)
    # Indication of plotting table completed
    print("Plotting table completed")

def main():
    # Peak-Signal-to-Noise-Ratio

    # Nearest-neighbor
    for i in range(25):
        try:
            reference_image = cv2.imread(input_image_files[i])
            test_image = cv2.imread(nearest_neighbor_gans_image_files[i])
            if reference_image is None:
                raise FileNotFoundError(f"Error: Unable to read reference image at {input_image_files[i]}")
            if test_image is None:
                raise FileNotFoundError(f"Error: Unable to read test image at {nearest_neighbor_gans_image_files[i]}")
            print("OpenCV converts image to grayscale image to avoid working with RGB channels")
            # Summing the squared differences across channels might not align with human perception
            reference_grayscale = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            test_grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if reference_grayscale.shape != test_grayscale.shape:
                print("Test image must have same dimensions as reference image for comparison\nTest image will be resized to reference image by bilinear interpolation")
                # cv2.resize() uses bilinear interpolation for upsampling
                # resize: x, y (width, height)
                # shape: height, width, channels
                # shape[1]: width, shape[0]: height
                test_grayscale = cv2.resize(test_grayscale, (reference_grayscale.shape[1], reference_grayscale.shape[0]))
            psnr_value_nearest = psnr(reference_grayscale, test_grayscale)
            # Append name and data into arrays
            nearest_neighbor_name_array.append(input_image_files[i])
            nearest_neighbor_psnr_array.append(psnr_value_nearest)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")

    # Bilinear
    for i in range(25):
        try:
            reference_image = cv2.imread(input_image_files[i])
            test_image = cv2.imread(bilinear_gans_image_files[i])
            if reference_image is None:
                raise FileNotFoundError(f"Error: Unable to read reference image at {input_image_files[i]}")
            if test_image is None:
                raise FileNotFoundError(f"Error: Unable to read test image at {bilinear_gans_image_files[i]}")
            print("OpenCV converts image to grayscale image to avoid working with RGB channels")
            # Summing the squared differences across channels might not align with human perception
            reference_grayscale = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            test_grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if reference_grayscale.shape != test_grayscale.shape:
                print("Test image must have same dimensions as reference image for comparison\nTest image will be resized to reference image by bilinear interpolation")
                # cv2.resize() uses bilinear interpolation for upsampling
                # resize: x, y (width, height)
                # shape: height, width, channels
                # shape[1]: width, shape[0]: height
                test_grayscale = cv2.resize(test_grayscale, (reference_grayscale.shape[1], reference_grayscale.shape[0]))
            psnr_value_bilinear = psnr(reference_grayscale, test_grayscale)
            # Append name and data into arrays
            bilinear_name_array.append(input_image_files[i])
            bilinear_psnr_array.append(psnr_value_bilinear)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")

    # Bicubic
    for i in range(25):
        try:
            reference_image = cv2.imread(input_image_files[i])
            test_image = cv2.imread(bicubic_gans_image_files[i])
            if reference_image is None:
                raise FileNotFoundError(f"Error: Unable to read reference image at {input_image_files[i]}")
            if test_image is None:
                raise FileNotFoundError(f"Error: Unable to read test image at {bicubic_gans_image_files[i]}")
            print("OpenCV converts image to grayscale image to avoid working with RGB channels")
            # Summing the squared differences across channels might not align with human perception
            reference_grayscale = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            test_grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if reference_grayscale.shape != test_grayscale.shape:
                print("Test image must have same dimensions as reference image for comparison\nTest image will be resized to reference image by bilinear interpolation")
                # cv2.resize() uses bilinear interpolation for upsampling
                # resize: x, y (width, height)
                # shape: height, width, channels
                # shape[1]: width, shape[0]: height
                test_grayscale = cv2.resize(test_grayscale, (reference_grayscale.shape[1], reference_grayscale.shape[0]))
            psnr_value_bicubic = psnr(reference_grayscale, test_grayscale)
            # Append name and data into arrays
            bicubic_name_array.append(input_image_files[i])
            bicubic_psnr_array.append(psnr_value_bicubic)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")

    # Lanczos
    for i in range(25):
        try:
            reference_image = cv2.imread(input_image_files[i])
            test_image = cv2.imread(lanczos_gans_image_files[i])
            if reference_image is None:
                raise FileNotFoundError(f"Error: Unable to read reference image at {input_image_files[i]}")
            if test_image is None:
                raise FileNotFoundError(f"Error: Unable to read test image at {lanczos_gans_image_files[i]}")
            print("OpenCV converts image to grayscale image to avoid working with RGB channels")
            # Summing the squared differences across channels might not align with human perception
            reference_grayscale = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            test_grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if reference_grayscale.shape != test_grayscale.shape:
                print("Test image must have same dimensions as reference image for comparison\nTest image will be resized to reference image by bilinear interpolation")
                # cv2.resize() uses bilinear interpolation for upsampling
                # resize: x, y (width, height)
                # shape: height, width, channels
                # shape[1]: width, shape[0]: height
                test_grayscale = cv2.resize(test_grayscale, (reference_grayscale.shape[1], reference_grayscale.shape[0]))
            psnr_value_lanczos = psnr(reference_grayscale, test_grayscale)
            # Append name and data into arrays
            lanczos_name_array.append(input_image_files[i])
            lanczos_psnr_array.append(psnr_value_lanczos)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")

    # Pixel area relation
    for i in range(25):
        try:
            reference_image = cv2.imread(input_image_files[i])
            test_image = cv2.imread(pixel_area_relation_gans_image_files[i])
            if reference_image is None:
                raise FileNotFoundError(f"Error: Unable to read reference image at {input_image_files[i]}")
            if test_image is None:
                raise FileNotFoundError(f"Error: Unable to read test image at {pixel_area_relation_gans_image_files[i]}")
            print("OpenCV converts image to grayscale image to avoid working with RGB channels")
            # Summing the squared differences across channels might not align with human perception
            reference_grayscale = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            test_grayscale = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            if reference_grayscale.shape != test_grayscale.shape:
                print("Test image must have same dimensions as reference image for comparison\nTest image will be resized to reference image by bilinear interpolation")
                # cv2.resize() uses bilinear interpolation for upsampling
                # resize: x, y (width, height)
                # shape: height, width, channels
                # shape[1]: width, shape[0]: height
                test_grayscale = cv2.resize(test_grayscale, (reference_grayscale.shape[1], reference_grayscale.shape[0]))
            psnr_value_pixel = psnr(reference_grayscale, test_grayscale)
            # Append name and data into arrays
            pixel_area_relation_name_array.append(input_image_files[i])
            pixel_area_relation_psnr_array.append(psnr_value_pixel)
        except FileNotFoundError as e:
            # Handle file not found exception
            print(f"File not found: {e}")
        except Exception as e:
            # Handle other exceptions
            print(f"An exception occurred: {e}")

    # Overall mean squared error in total 25 images in 5 methods
    
    # Nearest-neighbor
    tmp_nearest_neighbor = 0.0
    for i in range(25):
        tmp_nearest_neighbor = tmp_nearest_neighbor + nearest_neighbor_psnr_array[i]
    mean_nearest_neighbor = tmp_nearest_neighbor/25
    # Bilinear
    tmp_bilinear = 0.0
    for i in range(25):
        tmp_bilinear = tmp_bilinear + bilinear_psnr_array[i]
    mean_bilinear = tmp_bilinear/25
    # Bicubic
    tmp_bicubic = 0.0
    for i in range(25):
        tmp_bicubic = tmp_bicubic + bicubic_psnr_array[i]
    mean_bicubic = tmp_bicubic/25
    # Lanczos
    tmp_lanczos = 0.0
    for i in range(25):
        tmp_lanczos = tmp_lanczos + lanczos_psnr_array[i]
    mean_lanczos = tmp_lanczos/25
    # Pixel area relation
    tmp_pixel_area_relation = 0.0
    for i in range(25):
        tmp_pixel_area_relation = tmp_pixel_area_relation + pixel_area_relation_psnr_array[i]
    mean_pixel_area_relation = tmp_pixel_area_relation/25

    # plot table
    psnr_data = {
        'Processing metrics' : [
            # 25 images from Nearest-neighbor
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Nearest-neighbor + GANs (+ Bilinear)',
            'Average PSNR: Nearest-neighbor',
            # 25 images from Bilinear
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Bilinear + GANs (+ Bilinear)',
            'Average PSNR: Bilinear',
            # 25 images from Bicubic
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Bicubic + GANs (+ Bilinear)',
            'Average PSNR: Bicubic',
            # 25 images from Lanczos
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Lanczos + GANs (+ Bilinear)',
            'Average PSNR: Lanczos',
            # 25 images from Pixel area relation
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Pixel area relation + GANs (+ Bilinear)',
            'Average PSNR: Pixel area relation'
        ],
        'Index' : [
            # Nearest-neighbor
            '001',
            '002',
            '003',
            '004',
            '005',
            '006',
            '007',
            '008',
            '009',
            '010',
            '011',
            '012',
            '013',
            '014',
            '015',
            '016',
            '017',
            '018',
            '019',
            '020',
            '021',
            '022',
            '023',
            '024',
            '025',
            'N/A',
            # Bilinear
            '001',
            '002',
            '003',
            '004',
            '005',
            '006',
            '007',
            '008',
            '009',
            '010',
            '011',
            '012',
            '013',
            '014',
            '015',
            '016',
            '017',
            '018',
            '019',
            '020',
            '021',
            '022',
            '023',
            '024',
            '025',
            'N/A',
            # Bicubic
            '001',
            '002',
            '003',
            '004',
            '005',
            '006',
            '007',
            '008',
            '009',
            '010',
            '011',
            '012',
            '013',
            '014',
            '015',
            '016',
            '017',
            '018',
            '019',
            '020',
            '021',
            '022',
            '023',
            '024',
            '025',
            'N/A',
            # Lanczos
            '001',
            '002',
            '003',
            '004',
            '005',
            '006',
            '007',
            '008',
            '009',
            '010',
            '011',
            '012',
            '013',
            '014',
            '015',
            '016',
            '017',
            '018',
            '019',
            '020',
            '021',
            '022',
            '023',
            '024',
            '025',
            'N/A',
            # Pixel area relation
            '001',
            '002',
            '003',
            '004',
            '005',
            '006',
            '007',
            '008',
            '009',
            '010',
            '011',
            '012',
            '013',
            '014',
            '015',
            '016',
            '017',
            '018',
            '019',
            '020',
            '021',
            '022',
            '023',
            '024',
            '025',
            'N/A'
        ],
        'Peak-Signal-to-Noise-Ratio' : [
            # Nearest-neighbor
            nearest_neighbor_psnr_array[0],
            nearest_neighbor_psnr_array[1],
            nearest_neighbor_psnr_array[2],
            nearest_neighbor_psnr_array[3],
            nearest_neighbor_psnr_array[4],
            nearest_neighbor_psnr_array[5],
            nearest_neighbor_psnr_array[6],
            nearest_neighbor_psnr_array[7],
            nearest_neighbor_psnr_array[8],
            nearest_neighbor_psnr_array[9],
            nearest_neighbor_psnr_array[10],
            nearest_neighbor_psnr_array[11],
            nearest_neighbor_psnr_array[12],
            nearest_neighbor_psnr_array[13],
            nearest_neighbor_psnr_array[14],
            nearest_neighbor_psnr_array[15],
            nearest_neighbor_psnr_array[16],
            nearest_neighbor_psnr_array[17],
            nearest_neighbor_psnr_array[18],
            nearest_neighbor_psnr_array[19],
            nearest_neighbor_psnr_array[20],
            nearest_neighbor_psnr_array[21],
            nearest_neighbor_psnr_array[22],
            nearest_neighbor_psnr_array[23],
            nearest_neighbor_psnr_array[24],
            mean_nearest_neighbor,
            # Bilinear
            bilinear_psnr_array[0],
            bilinear_psnr_array[1],
            bilinear_psnr_array[2],
            bilinear_psnr_array[3],
            bilinear_psnr_array[4],
            bilinear_psnr_array[5],
            bilinear_psnr_array[6],
            bilinear_psnr_array[7],
            bilinear_psnr_array[8],
            bilinear_psnr_array[9],
            bilinear_psnr_array[10],
            bilinear_psnr_array[11],
            bilinear_psnr_array[12],
            bilinear_psnr_array[13],
            bilinear_psnr_array[14],
            bilinear_psnr_array[15],
            bilinear_psnr_array[16],
            bilinear_psnr_array[17],
            bilinear_psnr_array[18],
            bilinear_psnr_array[19],
            bilinear_psnr_array[20],
            bilinear_psnr_array[21],
            bilinear_psnr_array[22],
            bilinear_psnr_array[23],
            bilinear_psnr_array[24],
            mean_bilinear,
            # Bicubic
            bicubic_psnr_array[0],
            bicubic_psnr_array[1],
            bicubic_psnr_array[2],
            bicubic_psnr_array[3],
            bicubic_psnr_array[4],
            bicubic_psnr_array[5],
            bicubic_psnr_array[6],
            bicubic_psnr_array[7],
            bicubic_psnr_array[8],
            bicubic_psnr_array[9],
            bicubic_psnr_array[10],
            bicubic_psnr_array[11],
            bicubic_psnr_array[12],
            bicubic_psnr_array[13],
            bicubic_psnr_array[14],
            bicubic_psnr_array[15],
            bicubic_psnr_array[16],
            bicubic_psnr_array[17],
            bicubic_psnr_array[18],
            bicubic_psnr_array[19],
            bicubic_psnr_array[20],
            bicubic_psnr_array[21],
            bicubic_psnr_array[22],
            bicubic_psnr_array[23],
            bicubic_psnr_array[24],
            mean_bicubic,
            # Lanczos
            lanczos_psnr_array[0],
            lanczos_psnr_array[1],
            lanczos_psnr_array[2],
            lanczos_psnr_array[3],
            lanczos_psnr_array[4],
            lanczos_psnr_array[5],
            lanczos_psnr_array[6],
            lanczos_psnr_array[7],
            lanczos_psnr_array[8],
            lanczos_psnr_array[9],
            lanczos_psnr_array[10],
            lanczos_psnr_array[11],
            lanczos_psnr_array[12],
            lanczos_psnr_array[13],
            lanczos_psnr_array[14],
            lanczos_psnr_array[15],
            lanczos_psnr_array[16],
            lanczos_psnr_array[17],
            lanczos_psnr_array[18],
            lanczos_psnr_array[19],
            lanczos_psnr_array[20],
            lanczos_psnr_array[21],
            lanczos_psnr_array[22],
            lanczos_psnr_array[23],
            lanczos_psnr_array[24],
            mean_lanczos,
            # Pixel area relation
            pixel_area_relation_psnr_array[0],
            pixel_area_relation_psnr_array[1],
            pixel_area_relation_psnr_array[2],
            pixel_area_relation_psnr_array[3],
            pixel_area_relation_psnr_array[4],
            pixel_area_relation_psnr_array[5],
            pixel_area_relation_psnr_array[6],
            pixel_area_relation_psnr_array[7],
            pixel_area_relation_psnr_array[8],
            pixel_area_relation_psnr_array[9],
            pixel_area_relation_psnr_array[10],
            pixel_area_relation_psnr_array[11],
            pixel_area_relation_psnr_array[12],
            pixel_area_relation_psnr_array[13],
            pixel_area_relation_psnr_array[14],
            pixel_area_relation_psnr_array[15],
            pixel_area_relation_psnr_array[16],
            pixel_area_relation_psnr_array[17],
            pixel_area_relation_psnr_array[18],
            pixel_area_relation_psnr_array[19],
            pixel_area_relation_psnr_array[20],
            pixel_area_relation_psnr_array[21],
            pixel_area_relation_psnr_array[22],
            pixel_area_relation_psnr_array[23],
            pixel_area_relation_psnr_array[24],
            mean_pixel_area_relation
        ]
    }

    plot_table(data=psnr_data)

    plot_bar_chart(mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation)

if __name__ == "__main__":
    main()
