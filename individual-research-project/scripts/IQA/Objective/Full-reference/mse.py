################################################
# Mean Squared Error (MSE)
# https://arxiv.org/abs/1406.7799
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
# Mean-squared-error (MSE) is measured over the entire image by giving 
# equal weight to each corresponding pixel difference of the reference and distorted image.
#
# Low MSE -> Low numerical differences
# High MSE -> High numerical differences
#
# LaTeX formula:
# Mean-squared-error (MSE)
# \begin{equation}
# MSE = \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} (I_r(i,j) - I_d(i,j))^2
# \end{equation}
#
# Proposed Idea:
# [Reference Image]
#           \
#            \
#             ---> [Blockwise MSE] -> [Maximum MSE]
#            /
#           /
# [Distorted Image]
#
# The purpose of this script is to execute MSE IQA metric
# [MSE]
################################################

################################################
# Mean Squared Error
import cv2
import numpy as np
################################################

################################################
# Table
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
################################################

def mse(reference_image, test_image):
    """
    MSE denotes the power of the distortion, 
    i.e., the difference between the reference 
    and test image
    """
    # Convert images to float32 for accurate calculations
    reference_value = reference_image.astype(np.float32)
    test_value = test_image.astype(np.float32)
    # Calculate mean squared error by float32
    mean_squared_error = np.mean((reference_value - test_value) ** 2)
    return mean_squared_error

def plot_table(data):
    # Data to be used to plot table
    # data = {}
    data = data

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
    plt.suptitle('TABLE\nMean Squared Error', fontsize=14, weight='bold')
    # Save the figure as a PDF
    pdf_file = 'mse.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    # Save the figure as an image (PNG)
    png_file = 'mse.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    # Close the matplotlib figure
    plt.close(fig)
    # Indication of plotting table completed
    print("Plotting table completed")

def main():
    # Mean Squared Error
    # for loop
    try:
        reference_image = cv2.imread(...)
        test_image = cv2.imread(...)
        if reference_image is None:
            raise FileNotFoundError(f"Error: Unable to read reference image at {...}")
        if test_image is None:
            raise FileNotFoundError(f"Error: Unable to read test image at {...}")
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
        # Calculate Mean Squared Error
        mse_value = mse(reference_grayscale, test_grayscale)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

    # plot table
    mse_data = {mse_value}
    plot_table(data=)

if __name__ == "__main__":
    main()
