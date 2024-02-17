################################################
# Structural Similarity (SSIM)
# https://arxiv.org/abs/1406.7799
# https://www.cns.nyu.edu/~lcv/ssim/
# https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
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
# Skimage, OpenCV, Matplotlib, Pandas and Numpy packages are required in this script
# Instruction of installing packages:
# pip install scikit-image
# pip install opencv-python
# pip install matplotlib
# pip install pandas
# pip install numpy
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
################################################

################################################
# Structural Similarity Index
import cv2
from skimage.metrics import structural_similarity as ssim
################################################

################################################
# Table
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
################################################

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
    plt.suptitle('TABLE\nStructural Similarity Index', fontsize=14, weight='bold')
    # Save the figure as a PDF
    pdf_file = 'ssim.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    # Save the figure as an image (PNG)
    png_file = 'ssim.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    # Close the matplotlib figure
    plt.close(fig)
    # Indication of plotting table completed
    print("Plotting table completed")

def main():
    # Structural Similarity Index
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
        # Calculate Structural similarity index
        ssim_value, _ = ssim(reference_grayscale, test_grayscale, full=True)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

    # plot table
    ssim_data = {ssim_value}
    plot_table(data=)

if __name__ == "__main__":
    main()
