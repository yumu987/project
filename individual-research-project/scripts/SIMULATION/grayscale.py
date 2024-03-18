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

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

def grayscale_transform(input_path, output_path):
    try:
        # Read input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Convert input image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Save the grayscale image
        cv2.imwrite(output_path, gray_img)
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

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

    plt.suptitle('TABLE\nThe processing time of grayscale transformation', fontsize=14, weight='bold')

    pdf_file = 'grayscale.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    png_file = 'grayscale.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

    print("Grayscale transformation and plotting table completed")

def main():
    input_files = ['input/{}.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    grayscale_files = ['input_grayscale/{}_grayscale.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    grayscale_start_time = time.time()
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        grayscale_batch = grayscale_files[i:i+1]
        for input, grayscale in zip(input_batch, grayscale_batch):
            grayscale_transform(input, grayscale)
    grayscale_end_time = time.time()
    grayscale_elapsed_time = grayscale_end_time - grayscale_start_time
    grayscale_average_time = grayscale_elapsed_time/25

    data = {
        'Grayscale transformation':['Grayscale'],
        'Processing time':[grayscale_elapsed_time],
        'Average processing time':[grayscale_average_time]
    }

    plot_table(data)

if __name__ == "__main__":
    main()
