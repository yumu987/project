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

from PIL import Image
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

def opencv_jpeg_lossy_compression(input_path, output_path, quality):
    """
    OpenCV implementation of compressing an image by using JPEG lossy compression.
    input_path: path of input image files
    output_path: path of output image files
    quality: JPEG compression quality from 0 to 100
    Higher quality number means higher quality with larger size
    Lower quality number means lower quality with smaller size
    """
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Compress the image by using JPEG compression
        compressed_image = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()
        # Save the compressed image
        with open(output_path, 'wb') as output_file:
            output_file.write(compressed_image)
        # Indication of image compression finished and saved
        print(f"Image is compressed and saved to {output_path}")
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def jpeg_lossy_compression(input_path, output_path, quality):
    """
    Implementation of compressing an image by using JPEG lossy compression.
    input_path: path of input image files
    output_path: path of output image files
    quality: JPEG compression quality from 0 to 100
    Higher quality number means higher quality with larger size
    Lower quality number means lower quality with smaller size
    """
    try:
        # Read the input image
        img = Image.open(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Ensure output directory exists
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the compressed image
        img.save(output_path, quality=quality)
        # Indication of image compression finished and saved
        print(f"Image is compressed and saved to {output_path}")
    except FileNotFoundError as e:
        # Handle file not found exception
        print(f"File not found: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"An exception occurred: {e}")

def general_jpeg_lossy_compression(input_path, encode_path, decode_path):
    """
    General implementation of compressing an image by using JPEG lossy compression.
    input_path: path of input image files
    output_path: path of output image files
    """
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Encode the image by using JPEG compression
        success, encoded_image = cv2.imencode('.jpg', img)
        if success:
            # Save the encoded image
            with open(encode_path, 'wb') as output_file:
                output_file.write(encoded_image)
            # Indication of image encode finished and saved
            print(f"Image is encoded and saved to {encode_path}")
            # Decode the encoded image
            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            # Save the decoded image
            cv2.imwrite(decode_path, decoded_image)
            # Indication of image decode finished and saved
            print(f"Image is decoded and saved to {decode_path}")
        else:
            # Indication of image encode has been failed
            print("Failed to encode the image")
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

    plt.suptitle('TABLE\nThe processing time of JPEG image compression', fontsize=14, weight='bold')

    pdf_file = 'JPEG.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    png_file = 'JPEG.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

    print("JPEG image compression and plotting table completed")

def main():
    # General encode and decode by JPEG image compression
    # zfill() will fill 0 from left to right
    input_files = ['input/{}.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    encode_files = ['input_encoded/{}_encoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    decode_files = ['input_decoded/{}_decoded.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    general_start_time = time.time()
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        encode_batch = encode_files[i:i+1]
        decode_batch = decode_files[i:i+1]
        for input, encode, decode in zip(input_batch, encode_batch, decode_batch):
            general_jpeg_lossy_compression(input, encode, decode)
    general_end_time = time.time()
    general_elapsed_time = general_end_time - general_start_time
    general_average_time = general_elapsed_time/25
    # JPEG image compression: quality 1
    one_start_time = time.time()
    one_files = ['input_compressed_1/{}_compressed_1.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        one_batch = one_files[i:i+1]
        for input, one in zip(input_batch, one_batch):
            opencv_jpeg_lossy_compression(input, one, 1)
    one_end_time = time.time()
    one_elapsed_time = one_end_time - one_start_time
    one_average_time = one_elapsed_time/25
    # JPEG image compression: quality 25
    twofive_start_time = time.time()
    twofive_files = ['input_compressed_25/{}_compressed_25.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        twofive_batch = twofive_files[i:i+1]
        for input, twofive in zip(input_batch, twofive_batch):
            opencv_jpeg_lossy_compression(input, twofive, 25)
    twofive_end_time = time.time()
    twofive_elapsed_time = twofive_end_time - twofive_start_time
    twofive_average_time = twofive_elapsed_time/25
    # JPEG image compression: quality 50
    fivezero_start_time = time.time()
    fivezero_files = ['input_compressed_50/{}_compressed_50.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        fivezero_batch = fivezero_files[i:i+1]
        for input, fivezero in zip(input_batch, fivezero_batch):
            opencv_jpeg_lossy_compression(input, fivezero, 50)
    fivezero_end_time = time.time()
    fivezero_elapsed_time = fivezero_end_time - fivezero_start_time
    fivezero_average_time = fivezero_elapsed_time/25
    # JPEG image compression: quality 75
    sevenfive_start_time = time.time()
    sevenfive_files = ['input_compressed_75/{}_compressed_75.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        sevenfive_batch = sevenfive_files[i:i+1]
        for input, sevenfive in zip(input_batch, sevenfive_batch):
            opencv_jpeg_lossy_compression(input, sevenfive, 75)
    sevenfive_end_time = time.time()
    sevenfive_elapsed_time = sevenfive_end_time - sevenfive_start_time
    sevenfive_average_time = sevenfive_elapsed_time/25
    # JPEG image compression: quality 100
    onezerozero_start_time = time.time()
    onezerozero_files = ['input_compressed_100/{}_compressed_100.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
    for i in range(0, len(input_files), 1):
        input_batch = input_files[i:i+1]
        onezerozero_batch = onezerozero_files[i:i+1]
        for input, onezerozero in zip(input_batch, onezerozero_batch):
            opencv_jpeg_lossy_compression(input, onezerozero, 100)
    onezerozero_end_time = time.time()
    onezerozero_elapsed_time = onezerozero_end_time - onezerozero_start_time
    onezerozero_average_time = onezerozero_elapsed_time/25

    data = {
        'JPEG image compression quality':['general', 
                                          '1', 
                                          '25', 
                                          '50', 
                                          '75', 
                                          '100'],
        'Processing time':[general_elapsed_time, 
                           one_elapsed_time, 
                           twofive_elapsed_time, 
                           fivezero_elapsed_time, 
                           sevenfive_elapsed_time, 
                           onezerozero_elapsed_time],
        'Average processing time':[general_average_time,
                                   one_average_time,
                                   twofive_average_time,
                                   fivezero_average_time,
                                   sevenfive_average_time,
                                   onezerozero_average_time]
    }

    plot_table(data)

if __name__ == "__main__":
    main()
