################################################
# This script will no longer be used in later experiments
# It is archived here
################################################

import cv2
from ISR.models import RDN
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import numpy as np
import os
import tensorflow as tf
import gc # Import the garbage collector module

import atexit

def cleanup():
    cv2.destroyAllWindows()
    tf.keras.backend.clear_session()

def super_resolve_image(model, input_path, output_path):
    try:
        img = cv2.imread(input_path)

        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        
        sr_img = model.predict(img)
        cv2.imwrite(output_path, sr_img)
        
        # Release resources
        del img
        del sr_img

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An exception occurred: {e}")

def process_images(model, input_files, output_files, batch_size):
    start_time = time.time()

    for i in range(0, len(input_files), batch_size):
        input_batch = input_files[i:i+batch_size]
        output_batch = output_files[i:i+batch_size]

        for input_file, output_file in zip(input_batch, output_batch):
            input_path = input_file
            output_path = output_file

            if os.path.exists(output_path):
                print(f"Output image {output_path} already exists. Skipping.")
                continue

            print(f"Attempting to read input image: {input_path}")
            print(f"Attempting to write output image: {output_path}")

            super_resolve_image(model, input_path, output_path)

        # Release resources after processing each batch
        cv2.destroyAllWindows()
        tf.keras.backend.clear_session()
        gc.collect()  # Trigger garbage collection

    end_time = time.time()
    elapsed_time = end_time - start_time
    average_time = elapsed_time / len(input_files)

    return elapsed_time, average_time

def main():
    print("RDN psnr-large part is starting")

    # Load RDN model
    model = RDN(weights='psnr-large')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    for j in range(5):
        # Input and output image files
        input_files = ['nearest_neighbor_half/{}_nearest_neighbor_half.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
        output_files = ['optimise_nearest_neighbor_psnr_large/{}_nearest_neighbor_psnr_large.jpg'.format(str(i).zfill(3)) for i in range(1, 26)]
        # Process images
        elapsed_time, average_time = process_images(model, input_files, output_files, batch_size=1)

    # Display processing time
    print(f"RDN: psnr-large | total processing time: {elapsed_time} seconds")
    print(f"RDN: psnr-large | average processing time: {average_time} seconds")

    # Plotting table
    data = {
        'Processing metrics': ['Nearest-neighbor + psnr-large'],
        'Scale': ['2'],
        'Processing time': [elapsed_time],
        'Average processing time': [average_time]
    }

    df = pd.DataFrame(data)
    df.fillna('-', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    tab = table(ax, df, loc='center', cellLoc='center', colLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(len(df.columns))))

    for (i, j), val in np.ndenumerate(df.values):
        if j == 0:  # First column
            tab[(i+1, j)].set_facecolor('#dddddd')  # Shade the first column
        if j == -1 or j == 0:  # Header or first column
            tab[(i+1, j)].set_text_props(weight='bold')  # Bold the font of the first column

    plt.suptitle('TABLE\nThe processing time of optimise model', fontsize=14, weight='bold')

    pdf_file = 'optimise.pdf'
    plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)
    png_file = 'optimise.png'
    plt.savefig(png_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

    print("Super-resolution and plotting table completed")

if __name__ == "__main__":
    atexit.register(cleanup)
    main()
