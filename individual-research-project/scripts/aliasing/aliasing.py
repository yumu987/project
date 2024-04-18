import cv2
from ISR.models import RRDN, RDN
import time

def RDN_super_resolution(input_path, output_path, model_path):
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
        # Initialise the RRDN model
        model = RDN(weights=model_path)
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

def RRDN_super_resolution(input_path, output_path, model_path):
    try:
        # Read the input image
        img = cv2.imread(input_path)
        # Handle file not found error
        if img is None:
            raise FileNotFoundError(f"Error: Unable to read input image at {input_path}")
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

def main():
    # Dr Dimitris Agrafiotis
    ####################
    # RDN (scaling factor is 2)
    ####################
    # psnr-large
    a_time = time.time()
    RDN_super_resolution('image.png', 'psnr-large.png', 'psnr-large')
    b_time = time.time()
    # psnr-small
    c_time = time.time()
    RDN_super_resolution('image.png', 'psnr-small.png', 'psnr-small')
    d_time = time.time()
    # noise-cancel
    e_time = time.time()
    RDN_super_resolution('image.png', 'noise-cancel.png', 'noise-cancel')
    f_time = time.time()
    ####################
    # RRDN (scaling factor is 4)
    ####################
    # gans
    g_time = time.time()
    RRDN_super_resolution('image.png', 'gans.png', 'gans')
    h_time = time.time()
    ####################

    # Dr Fadi Karameh
    ####################
    # RDN (scaling factor is 2)
    ####################
    # psnr-large
    aa_time = time.time()
    RDN_super_resolution('portrait.png', 'portrait_psnr-large.png', 'psnr-large')
    bb_time = time.time()
    # psnr-small
    cc_time = time.time()
    RDN_super_resolution('portrait.png', 'portrait_psnr-small.png', 'psnr-small')
    dd_time = time.time()
    # noise-cancel
    ee_time = time.time()
    RDN_super_resolution('portrait.png', 'portrait_noise-cancel.png', 'noise-cancel')
    ff_time = time.time()
    ####################
    # RRDN (scaling factor is 4)
    ####################
    # gans
    gg_time = time.time()
    RRDN_super_resolution('portrait.png', 'portrait_gans.png', 'gans')
    hh_time = time.time()
    ####################

    # Time
    print("##################################################")
    psnr_large_dimitris = b_time - a_time
    psnr_small_dimitris = d_time - c_time
    noise_cancel_dimitris = f_time - e_time
    gans_dimitris = h_time - g_time
    print("Dr Dimitris Agrafiotis:")
    print(f"psnr-large: {psnr_large_dimitris} seconds")
    print(f"psnr-small: {psnr_small_dimitris} seconds")
    print(f"noise-cancel: {noise_cancel_dimitris} seconds")
    print(f"gans: {gans_dimitris} seconds")

    psnr_large_fadi = bb_time - aa_time
    psnr_small_fadi = dd_time - cc_time
    noise_cancel_fadi = ff_time - ee_time
    gans_fadi = hh_time - gg_time
    print("Dr Fadi Karameh:")
    print(f"psnr-large: {psnr_large_fadi} seconds")
    print(f"psnr-small: {psnr_small_fadi} seconds")
    print(f"noise-cancel: {noise_cancel_fadi} seconds")
    print(f"gans: {gans_fadi} seconds")
    print("##################################################")

if __name__ == "__main__":
    main()
