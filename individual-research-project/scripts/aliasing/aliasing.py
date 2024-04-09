import cv2
from ISR.models import RRDN, RDN

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
    ####################
    # RDN (scaling factor is 2)
    ####################
    # psnr-large
    RDN_super_resolution('image.png', 'psnr-large.png', 'psnr-large')
    # psnr-small
    RDN_super_resolution('image.png', 'psnr-small.png', 'psnr-small')
    # noise-cancel
    RDN_super_resolution('image.png', 'noise-cancel.png', 'noise-cancel')
    ####################
    # RRDN (scaling factor is 4)
    ####################
    # gans
    RRDN_super_resolution('image.png', 'gans.png', 'gans')
    ####################

if __name__ == "__main__":
    main()
