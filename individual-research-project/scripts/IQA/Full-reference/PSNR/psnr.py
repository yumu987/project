# the pixel values are being treated as 8-bit unsigned integers (uint8),
# but the PSNR calculation assumes a floating-point scale. 
# Let's address this by normalizing the pixel values to the [0, 1] 
# range before performing the PSNR calculation.
import cv2
import numpy as np

def calculate_psnr(original, distorted):
    # Ensure images have the same shape and type
    if original.shape != distorted.shape or original.dtype != distorted.dtype:
        raise ValueError("Input images must have the same shape and type")

    # Convert images to float32 and normalize to [0, 1] for accurate calculations
    # original = original.astype(np.float32) / 255.0
    # distorted = distorted.astype(np.float32) / 255.0
    original = original.astype(np.float32)
    distorted = distorted.astype(np.float32)

    # Calculate mean squared error (MSE)
    mse = np.mean((original - distorted) ** 2)

    # Calculate PSNR
    # max could be 255
    # max_pixel_value = 1.0  # Maximum pixel value is 1 after normalization
    max_pixel_value = 255.0 # Maximon pixel value
    # psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    psnr = (20 * np.log10(max_pixel_value)) - (10 * np.log10(mse))

    return psnr

if __name__ == "__main__":
    # Load original and distorted images
    original_image = cv2.imread("baboon.png")
    distorted_image = cv2.imread("output.png")

    # Check image properties
    print(f"Original Image Shape: {original_image.shape}, Data Type: {original_image.dtype}")
    print(f"Distorted Image Shape: {distorted_image.shape}, Data Type: {distorted_image.dtype}")

    # Resize distorted image to match the size of the original image
    distorted_image = cv2.resize(distorted_image, (original_image.shape[1], original_image.shape[0]))

    # Check resized image properties
    print(f"Resized Distorted Image Shape: {distorted_image.shape}, Data Type: {distorted_image.dtype}")

    # Calculate PSNR
    psnr_value = calculate_psnr(original_image, distorted_image)

    print(f"PSNR: {psnr_value:.2f} dB")
