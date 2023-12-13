import cv2
from skimage.metrics import structural_similarity as ssim

# def resize_images(original, distorted):
#     # Resize distorted image to match the size of the original image
#     distorted_resized = cv2.resize(distorted, (original.shape[1], original.shape[0]))
#     return distorted_resized

if __name__ == "__main__":
    # Load reference and comparison images
    ref_image = cv2.imread('baboon.png')
    comp_image = cv2.imread('output.png')

    # Resize
    # comp_image = resize_images(ref_image, comp_image)
    comp_image = cv2.resize(comp_image, (ref_image.shape[1], ref_image.shape[0]))

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    comp_gray = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_score, _ = ssim(ref_gray, comp_gray, full=True)

    print(f"SSIM Score: {ssim_score}")
