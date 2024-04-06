# import numpy as np
import cv2

img = cv2.imread('image.png')
print('Input sample:')
print(img)
print('Pixel area relation (Shrunk):')
area_resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imwrite('area_shrunk.png', area_resized)
area_shrunk = cv2.imread('area_shrunk.png')
print(area_shrunk)
print('Pixel area relation (Enlarged):')
area_enlarged = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)), interpolation=cv2.INTER_AREA)
cv2.imwrite('area_enlarged.png', area_enlarged)
area_expanded = cv2.imread('area_enlarged.png')
print(area_expanded)
print('Bilinear (Shrunk):')
linear_resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('linear_shrunk.png', linear_resized)
linear_shrunk = cv2.imread('linear_shrunk.png')
print(linear_shrunk)
print('Bilinear (Enlarged):')
linear_enlarged = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('linear_enlarged.png', linear_enlarged)
linear_expanded = cv2.imread('linear_enlarged.png')
print(linear_expanded)

# Create a numpy array, pretending it is our image
# img = np.array([[3, 106, 107, 40, 148, 112, 254, 151],
#                 [62, 173, 91, 93, 33, 111, 139, 25],
#                 [99, 137, 80, 231, 101, 204, 74, 219],
#                 [240, 173, 85, 14, 40, 230, 160, 152],
#                 [230, 200, 177, 149, 173, 239, 103, 74],
#                 [19, 50, 209, 82, 241, 103, 3, 87],
#                 [252, 191, 55, 154, 171, 107, 6, 123],
#                 [7, 101, 168, 85, 115, 103, 32, 11]],
#                 dtype=np.uint8)
# # Resize the width and height in half
# resized = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2),
#                      interpolation=cv2.INTER_AREA)
# print(resized)
# Result:
# [[ 86  83 101 142]
# [162 103 144 151]
# [125 154 189  67]
# [138 116 124  43]]

# img = np.array([[ 86 83 101 142]
#                 [162 103 144 151]
#                 [125 154 189  67]
#                 [138 116 124  43]], dtype=np.uint8)
# enlarged = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
# print(enlarged)
# Result:
#[[ 86  86  83  83 101 101 142 142]
# [ 86  86  83  83 101 101 142 142]
# [162 162 103 103 144 144 151 151]
# [162 162 103 103 144 144 151 151]
# [125 125 154 154 189 189  67  67]
# [125 125 154 154 189 189  67  67]
# [138 138 116 116 124 124  43  43]
# [138 138 116 116 124 124  43  43]]
