import cv2

img = cv2.imread('image.png')
area_resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_AREA)
cv2.imwrite('area_shrunk.png', area_resized)
area_shrunk = cv2.imread('area_shrunk.png')
print(area_shrunk)
linear_resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('linear_shrunk.png', linear_resized)
linear_shrunk = cv2.imread('linear_shrunk.png')
print(linear_shrunk)
area_enlarged = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)), interpolation=cv2.INTER_AREA)
cv2.imwrite('area_enlarged.png', area_enlarged)
area_expanded = cv2.imread('area_enlarged.png')
print(area_expanded)
linear_enlarged = cv2.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('linear_enlarged.png', linear_enlarged)
linear_expanded = cv2.imread('linear_enlarged.png')
print(linear_expanded)

# Input sample =
# [[[ 59  92 131]
#   [ 55  87 123]
#   [ 53  81 115]
#   ...
#   [ 95 121 105]
#   [ 93 115  97]
#   [ 88 110  92]]

#  [[ 60  91 130]
#   [ 55  87 122]
#   [ 57  84 118]
#   ...

# Pixel area relation (Shrunk) =
# [[[ 57  89 127]
#   [ 53  80 111]
#   [ 37  57  75]
#   ...
#   [ 89 115  99]
#   [ 91 117 101]
#   [ 88 111  93]]

#  [[ 66  95 130]
#   [ 57  82 110]
#   [ 41  63  83]
#   ...

# Bilinear (Shrunk) =
# [[[ 57  89 127]
#   [ 53  80 111]
#   [ 37  57  75]
#   ...
#   [ 89 115  99]
#   [ 91 117 101]
#   [ 88 111  93]]

#  [[ 66  95 130]
#   [ 57  82 110]
#   [ 41  63  83]
#   ...

# Pixel area relation (Enlarged) =
# [[[ 59  92 131]
#   [ 59  92 131]
#   [ 55  87 123]
#   ...
#   [ 93 115  97]
#   [ 88 110  92]
#   [ 88 110  92]]

#  [[ 59  92 131]
#   [ 59  92 131]
#   [ 55  87 123]
#   ...

# Bilinear (Enlarged) =
# [[[ 59  92 131]
#   [ 58  91 129]
#   [ 56  88 125]
#   ...
#   [ 92 114  96]
#   [ 89 111  93]
#   [ 88 110  92]]

#  [[ 59  92 131]
#   [ 58  91 129]
#   [ 56  88 125]
#   ...
