#!/bin/bash

####################
# To give this shell script permission to execute:
# chmod +x script.sh
####################
# To run this shell script:
# ./script.sh
####################

####################
# Commands to run python scripts:
# -dir    : Pass the relative/full path of a directory containing a set of images. Only png, jpg and jpeg images will be scored.
# -img    : Pass one or more relative/full paths of images to score them. Can support all image types supported by PIL.
# -resize : Pass "true" or "false" as values. Resize an image prior to scoring it. Not supported on NASNet models.
####################

echo "Script is running!"

####################
# Functions of executing scripts
# Input
evaluate_input() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir input > "$log_file"
}
# Nearest-neighbor
evaluate_nearest_neighbor() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir nearest_neighbor_gans > "$log_file"
}
# Bilinear
evaluate_bilinear() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir bilinear_gans > "$log_file"
}
# Bicubic
evaluate_bicubic() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir bicubic_gans > "$log_file"
}
# Lanczos
evaluate_lanczos() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir lanczos_gans > "$log_file"
}
# Pixel area relation
evaluate_pixel_area_relation() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir pixel_area_relation_gans > "$log_file"
}
# Commands of executing scripts
# Input
evaluate_input "evaluate_inception_resnet.py" "inception_resnet_input.txt"
evaluate_input "evaluate_mobilenet.py" "mobilenet_input.txt"
evaluate_input "evaluate_nasnet.py" "nasnet_input.txt"
# Nearest-neighbor
evaluate_nearest_neighbor "evaluate_inception_resnet.py" "inception_resnet_nearest_neighbor.txt"
evaluate_nearest_neighbor "evaluate_mobilenet.py" "mobilenet_nearest_neighbor.txt"
evaluate_nearest_neighbor "evaluate_nasnet.py" "nasnet_nearest_neighbor.txt"
# Bilinear
evaluate_bilinear "evaluate_inception_resnet.py" "inception_resnet_bilinear.txt"
evaluate_bilinear "evaluate_mobilenet.py" "mobilenet_bilinear.txt"
evaluate_bilinear "evaluate_nasnet.py" "nasnet_bilinear.txt"
# Bicubic
evaluate_bicubic "evaluate_inception_resnet.py" "inception_resnet_bicubic.txt"
evaluate_bicubic "evaluate_mobilenet.py" "mobilenet_bicubic.txt"
evaluate_bicubic "evaluate_nasnet.py" "nasnet_bicubic.txt"
# Lanczos
evaluate_lanczos "evaluate_inception_resnet.py" "inception_resnet_lanczos.txt"
evaluate_lanczos "evaluate_mobilenet.py" "mobilenet_lanczos.txt"
evaluate_lanczos "evaluate_nasnet.py" "nasnet_lanczos.txt"
# Pixel area relation
evaluate_pixel_area_relation "evaluate_inception_resnet.py" "inception_resnet_pixel_area_relation.txt"
evaluate_pixel_area_relation "evaluate_mobilenet.py" "mobilenet_pixel_area_relation.txt"
evaluate_pixel_area_relation "evaluate_nasnet.py" "nasnet_pixel_area_relation.txt"
####################

echo "Script is completed!"

# Exit the script
exit
