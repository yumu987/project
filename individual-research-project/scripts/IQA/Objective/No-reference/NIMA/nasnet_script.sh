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

echo "nasnet is running"

evaluate() {
    local script_name="$1"
    local log_file="$2"
    python "$script_name" -dir all > "$log_file"
}

evaluate "evaluate_nasnet.py" "nasnet_all.txt"

echo "nasnet is finished"
