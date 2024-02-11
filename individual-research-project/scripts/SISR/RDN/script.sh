#!/bin/bash

####################
# To give this shell script permission to execute:
# chmod +x script.sh
####################
# To run this shell script:
# ./script.sh
####################

# Run python scripts
python psnr-large.py
python psnr-small.py
python noise-cancel.py

# Exit the script
exit
