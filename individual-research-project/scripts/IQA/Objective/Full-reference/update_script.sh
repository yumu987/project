#!/bin/bash

####################
# To give this shell script permission to execute:
# chmod +x script.sh
####################
# To run this shell script:
# ./script.sh
####################

echo "Script is running!"

# Run python script
python mse.py
python psnr.py
python ssim.py
python compression.py

echo "Script is completed!"

# Exit the script
exit
