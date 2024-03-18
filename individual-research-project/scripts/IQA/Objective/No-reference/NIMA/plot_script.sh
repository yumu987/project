#!/bin/bash

####################
# To give this shell script permission to execute:
# chmod +x script.sh
####################
# To run this shell script:
# ./script.sh
####################

# Execute python scripts
python nasnet_input_plot.py
python nasnet_downsampling_plot.py
python nasnet_average_plot.py

# Exit the script
exit
