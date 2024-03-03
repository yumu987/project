################################################
# Schematic diagram of image compression
# This diagram is inspired by "Comparison of Commonly Used Image Interpolation Methods" paper (4822.pdf)
# https://www.semanticscholar.org/paper/Comparison-of-Commonly-Used-Image-Interpolation-Han/47408d4854044f900d6ee757ca322de983d702ce
################################################

################################################
#
# EENG30009 Individual Research Project 3
# TOPIC: Single-image Super-resolution
#
# Yumu Xie
# University of Bristol
#
################################################

################################################
# Programming environment: WSL Linux subsystem of Windows
# Python version: 3.7.9
# Python setup is based on 'pyenv' (Simple Python version management)
# https://github.com/pyenv/pyenv
################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # Set up the plotting area
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    # Loop through each subplot to configure
    for ax in axs:
        # Draw a 4x4 grid
        ax.set_xticks(range(0, 5))
        ax.set_yticks(range(0, 5))
        # Turn grid on
        ax.grid(True)
        # Set the aspect of the plot to be equal
        ax.set_aspect('equal')
        # Turn off the axis labels and parameters
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
    # Original image
    # Add circles and letters to the left subplot (a)
    axs[0].add_patch(patches.Circle((0.5, 3.5), 0.5, fill=False)) # 'A'
    axs[0].add_patch(patches.Circle((3.5, 3.5), 0.5, fill=False)) # 'B'
    axs[0].add_patch(patches.Circle((0.5, 0.5), 0.5, fill=False)) # 'C'
    axs[0].add_patch(patches.Circle((3.5, 0.5), 0.5, fill=False)) # 'D'
    axs[0].add_patch(patches.Circle((2.5, 1.5), 0.5, fill=False)) # 'P'
    axs[0].text(0.5, 3.5, 'A', ha='center', va='center')
    axs[0].text(3.5, 3.5, 'B', ha='center', va='center')
    axs[0].text(0.5, 0.5, 'C', ha='center', va='center')
    axs[0].text(3.5, 0.5, 'D', ha='center', va='center')
    axs[0].text(2.5, 1.5, 'P', ha='center', va='center')
    # Compressed image
    # Add circles and letters to the right subplot (b)
    axs[1].add_patch(patches.Circle((0.5, 3.5), 0.5, fill=False)) # 'A'
    axs[1].add_patch(patches.Circle((1.5, 3.5), 0.5, fill=False)) # 'B'
    axs[1].add_patch(patches.Circle((0.5, 2.5), 0.5, fill=False)) # 'C'
    axs[1].add_patch(patches.Circle((1.5, 2.5), 0.5, fill=False)) # 'D'
    axs[1].text(0.5, 3.5, 'A', ha='center', va='center')
    axs[1].text(1.5, 3.5, 'B', ha='center', va='center')
    axs[1].text(0.5, 2.5, 'C', ha='center', va='center')
    axs[1].text(1.5, 2.5, 'D', ha='center', va='center')
    # Set subplot titles
    axs[0].set_title('(a)', loc='left') # left
    axs[1].set_title('(b)', loc='left') # right
    # Adjust the plot to fit figures without overlapping
    plt.tight_layout()
    # Save the plot
    plt.savefig('image_compression.png')
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
