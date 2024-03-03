################################################
# plot
################################################

################################################
# Dataset from Nick Yue
# Special thanks to nick :)
# Pre-processing images produced by Yumu Xie
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

##############################################
# Programming environment: WSL Linux subsystem of Windows
# Python version: 3.7.9
# Python setup is based on 'pyenv' (Simple Python version management)
# https://github.com/pyenv/pyenv
################################################

import matplotlib.pyplot as plt

nasnet_data = {
    "025.jpg": 5.94897,
    "009.jpg": 5.88139,
    "024.jpg": 5.86458,
    "016.jpg": 5.73838,
    "008.jpg": 5.70115,
    "017.jpg": 5.69317,
    "018.jpg": 5.67740,
    "019.jpg": 5.53611,
    "007.jpg": 5.46491,
    "013.jpg": 5.44254,
    "001.jpg": 5.32645,
    "021.jpg": 5.27846,
    "002.jpg": 5.26355,
    "015.jpg": 5.25994,
    "006.jpg": 5.24959,
    "022.jpg": 5.23635,
    "012.jpg": 5.22039,
    "023.jpg": 5.15755,
    "014.jpg": 5.15003,
    "003.jpg": 5.13021,
    "020.jpg": 5.12882,
    "010.jpg": 4.96822,
    "011.jpg": 4.90023,
    "004.jpg": 4.64797,
    "005.jpg": 4.43923
}

index_array = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
     "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
     "021", "022", "023", "024", "025"]

# Sample data
sorted_nasnet_name = []
sorted_nasnet_value = []

def sort_algorithm():
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_nasnet_data = dict(sorted(nasnet_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False))

    for filename, value in sorted_nasnet_data.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_name.append(filename)
        sorted_nasnet_value.append(value)

def bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def main():
    sort_algorithm()
    bar()
    line()
    general()

if __name__ == "__main__":
    main()
