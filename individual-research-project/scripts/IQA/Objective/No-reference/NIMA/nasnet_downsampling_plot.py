################################################
# nasnet_downsampling_plot.py
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

nearest_neighbor_data = {
    "009.jpg": 6.05907,
    "024.jpg": 5.821,
    "025.jpg": 5.72512,
    "019.jpg": 5.60263,
    "017.jpg": 5.49294,
    "016.jpg": 5.48008,
    "018.jpg": 5.37094,
    "007.jpg": 5.31582,
    "020.jpg": 5.26156,
    "006.jpg": 5.2205,
    "008.jpg": 5.16699,
    "013.jpg": 5.0824,
    "021.jpg": 5.04866,
    "014.jpg": 5.02159,
    "015.jpg": 5.01167,
    "012.jpg": 4.96141,
    "001.jpg": 4.89638,
    "002.jpg": 4.8383,
    "022.jpg": 4.79359,
    "003.jpg": 4.72117,
    "023.jpg": 4.53085,
    "004.jpg": 4.46936,
    "010.jpg": 4.41079,
    "005.jpg": 4.39061,
    "011.jpg": 4.16031
}

bilinear_data = {
    "009.jpg": 5.93996,
    "024.jpg": 5.82669,
    "025.jpg": 5.81991,
    "018.jpg": 5.58137,
    "016.jpg": 5.53102,
    "019.jpg": 5.52526,
    "017.jpg": 5.48072,
    "007.jpg": 5.33893,
    "020.jpg": 5.26877,
    "021.jpg": 5.21762,
    "008.jpg": 5.09696,
    "006.jpg": 5.09417,
    "013.jpg": 5.05256,
    "014.jpg": 5.03159,
    "001.jpg": 5.01213,
    "012.jpg": 4.99597,
    "015.jpg": 4.92997,
    "002.jpg": 4.8581,
    "003.jpg": 4.82576,
    "022.jpg": 4.82268,
    "023.jpg": 4.67512,
    "010.jpg": 4.64862,
    "004.jpg": 4.52856,
    "005.jpg": 4.20334,
    "011.jpg": 4.18857
}

bicubic_data = {
    "009.jpg": 6.00954,
    "025.jpg": 5.82039,
    "024.jpg": 5.78718,
    "018.jpg": 5.54448,
    "019.jpg": 5.52049,
    "016.jpg": 5.50602,
    "017.jpg": 5.43345,
    "020.jpg": 5.36068,
    "007.jpg": 5.35186,
    "006.jpg": 5.25425,
    "008.jpg": 5.19225,
    "021.jpg": 5.15233,
    "012.jpg": 5.04459,
    "014.jpg": 5.01932,
    "015.jpg": 4.95426,
    "013.jpg": 4.95116,
    "002.jpg": 4.84985,
    "001.jpg": 4.82909,
    "003.jpg": 4.7473,
    "023.jpg": 4.70949,
    "022.jpg": 4.64879,
    "004.jpg": 4.4754,
    "010.jpg": 4.47321,
    "005.jpg": 4.28488,
    "011.jpg": 4.22047
}

lanczos_data = {
    "009.jpg": 5.94464,
    "025.jpg": 5.83456,
    "024.jpg": 5.79776,
    "018.jpg": 5.55219,
    "016.jpg": 5.54072,
    "019.jpg": 5.48794,
    "017.jpg": 5.44811,
    "020.jpg": 5.36409,
    "006.jpg": 5.32037,
    "007.jpg": 5.2053,
    "021.jpg": 5.19494,
    "008.jpg": 5.14297,
    "014.jpg": 5.01716,
    "013.jpg": 5.00141,
    "012.jpg": 4.94881,
    "001.jpg": 4.93597,
    "015.jpg": 4.89811,
    "003.jpg": 4.85595,
    "002.jpg": 4.84217,
    "023.jpg": 4.74674,
    "022.jpg": 4.67196,
    "004.jpg": 4.51052,
    "010.jpg": 4.46507,
    "005.jpg": 4.33236,
    "011.jpg": 4.17248
}

pixel_area_relation_data = {
    "009.jpg": 5.92657,
    "025.jpg": 5.85245,
    "024.jpg": 5.83655,
    "016.jpg": 5.63415,
    "018.jpg": 5.5704,
    "019.jpg": 5.52512,
    "013.jpg": 5.50536,
    "007.jpg": 5.47501,
    "001.jpg": 5.47379,
    "017.jpg": 5.46652,
    "008.jpg": 5.38836,
    "015.jpg": 5.31872,
    "023.jpg": 5.29383,
    "012.jpg": 5.28195,
    "021.jpg": 5.26984,
    "010.jpg": 5.26907,
    "014.jpg": 5.22827,
    "020.jpg": 5.22138,
    "006.jpg": 5.18048,
    "022.jpg": 5.10203,
    "002.jpg": 5.06178,
    "003.jpg": 5.05925,
    "011.jpg": 4.77116,
    "004.jpg": 4.6417,
    "005.jpg": 4.10807
}

index_array = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
     "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
     "021", "022", "023", "024", "025"]

# Sample data
sorted_nasnet_nearest_neighbor_name = []
sorted_nasnet_nearest_neighbor_value = []
sorted_nasnet_bilinear_name = []
sorted_nasnet_bilinear_value = []
sorted_nasnet_bicubic_name = []
sorted_nasnet_bicubic_value = []
sorted_nasnet_lanczos_name = []
sorted_nasnet_lanczos_value = []
sorted_nasnet_pixel_area_relation_name = []
sorted_nasnet_pixel_area_relation_value = []

def sort_algorithm():
    # Nearest-neighbor
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_nearest_neighbor = dict(sorted(nearest_neighbor_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False)) # "_"
    for filename, value in sorted_nearest_neighbor.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_nearest_neighbor_name.append(filename)
        sorted_nasnet_nearest_neighbor_value.append(value)
    # Bilinear
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_bilinear = dict(sorted(bilinear_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False)) # "_"
    for filename, value in sorted_bilinear.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_bilinear_name.append(filename)
        sorted_nasnet_bilinear_value.append(value)
    # Bicubic
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_bicubic = dict(sorted(bicubic_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False)) # "_"
    for filename, value in sorted_bicubic.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_bicubic_name.append(filename)
        sorted_nasnet_bicubic_value.append(value)
    # Lanczos
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_lanczos = dict(sorted(lanczos_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False)) # "_"
    for filename, value in sorted_lanczos.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_lanczos_name.append(filename)
        sorted_nasnet_lanczos_value.append(value)
    # Pixel area relation
    # Reorder the dictionary based on the numeric part of the filenames
    sorted_pixel_area_relation = dict(sorted(pixel_area_relation_data.items(), key=lambda item: int(item[0].split(".")[0]), reverse=False)) # "_"
    for filename, value in sorted_pixel_area_relation.items():
        # Print the reordered data
        print(f"{filename}: {value}")
        # Save the reordered data
        sorted_nasnet_pixel_area_relation_name.append(filename)
        sorted_nasnet_pixel_area_relation_value.append(value)

# Nearest-neighbor
def nearest_neighbor_bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_nearest_neighbor_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (nearest-neighbor)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_nearest_neighbor_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def nearest_neighbor_line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_nearest_neighbor_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images (nearest-neighbor)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_nearest_neighbor_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def nearest_neighbor_general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_nearest_neighbor_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_nearest_neighbor_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (nearest-neighbor)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_nearest_neighbor_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

# Bilinear
def bilinear_bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_bilinear_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (bilinear)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bilinear_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def bilinear_line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_bilinear_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images (bilinear)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bilinear_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def bilinear_general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_bilinear_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_bilinear_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (bilinear)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bilinear_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    
# Bicubic
def bicubic_bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_bicubic_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (bicubic)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bicubic_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def bicubic_line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_bicubic_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images (bicubic)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bicubic_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def bicubic_general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_bicubic_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_bicubic_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (bicubic)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_bicubic_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    
# Lanczos
def lanczos_bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_lanczos_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (lanczos)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_lanczos_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def lanczos_line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_lanczos_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images (lanczos)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_lanczos_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def lanczos_general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_lanczos_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_lanczos_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (lanczos)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_lanczos_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    
# Pixel area relation
def pixel_area_relation_bar():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_pixel_area_relation_value)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (pixel area relation)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_pixel_area_relation_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def pixel_area_relation_line():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_pixel_area_relation_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the line chart
    plt.title('NIMA nasnet score in 25 images (pixel area relation)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_pixel_area_relation_line_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()

def pixel_area_relation_general():
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, sorted_nasnet_pixel_area_relation_value)
    # Plot the line chart
    plt.plot(index_array, sorted_nasnet_pixel_area_relation_value, marker='o', linestyle='-', color='b', label='Score')
    # Add legend
    plt.legend()
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('NIMA nasnet score in 25 images (pixel area relation)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('nasnet_pixel_area_relation_general_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    
# Main
def main():
    print("nasnet_downsampling_plot.py is executing")
    sort_algorithm()
    nearest_neighbor_bar()
    nearest_neighbor_line()
    nearest_neighbor_general()
    bilinear_bar()
    bilinear_line()
    bilinear_general()
    bicubic_bar()
    bicubic_line()
    bicubic_general()
    lanczos_bar()
    lanczos_line()
    lanczos_general()
    pixel_area_relation_bar()
    pixel_area_relation_line()
    pixel_area_relation_general()
    print("nasnet_downsampling_plot.py is finished")

if __name__ == "__main__":
    main()
