################################################
# nasnet_average_plot.py
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

def compute_mean():
    input_values = [value for value in nasnet_data.values()]
    nearest_neighbor_values = [value for value in nearest_neighbor_data.values()]
    bilinear_values = [value for value in bilinear_data.values()]
    bicubic_values = [value for value in bicubic_data.values()]
    lanczos_values = [value for value in lanczos_data.values()]
    pixel_area_relation_values = [value for value in pixel_area_relation_data.values()]
    input_tmp = 0.0
    nearest_neighbor_tmp = 0.0
    bilinear_tmp = 0.0
    bicubic_tmp = 0.0
    lanczos_tmp = 0.0
    pixel_area_relation_tmp = 0.0
    for i in range(25):
        input_tmp = input_tmp + input_values[i]
    mean_input = input_tmp/25
    for i in range(25):
        nearest_neighbor_tmp = nearest_neighbor_tmp + nearest_neighbor_values[i]
    mean_nearest_neighbor = nearest_neighbor_tmp/25
    for i in range(25):
        bilinear_tmp = bilinear_tmp + bilinear_values[i]
    mean_bilinear = bilinear_tmp/25
    for i in range(25):
        bicubic_tmp = bicubic_tmp + bicubic_values[i]
    mean_bicubic = bicubic_tmp/25
    for i in range(25):
        lanczos_tmp = lanczos_tmp + lanczos_values[i]
    mean_lanczos = lanczos_tmp/25
    for i in range(25):
        pixel_area_relation_tmp = pixel_area_relation_tmp + pixel_area_relation_values[i]
    mean_pixel_area_relation = pixel_area_relation_tmp/25
    return mean_input, mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation

def plot_bar_chart(mean_input, mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation):
    # Sample data
    sample_name_array = ['Input', 'Nearest-neighbor', 'Bilinear', 'Bicubic', 'Lanczos', 'Pixel area relation']
    sample_data_array = [mean_input, mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Average NIMA in 25 images')
    plt.xlabel('Downsampling (Distortion)')
    plt.ylabel('Neural Image Assessment')
    # Save the figure
    plt.savefig('NIMA_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("NIMA: Bar chart has been drawn")

def main():
    mean_input, mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation = compute_mean()
    plot_bar_chart(mean_input, mean_nearest_neighbor, mean_bilinear, mean_bicubic, mean_lanczos, mean_pixel_area_relation)

if __name__ == "__main__":
    main()
