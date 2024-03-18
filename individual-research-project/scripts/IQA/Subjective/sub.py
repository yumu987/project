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

################################################
# Subjective input data comes from Microsoft Forms
################################################

import matplotlib.pyplot as plt

nima_input_data = {
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

subjective_input_data = {
    "001.jpg": 7.36,
    "002.jpg": 6.58,
    "003.jpg": 6.89,
    "004.jpg": 6.81,
    "005.jpg": 7.09,
    "006.jpg": 7.53,
    "007.jpg": 7.29,
    "008.jpg": 7.09,
    "009.jpg": 6.64,
    "010.jpg": 7.38,
    "011.jpg": 7.69,
    "012.jpg": 7.14,
    "013.jpg": 6.72,
    "014.jpg": 7.09,
    "015.jpg": 7.92,
    "016.jpg": 6.47,
    "017.jpg": 7.56,
    "018.jpg": 7.25,
    "019.jpg": 6.81,
    "020.jpg": 6.75,
    "021.jpg": 7.74,
    "022.jpg": 7.72,
    "023.jpg": 6.97,
    "024.jpg": 6.56,
    "025.jpg": 6.63
}

index_array = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
     "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
     "021", "022", "023", "024", "025"]

subjective_values_array = []

def compute_mean():
    nima_input_values = [value for value in nima_input_data.values()]
    subjective_input_values = [value for value in subjective_input_data.values()]
    nima_tmp = 0.0
    subjective_tmp = 0.0
    for i in range(25):
        nima_tmp = nima_tmp + nima_input_values[i]
        subjective_tmp = subjective_tmp + subjective_input_values[i]
    nima_mean = nima_tmp/25
    subjective_mean = subjective_tmp/25
    return nima_mean, subjective_mean

def plot_bar_chart(nima_mean, subjective_mean):
    # Sample data
    sample_name_array = ['NIMA input score', 'Subjective input score']
    sample_data_array = [nima_mean, subjective_mean]
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(sample_name_array, sample_data_array)
    # Customise x-axis labels
    plt.xticks(rotation=15, ha="right", rotation_mode="anchor", fontsize=8)
    # Title and label the bar chart
    plt.title('Average score in 25 images')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('Input_NIMA_Subjective_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("NIMA_Subjective: Bar chart has been drawn")

def bar():
    subjective_input_values = [value for value in subjective_input_data.values()]
    for i in range(25):
        subjective_values_array.append(subjective_input_values[i])
    # Enable grid on
    # plt.grid(True)
    plt.gca().yaxis.grid(True, zorder=0)
    # Plot the bar chart
    plt.bar(index_array, subjective_values_array)
    # Customise x-axis labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # Title and label the bar chart
    plt.title('Subjective score in 25 images (input)')
    # plt.xlabel('Index')
    plt.ylabel('Score')
    # Save the figure
    plt.savefig('Subjective_bar_chart.png')
    # Show the plot
    # plt.show()
    # Close the plot
    plt.close()
    # Indication of plotting bar chart completed
    print("Subjective: Bar chart has been drawn")

def main():
    nima_mean, subjective_mean = compute_mean()
    plot_bar_chart(nima_mean, subjective_mean)
    bar()

if __name__ == "__main__":
    main()
