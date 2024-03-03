import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Read the CSV file into a pandas DataFrame
    csv_file_path = 'your_file.csv'  # Replace 'your_file.csv' with the actual file path
    data = pd.read_csv(csv_file_path)

    # Assuming your CSV has columns named 'Category' and 'Value'
    # Adjust column names based on your CSV file
    category_column = 'Category'
    value_column = 'Value'

    # Plotting the bar chart
    plt.bar(data[category_column], data[value_column])

    # Adding labels and title
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.title('Bar Chart from CSV Data')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
    