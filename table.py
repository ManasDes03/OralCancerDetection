import pandas as pd

# Load the CSV file
file_path = "data/Sri Lankan Dataset/Imagewise_Data.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Specify the columns
column_1 = "Category"   # Replace with the first column name
column_2 = "Clinical Diagnosis"  # Replace with the second column name

# Group the second column by the first column
grouped_data = df.groupby(column_1)[column_2].unique().apply(list).reset_index()

# Print the grouped data
print(grouped_data)

# Optionally, save to a new CSV
grouped_data.to_csv("grouped_values.csv", index=False)
