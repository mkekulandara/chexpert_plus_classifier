import pandas as pd
import json

# Path to your JSON file
json_file_path = '/Users/chamudikashmila/Documents/chex_classify/chexpert_plus_classifier/data/impression_fixed.json'

# Read the file line by line, and load each JSON object
data = []
with open(json_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)
print(df.head(2))

# List of columns to be cleaned
columns_to_clean = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 
                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']

# Function to print the count of -1s and NaNs in each column
def print_counts(df, columns):
    for column in columns:
        num_neg_ones = (df[column] == -1).sum()
        num_nans = df[column].isna().sum()
        print(f"{column}: -1s = {num_neg_ones}, NaNs = {num_nans}")

# Print counts before cleaning
print("Before cleaning:")
print_counts(df, columns_to_clean)

# Replace NaN and -1 values with 0 in the specified columns
df[columns_to_clean] = df[columns_to_clean].replace([-1, pd.NA, None], 0)

# Print counts after cleaning
print("\nAfter cleaning:")
print_counts(df, columns_to_clean)

# Display the first few rows of the modified DataFrame
print("\nModified DataFrame:")
print(df.head(2))

# List of columns to be cleaned
target_cols = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 
                    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                    'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']


# Replace NaN values in the target columns with 0
df[target_cols] = df[target_cols].fillna(0)

# Create a new column 'data_label' that contains a list of the target column values for each row
df['data_label'] = df[target_cols].values.tolist()

# Display the first few rows of the DataFrame, including the new 'data_label' column
print(df[['data_label']].head(2))

# Optionally, display the first few rows of the entire DataFrame to check the label generation
print(df.head(2))