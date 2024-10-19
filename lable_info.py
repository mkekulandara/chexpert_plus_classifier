import pandas as pd
import json

print("JFNLJFJBLADFBN")
# Path to your JSON file
json_file_path = '/Users/chamudikashmila/Documents/chex_classify/chexpert_plus_classifier/data/impression_fixed.json'

# Read the file line by line, and load each JSON object
data = []
with open(json_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head(2))

# Print the columns of the DataFrame
print("Columns in the DataFrame:")
print(df.columns)

# Print the value counts for each column
print("\nValue Counts for Each Column:")
for column in df.columns:
    print(f"\nColumn: {column}")
    print(df[column].value_counts())

# Path to output text file
output_file_path = 'impression_fixed_results.txt'

# Open a file to write the results
with open(output_file_path, 'w') as output_file:
    
    # Write number of data rows in the DataFrame
    output_file.write(f"Number of data rows: {len(df)}\n\n")
    
    # Write the columns in the DataFrame
    output_file.write("Columns in the DataFrame:\n")
    output_file.write(', '.join(df.columns) + "\n\n")
    
    # Write value counts for each column (ignoring the first column 'path_to_image')
    output_file.write("Value Counts for Each Column (excluding 'path_to_image'):\n\n")
    for column in df.columns[1:]:  # Skip the first column
        output_file.write(f"Column: {column}\n")
        output_file.write(df[column].value_counts().to_string() + "\n\n")

# Inform the user that the results have been written
print(f"Results written to {output_file_path}")    