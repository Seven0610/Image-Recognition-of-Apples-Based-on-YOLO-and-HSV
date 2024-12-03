import os

# Set folder path
folder_path = 'VOCdevkit/txt'

# Gets the path of all txt files in the folder
txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

# Loop through each txt file
for file_path in txt_files:
    # Read file contents
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove empty lines and lines with only whitespace characters
    lines = [line for line in lines if line.strip()]

    # Write back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"Processed file:{file_path}")

print("Processing complete!")