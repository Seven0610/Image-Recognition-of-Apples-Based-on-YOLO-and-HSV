import pandas as pd
from openpyxl import Workbook
import os

# Create a new Excel file
wb = Workbook()
ws = wb.active

folder_path = r"D:\\APMCM\\Attachment1"# Replace with a folder path containing 200 text files

# Walk through every text file in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".txt"): # Only process text files ending in.txt
        file_path = os.path.join(folder_path, filename)

        # Read the contents of the text file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Add content to a row in an Excel table
        ws.append([filename, content])

    # Save every 10 text files processed
    if i % 10 == 0:
        wb.save("output.xlsx")

# Save the Excel file
wb.save("output.xlsx")
