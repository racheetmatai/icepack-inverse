import os
import matplotlib.pyplot as plt
import pandas as pd
import firedrake
from data_preprocessing import clean_imported_data

# Assuming your data folder is in the current working directory
data_folder = 'completed'

# Loop through each folder
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    try:
        # Check if the item in the data folder is a directory
        if os.path.isdir(folder_path) and not folder_name.startswith(".ipynb"):
    
            print("Running folder:", folder_name)
            file_names = [file for file in os.listdir(folder_path) if file.startswith("ASE")]
            print(file_names)
            name = file_names[-1]
            name = name[:40]
    
            print(folder_name, " ", name)
                
            # Run your code for each folder
            file_path = os.path.join(folder_path, name)
            clean_imported_data(file_path)
            
    except Exception as e:
        # Handle the exception (print an error message, log it, etc.)
        print(f"Error processing folder {folder_name}: {e}")