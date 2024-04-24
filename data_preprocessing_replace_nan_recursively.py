import os
import numpy as np

from data_preprocessing import replace_nans_recursive

# Replace "your_base_folder" with the actual path to your main folder
# Replace "total_folders" with the total number of folders
total_folders = 34  # Change this to the actual total number of folders
base_folder = "completed"  # Change this to the actual path to your main folder

# Iterate over folder numbers from 1 to total_folders
#for current_folder_number in range(1, total_folders + 1):
for current_folder_number in range(1, 2):
    print("FOLDER ___________________ : ", current_folder_number)
    current_folder_path = os.path.join(base_folder, str(current_folder_number))
    next_folder_number = (current_folder_number % total_folders) + 1
    next_folder_path = os.path.join(base_folder, str(next_folder_number))

    replace_nans_recursive(base_folder, current_folder_path, next_folder_path, total_folders)


current_folder_path = os.path.join(base_folder, str(34))
next_folder_number = 33
next_folder_path = os.path.join(base_folder, str(next_folder_number))

replace_nans_recursive(base_folder, current_folder_path, next_folder_path, total_folders)