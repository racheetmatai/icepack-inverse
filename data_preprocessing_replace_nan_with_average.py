import os

# Assuming your function is defined in a module named my_module
from data_preprocessing import take_average_where_nan

def process_completed_folder(base_folder, total_folders):
    for folder_number in range(1, total_folders+1):

        print("------------ IN FOLDER -------------:", folder_number)

        file_types = ['vx', 'vy', 'ex', 'ey']
    
        for file_type in file_types:
            current_folder = os.path.join(base_folder, str(folder_number))
            previous_folder = os.path.join(base_folder, str(folder_number - 1))
            next_folder = os.path.join(base_folder,  str(folder_number + 1))
    
            take_average_where_nan(base_folder, current_folder, previous_folder, next_folder, total_folders, file_type)

if __name__ == "__main__":
    base_folder = "completed"
    total_folders = 34

    process_completed_folder(base_folder, total_folders)