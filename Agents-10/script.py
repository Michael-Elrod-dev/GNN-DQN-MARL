import os

# Function to rename directories by adding a '+' to the beginning of the name
def rename_directories(root_directory):
    for dir_name in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, dir_name)

        if os.path.isdir(dir_path):
            new_dir_name = f"+{dir_name}"
            new_dir_path = os.path.join(root_directory, new_dir_name)

            try:
                os.rename(dir_path, new_dir_path)
                print(f"Renamed: {dir_name} -> {new_dir_name}")
            except OSError as e:
                print(f"Error renaming {dir_name}: {e}")

# Replace '.' with the path to your main directory if running from within the parent directory
root_directory = '.'
rename_directories(root_directory)
