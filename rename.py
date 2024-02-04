import os
import sys

def rename_files(folder_path, start_index):
    files = os.listdir(folder_path)
    files.sort()

    for index, file_name in enumerate(files):
        if file_name.endswith('.png'):
            new_name = f"{index + 1 + start_index}.png"
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} to {new_name}")

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python rename.py <folder_path> <start_index>")
        sys.exit(1)

    folder_path = sys.argv[1]
    start_index = int(sys.argv[2])

    
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a valid directory.")
        sys.exit(1)

    rename_files(folder_path, start_index)