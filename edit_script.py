import os

def delete_backup_files():
    target_dirs = ['GNN', 'DQN']
    deleted_files = []
    
    for dir_name in target_dirs:
        if not os.path.exists(dir_name):
            print(f"Warning: Directory {dir_name}/ not found")
            continue
            
        # Walk through the specific directory and its subdirectories
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if file.endswith('.bak'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")
    
    return deleted_files

if __name__ == "__main__":
    print("Starting deletion of .bak files in GNN/ and DQN/ directories...")
    deleted_files = delete_backup_files()
    
    if deleted_files:
        print(f"\nTotal files deleted: {len(deleted_files)}")
    else:
        print("\nNo backup files were found")