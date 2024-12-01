import os
import fileinput
import sys

def modify_script_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'script.sh' in filenames:
            script_path = os.path.join(dirpath, 'script.sh')
            print(f"Modifying: {script_path}")
            
            with fileinput.FileInput(script_path, inplace=True) as file:
                for line in file:
                    if '#SBATCH --time=78:00:00' in line:
                        print(line.replace('78:00:00', '71:00:00'), end='')
                    else:
                        print(line, end='')

if __name__ == "__main__":
    root_directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    modify_script_files(root_directory)
    print("Modification complete!")