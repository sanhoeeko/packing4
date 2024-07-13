import os
import shutil

from strip_hints import strip_file_to_string


def backup_directory(dir_path):
    parent_dir = os.path.dirname(dir_path)
    if parent_dir == '':
        raise Exception("Cannot back up")
    backup_dir = os.path.join(parent_dir, 'back_up')
    shutil.copytree(dir_path, backup_dir)


def process_directory(dir_path):
    backup_directory(dir_path)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                code_without_hints = strip_file_to_string(file_path)
                with open(file_path, 'w') as f:
                    f.write(code_without_hints)


process_directory(os.getcwd())
