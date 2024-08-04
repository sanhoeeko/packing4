import os
import threading
import py7zr


def extract_7z_file(folder_path):
    zip_path = os.path.join(folder_path, 'results.7z')
    if os.path.exists(zip_path):
        with py7zr.SevenZipFile(zip_path, 'r') as archive:
            archive.reset()
            archive.extractall(path=folder_path)
        print('Extracting:', zip_path)


def collect_files(folder_name, collected_files):
    current_files = []
    for root, _, files in os.walk(folder_name):
        for file in files:
            if file.endswith('.h5') or file.endswith('.log.txt'):
                current_files.append(os.path.join(root, file))
    collected_files.extend(current_files)
    print(f'Collecting {len(current_files)} files in', folder_name)


def create_zip_from_collected_files(collected_files, zip_name):
    print(f'{len(collected_files)} files are found in total.')
    with py7zr.SevenZipFile(zip_name, 'w') as archive:
        for file in collected_files:
            archive.write(file, os.path.relpath(file, os.getcwd()))


def main():
    current_dir = os.getcwd()
    threads = []
    collected_files = []

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and item.startswith('EXEC'):
            thread = threading.Thread(target=extract_7z_file, args=(item_path,))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and item.startswith('EXEC'):
            collect_files(item_path, collected_files)

    create_zip_from_collected_files(collected_files, 'all_results.7z')


if __name__ == "__main__":
    main()
