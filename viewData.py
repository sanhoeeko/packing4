from visualize import loadAll
import traceback

if __name__ == '__main__':
    target_dir = input('data folder name: ')
    viewer = loadAll(target_dir).print()
    while True:
        try:
            print(viewer.parse(input('>>>')))
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
