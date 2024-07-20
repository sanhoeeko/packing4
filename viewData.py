from visualize import loadAll

if __name__ == '__main__':
    target_dir = input('data folder name: ')
    viewer = loadAll(target_dir).print()
    while True:
        try:
            viewer.parse(input('>>>'))
        except Exception as e:
            print(e)
