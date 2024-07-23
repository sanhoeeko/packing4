import json
import os
import re
import shutil

std_params = ['N', 'n', 'd', 'phi0', 'potential_name', 'Gammas', 'SIBLINGS']


def createProject(src_folder, config_file):
    # 读取配置文件
    with open(config_file, 'r') as f:
        configs = json.load(f)

    for _dst_folder, config in configs.items():
        dst_folder = 'EXEC' + _dst_folder
        # 复制项目文件夹
        shutil.copytree(src_folder, dst_folder)

        # 定义文件路径
        example_file = os.path.join(dst_folder, 'deploy.py')

        # 读取example.py文件
        with open(example_file, 'r') as f:
            lines = f.readlines()

        # 定义要替换的参数
        params_to_replace = list(config.keys())
        if set(params_to_replace) != set(std_params):
            raise ValueError("Please check if there are unknown parameters.")

        # 遍历参数，检查是否在文件中
        for param in params_to_replace:
            if not any(param in line for line in lines):
                raise ValueError(f"Parameter '{param}' not found in the code file.")

        # 遍历每一行，查找并替换参数
        for i, line in enumerate(lines):
            if '# mutable' in line:
                for param in params_to_replace:
                    if param in line:
                        # 使用正则表达式找到参数值并替换
                        pattern = rf'(\s*{param}\s*=\s*)([^,)]*)(,|\))?'
                        replacement = rf'\g<1>{config[param]}\g<3>'
                        lines[i] = re.sub(pattern, replacement, line)

        # 将修改后的内容写回文件
        with open(example_file, 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    createProject('code', 'config.json')
