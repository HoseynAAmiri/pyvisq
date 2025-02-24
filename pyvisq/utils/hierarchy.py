import os
import ast
import pyvisq


def get_classes_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=file_path)
    classes = []
    for n in node.body:
        if isinstance(n, ast.ClassDef) and 'Params' not in n.name:
            classes.append(n.name)
    return classes


def generate_hierarchy_tree(root_dir):
    hierarchy = {}
    for dirpath, _, filenames in os.walk(root_dir):
        relative_path = os.path.relpath(dirpath, root_dir)
        if relative_path == ".":
            relative_path = ""

        for filename in filenames:
            if filename.endswith(".py") and filename != "model.py":
                file_path = os.path.join(dirpath, filename)
                class_list = get_classes_from_file(file_path)
                if class_list:
                    hierarchy[os.path.join(
                        relative_path, filename)] = class_list
    return hierarchy


def print_hierarchy_tree(hierarchy):
    for file, classes in sorted(hierarchy.items()):
        print(file[:-3])
        for cls in classes:
            print(f"  ├── {cls}")


def show():
    root_directory = os.path.dirname(pyvisq.__file__)
    hierarchy_tree = generate_hierarchy_tree(root_directory)
    print_hierarchy_tree(hierarchy_tree)
