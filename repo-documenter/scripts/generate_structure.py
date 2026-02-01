import os
import argparse
from pathlib import Path
import fnmatch

def should_ignore(name, ignore_patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns)

def print_tree(directory, ignore_patterns=None, prefix=""):
    if ignore_patterns is None:
        ignore_patterns = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '*.pyc', '.DS_Store', '.idea', '.vscode'}

    path = Path(directory)
    if not path.is_dir():
        print(f"Error: {directory} is not a directory")
        return

    try:
        # Get sorted list of items
        items = sorted([
            item for item in path.iterdir()
            if not should_ignore(item.name, ignore_patterns)
        ], key=lambda x: (x.is_file(), x.name.lower())) # Dirs first, then files
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return

    count = len(items)
    for index, item in enumerate(items):
        connector = "└── " if index == count - 1 else "├── "
        print(f"{prefix}{connector}{item.name}")
        
        if item.is_dir():
            extension = "    " if index == count - 1 else "│   "
            print_tree(item, ignore_patterns, prefix=prefix + extension)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a file structure tree.")
    parser.add_argument("path", nargs="?", default=".", help="Root directory path")
    args = parser.parse_args()
    
    target_path = os.path.abspath(args.path)
    print(f"Directory structure for: {target_path}")
    print_tree(target_path)
