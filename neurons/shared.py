import sys, pathlib


def add_repo_to_path():
    """
    Add the base repository to path.
    """
    file_path = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
    if not file_path in sys.path:
        sys.path.append(file_path)
