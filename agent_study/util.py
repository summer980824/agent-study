from pathlib import Path


def write_file(file_path_str, str_txt):
    file_path = Path(file_path_str)
    with file_path.open("w", encoding="utf-8") as file:
        file.write(str_txt)


def read_file(file_path_str):
    file_path = Path(file_path_str)
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def append_file(file_path_str, str_txt):
    file_path = Path(file_path_str)
    with file_path.open("a", encoding="utf-8") as file:
        file.write(str_txt)