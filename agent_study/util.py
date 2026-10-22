from pathlib import Path


def write_file(file_path: Path | str, str_txt: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(str_txt)


def read_file(file_path: Path | str) -> str | None:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def append_file(file_path: Path | str, str_txt: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(str_txt)
