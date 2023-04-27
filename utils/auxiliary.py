from pathlib import Path

def get_files(
        path, extension = '.wav'
):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))