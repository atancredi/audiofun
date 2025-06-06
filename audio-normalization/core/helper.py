from os import makedirs
from os.path import exists


def ensure_dir(path):
    if not exists(path):
        makedirs(path)
