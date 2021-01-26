import sys


def add_path(path):
    print("added: {0}".format(path))
    if path not in sys.path:
        sys.path.insert(0, path)


CAFFE_PATH = ""
add_path(CAFFE_PATH)
