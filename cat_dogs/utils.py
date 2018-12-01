import os


def sum_of_file(dirpath):
    """sum of images.

    Args:
        dirpath: input dir

    Returns:
        image num

    """
    num = 0
    for dir in os.listdir(dirpath):
        for _ in os.listdir(dirpath + '/' + dir):
            num += 1
    return num
