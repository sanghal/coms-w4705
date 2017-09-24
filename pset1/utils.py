import os


def refresh(path):
    if os.path.exists(path):
        os.remove(path)
    return path

def fname(old_name, sep):
    old = old_name.split('.')
    old.append(sep)
    old[-1], old[-2] = old[-2], old[-1]
    return '.'.join(old)

def fext(old_name, ext):
    old = old_name.split('.')
    old[-1] = ext
    return '.'.join(old)
