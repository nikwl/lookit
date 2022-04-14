import os


def get_file(d, fe, fn=None):
    """
    Given a root directory and a list of file extensions, recursively
    return all files in that directory that have that extension.
    """
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            yield from get_file(fp, fe, fn)
        elif os.path.splitext(fp)[-1] in fe:
            if fn is None:
                yield fp
            elif fn == os.path.splitext(f)[0]:
                yield fp