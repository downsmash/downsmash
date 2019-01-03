import argparse
import os.path
from json import dump

from core.segmenter import Segmenter


def __main__(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("file", help="stream", type=str)

    args = parser.parse_args(args)

    stream = os.path.realpath(args.file)
    print(stream)

    match = Segmenter(stream)
    match.parse()

    _, basename = os.path.split(stream)
    basename, _ = os.path.splitext(basename)

    outpath = os.path.join("data/{0}.json".format(basename))
    with open(outpath, "w") as f:
        dump(match.data, f, default=lambda obj: obj.__dict__,
             indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
