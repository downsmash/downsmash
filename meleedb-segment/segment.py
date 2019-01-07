import argparse
import os.path
from json import dump

from core.segmenter import Segmenter


def __main__(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("file", help="filepath for the VOD to be segmented", type=str)
    parser.add_argument("outfile", help="filepath for the output JSON file", type=str)

    args = parser.parse_args(args)

    stream = os.path.realpath(args.file)
    print(stream)

    match = Segmenter(stream)
    match.parse()

    with open(args.outfile, "w") as f:
        dump(match.data, f, default=lambda obj: obj.__dict__,
             indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
