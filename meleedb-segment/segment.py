import argparse
import os.path
from json import dump

import cv2

from core.segmenter import Segmenter
from core.viewfinder import Viewfinder

def __main__(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("file", help="filepath for the VOD to be segmented", type=str)
    parser.add_argument("outfile", help="filepath for the output JSON file", type=str)

    args = parser.parse_args(args)

    stream = os.path.realpath(args.file)
    outfile = os.path.realpath(args.outfile)
    
    filename = os.path.basename(stream)
    filename, _ = os.path.splitext(filename)

    match = Segmenter(stream)
    match.parse()

    with open(outfile, "w") as f:
        dump(match.data, f, default=lambda obj: obj.__dict__,
             indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
