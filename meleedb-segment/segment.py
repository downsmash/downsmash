import argparse
import os.path
import logging
from json import dump

import cv2

from core.segmenter import Segmenter
from core.viewfinder import Viewfinder

def __main__(args):
    parser = argparse.ArgumentParser(description='')
<<<<<<< HEAD
    parser.add_argument("file", help="filepath for the VOD to be segmented", type=str)
    parser.add_argument("outfile", help="filepath for the output JSON file", type=str)
=======
    parser.add_argument("file", help="stream", type=str)
    parser.add_argument("outfile", help="output file", type=str)
>>>>>>> master

    args = parser.parse_args(args)

    stream = os.path.realpath(args.file)
<<<<<<< HEAD
    filename = os.path.basename(stream)
    filename, _ = os.path.splitext(filename)

    LOGFMT = "[{0}] [%(msecs)d] [%(filename)s/%(funcName)s] %(message)s".format(filename)
    logging.basicConfig(format=LOGFMT)
=======
    outfile = os.path.realpath(args.outfile)
>>>>>>> master

    print(stream)
    match = Segmenter(stream)
    match.parse()

<<<<<<< HEAD
    with open(args.outfile, "w") as f:
=======
    with open(outfile, "w") as f:
>>>>>>> master
        dump(match.data, f, default=lambda obj: obj.__dict__,
             indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
