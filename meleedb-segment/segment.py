import argparse
import os.path
import logging
from json import dump
import sys

import cv2

from core.segmenter import Segmenter
from core.viewfinder import Viewfinder

logger = logging.getLogger(__name__)

def __main__(args):
    parser = argparse.ArgumentParser(description="",
                                     epilog="exactly one of --output [file] or"
                                            "--stdout is required.")
    parser.add_argument("infile", help="filepath for the VOD to be segmented", type=str)
    parser.add_argument("-o", "--output", help="filepath for the output JSON file", type=str)
    parser.add_argument("--stdout", help="print segmentation data to stdout", action="store_true")

    args = parser.parse_args(args)

    stream = os.path.realpath(args.infile)

    if args.stdout and not args.output:
        output = sys.stdout
    elif args.output and not args.stdout:
        output = os.path.realpath(args.output)
    else:
        raise ArgumentError('exactly one of --output [file] or --stdout is required')
    
    filename = os.path.basename(stream)
    filename, _ = os.path.splitext(filename)

    match = Segmenter(stream)
    match.parse()

    logging.warn("Segmentation succeeded!")
    if args.stdout:
        logger.warn("Writing data to <stdout>...")
        dump(match.data, sys.stdout, default=lambda obj: obj.__dict__,
             indent=4, sort_keys=True)
        print()
    else:
        with open(output, "w") as f:
            logger.warn("Writing data to {0}...".format(f.name))
            dump(match.data, f, default=lambda obj: obj.__dict__,
                 indent=4, sort_keys=True)


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
