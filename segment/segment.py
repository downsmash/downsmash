import argparse
import os.path
import logging
from json import dump
import sys

from core.segmenter import Segmenter

LOGGER = logging.getLogger(__name__)

def __main__(args):
    parser = argparse.ArgumentParser(description="",
                                     epilog="exactly one of --output [file] "
                                            "or --stdout is required.")
    parser.add_argument("infile",
                        help="filepath for the VOD to be segmented",
                        type=str)
    parser.add_argument("-o", "--output",
                        help="filepath for the output JSON file",
                        type=str)
    parser.add_argument("--stdout",
                        help="print segmentation data to stdout",
                        action="store_true")

    args = parser.parse_args(args)

    stream = os.path.realpath(args.infile)

    if args.stdout and not args.output:
        output = sys.stdout
    elif args.output and not args.stdout:
        output = os.path.realpath(args.output)
    else:
        raise ArgumentError("exactly one of --output [file] "  # pylint:disable=undefined-variable
                            "or --stdout is required")

    filename = os.path.basename(stream)
    filename, _ = os.path.splitext(filename)

    match = Segmenter(stream)
    match.parse()

    LOGGER.warning("Segmentation succeeded!")
    data = vars(match)
    if args.stdout:
        LOGGER.warning("Writing data to <stdout>...")
        dump(data, sys.stdout, default=repr, indent=4, sort_keys=True)
        print()
    else:
        with open(output, "w") as outfile:
            LOGGER.warning("Writing data to %s...", outfile.name)
            dump(data, outfile, default=repr, indent=4, sort_keys=True)


if __name__ == "__main__":
    __main__(sys.argv[1:])
