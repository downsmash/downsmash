import argparse
from MatchParser import MatchParser


def __main__():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("input", help="the JSON file (filename/players)", type=str)
    parser.add_argument("file", help="stream", type=str)

    args = parser.parse_args()

    # with open(args.input) as f:
    #    data = json.load(f)
    stream = args.file

    MatchParser(stream).parse()


if __name__ == "__main__":
    __main__()
