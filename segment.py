import argparse
from MatchParser import MatchParser
from StreamParser import MeleeVODParser


def __main__():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("input", help="the JSON file (filename/players)", type=str)
    parser.add_argument("file", help="stream", type=str)
    parser.add_argument("-n", "--new", action='store_true')

    args = parser.parse_args()

    # with open(args.input) as f:
    #    data = json.load(f)
    stream = args.file

    if args.new:
        match = MeleeVODParser(stream)
    else:
        match = MatchParser(stream)
    match.parse()

    def timeify(n):
        mins, secs = int(n // 60), n % 60
        return "{:d}:{:05.2f}".format(mins, secs)

    for chunk in match.chunks:
        start, end = chunk
        # print("{0} - {1}".format(timeify(start), timeify(end)))
        match.get_percents(chunk)


if __name__ == "__main__":
    __main__()
