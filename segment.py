import argparse

from MeleeVODParser import MeleeVODParser


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("file", help="stream", type=str)

    args = parser.parse_args()

    # with open(args.input) as f:
    #    data = json.load(f)
    stream = args.file

    match = MeleeVODParser(stream)
    match.parse()

    def timeify(n):
        mins, secs = int(n // 60), n % 60
        return "{:d}:{:05.2f}".format(mins, secs)

    for start, end in match.chunks:
        print(timeify(start), timeify(end))

if __name__ == "__main__":
    __main__()
