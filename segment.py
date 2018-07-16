import argparse
import logging
import pandas as pd

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

    corr_series = match.detect_match_chunks()

    df = pd.DataFrame(corr_series, columns=('time', 'corr'))

    medians = df['corr'].rolling(5, center=True).median()
    medians = medians.fillna(method='bfill').fillna(method='ffill')
    df['median'] = medians

    plot = df.plot(x='time')
    fig = plot.get_figure()

    fig.savefig('ts.png')
    df.to_csv('ts.csv')


if __name__ == "__main__":
    __main__()
