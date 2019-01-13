from train.trainer import Trainer
import argparse


def __main__():
    parser = argparse.ArgumentParser(description="Generate training images from a VOD.")
    parser.add_argument("infile", type=str, help="video to be processed")

    args = parser.parse_args()

    tr = Trainer(args.infile)

    tr.train()


if __name__ == "__main__":
    __main__()
