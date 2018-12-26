from __future__ import unicode_literals

import segment
import pandas as pd
import youtube_dl


def __main__(args):
    cachefile = "cache"
    videosfile = "videos_melee.tsv"

    ydl_opts = {
        "format": "134",
        "outtmpl": "vods/%(id)s.%(ext)s"
    }

    with open(cachefile, "a+") as cache:
        videos = pd.read_csv(videosfile, sep="\t", header=None)
        for video in videos.sample(n=10)[0].tolist():
            if video not in cache.read().split("\n"):
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video])
                try:
                    segment.__main__(["vods/{0}.mp4".format(video)])
                except RuntimeError:
                    pass

                cache.write("{0}\n".format(video))


if __name__ == "__main__":
    import sys
    __main__(sys.argv[1:])
