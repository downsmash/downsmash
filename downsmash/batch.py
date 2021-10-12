from __future__ import unicode_literals

import logging
import sys
import os.path
import youtube_dl

import segment

LOGGER = logging.getLogger(__name__)


def __main__():
    batch_dir = os.path.dirname("batch/")
    cachefile = os.path.join(batch_dir, "cache")

    ydl_opts = {
        "format": "134",
        "outtmpl": "vods/%(id)s.%(ext)s"
    }

    with open(cachefile, "a+") as cache:
        for video in sys.stdin:
            # Kill the newline at the end
            video = video.strip()
            if video not in cache.read().split("\n"):
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    try:
                        ydl.download([video])
                        segment.__main__(["vods/{0}.mp4".format(video),
                                          "-o",
                                          "../data/{0}.json".format(video)])
                    except youtube_dl.utils.DownloadError:
                        # Video predates 480p
                        LOGGER.error("Segmentation failed!")
                        err = ("The parser does not currently support videos "
                               "in resolution lower than 480p.")
                        LOGGER.error("Guru meditation: %s", err)
                        continue
                    except RuntimeError as err:
                        # Parser had a problem somewhere
                        LOGGER.error("Segmentation failed!")
                        LOGGER.error("Guru meditation: %s", str(err))

                cache.write("{0}\n".format(video))


if __name__ == "__main__":
    # __main__(sys.argv[1:])
    __main__()
