# downsmash/segment

Part of the [Downsmash](downsma.sh) project.

Watch a video of a Melee set and output the (approximate) start and end times of contiguous Melee.

## Synopsis
Segment an individual VOD:
```
$ python segment.py [vod.mp4] -o [output.json]  # write to output.json
$ python segment.py [vod.mp4] --stdout          # write to stdout
```
Batch-process a set of VODs:
```
$ echo [youtubeID] | python batch.py
$ sort -R batch/videos_melee.tsv | head -n10 | cut -f1 | python batch.py
```

## If you'd like to help
I'm still working on getting this to a point where others can maintain it. Some of the code is still a mess.
