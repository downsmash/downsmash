# meleedb-segment

meleedb-segment watches a video of a Melee set and outputs the (approximate) start and end times of contiguous Melee.

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
Good things come to those who wait.
