# meleedb-segment

meleedb-segment watches a video of a Melee set and outputs the (approximate) start and end times of contiguous Melee.

## Synopsis
Segment an individual VOD:
```
$ python segment.py [vod.mp4] [output.json]
```
Batch-process a set of VODs:
```
$ echo [youtubeID] | python batch.py
$ sort -R batch/videos_melee.tsv | head -n10 | cut -f1 | python batch.py
```
