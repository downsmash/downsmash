# downsmash/segment

Part of the [Downsmash](downsma.sh) project.

Watches a video of a Melee set and determines the (approximate) start and end times of contiguous Melee.

## Usage
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

## Requirements
  - cv2
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - sqlalchemy
  - youtube_dl
