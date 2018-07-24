# meleedb-segment

Working title TBA.

Watches a video of a Melee set and runs a series of increasingly tenuous
statistical whatsits to determine when Melee happens.

The methodology used is fairly brittle and probably will not generalize to
other games or video sources.

In particular, the game window is detected in part by taking a random sample
of frames and plotting skewness times kurtosis, then edge-detecting the
result. This works because Melee's camera is so tight that any given pixel
from the game window will most likely range over a large number of world
coordinates and thus approach a normal distribution, whereas playercam
or overlay coordinates will not. However, many other fighting games have
either a wider and/or deeper field of view (Smash 4); or very homogeneous
backgrounds (Street Fighter, Marvel.) This means that the distribution of
certain parts of the game window will not be statistically distinguishable
from the overlay, at least by this crude methodology.

I'm still very proud of the idea.

Dependencies are:
  - OpenCV ([3.3.2](issues/5))
  - pandas
  - scikit-learn

## Synopsis
```
$ python segment.py [vod]
```
