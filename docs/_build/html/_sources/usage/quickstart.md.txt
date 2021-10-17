# Quickstart

Download a video containing one Melee set to `melee_vod.mp4`, then:

```python
from downsmash.watcher import watch
watch('melee_vod.mp4')
```

`watch` will send back a JSON object. Let's use [this set from Low Tide City](https://www.youtube.com/watch?v=ZhhGAI8Fk1c).

```json
{
  "segments": [
    [ 16.7109375, 222.734375 ],
    [ 249.171875, 472.65625 ],
    [ 504.171875, 607.65625 ]
  ],
  "threshold": 0.3982072589975415,
  "view": {
    "ports": [
      {
        "height": 96,
        "left": 102,
        "top": 255,
        "width": 119
      },
      null,
      null,
      {
        "height": 96,
        "left": 403,
        "top": 255,
        "width": 130
      }
    ],
    "scale": 0.7853535353535352,
    "screen": {
      "height": 322.8675645342311,
      "left": 102.61111111111106,
      "top": 32.84343434343428,
      "width": 430.4900860456415
    }
  }
}
```

In this case, `segments` contains the timestamps of the start and end of the three games in the set, and `view` contains the scale and location of the game feed, as well as the ports in use and their locations.
