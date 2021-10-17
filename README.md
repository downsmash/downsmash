<div align="center">
<img alt="Downsmash" src="https://raw.githubusercontent.com/downsmash/downsmash/master/docs/downsmash.png" />
<br />
<img alt="What Downsmash does" src="https://raw.githubusercontent.com/downsmash/downsmash/master/docs/what_it_does.png" />
<br />
<b>Downsmash turns Melee into data.</b>
</div>

Downsmash is a parser for _Super Smash Bros. Melee_. It extracts data from raw video feed.

# Installation

This should work on Windows, since `opencv-python` packages its own OpenCV. I cannot guarantee this, though, and I cannot help Windows users with configuration.

```sh
$ git clone https://github.com/downsmash/downsmash
$ pip install -r requirements.txt
```

You will also probably want to use a video downloader such as `yt-dlp`.

# Usage
```python
from downsmash.watcher import watch
watch('melee_vod.mp4')
```

# Contributing
~Downsmash provides a flexible plugin framework that I haven't written yet.~
