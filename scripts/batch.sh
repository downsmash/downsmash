#!/bin/bash

mkdir -p /tmp/downsmash
for id in $(sort -R downsmash/batch/videos_melee.tsv | head -n$1 | cut -f1); do
  VOD="/tmp/downsmash/$id.mp4"
  yt-dlp -f 134 "$id" -o "$VOD"
  python -c "import downsmash.watcher; print(downsmash.watcher.watch(\"$VOD\"))"
  rm $VOD
done
