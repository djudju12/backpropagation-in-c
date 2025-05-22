#!/bin/bash

echo "creating spectograms...."

count=$(find .. -type f -name "*.mp3" | wc -l)
total=0
for audio in $(find .. -type f -name "*.mp3")
do
    bname="${audio##*/}"
    out="${bname%.mp3}.png"
    ffmpeg -hide_banner -loglevel error -i $audio -filter_complex showspectrumpic=s=128x128:legend=disabled -y $out
    ((total++))
    if (( total % 100 == 0 )); then
        echo "$total/$count"
    fi
done
