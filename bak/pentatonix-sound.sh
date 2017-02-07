#!/usr/bin/env bash

INPUT="out-Daft Punk - Pentatonix.mp4"
OUTPUT="~/Downloads/Daft Punk - Pentatonix.mp4"

ffmpeg -i "$INPUT" -i "$OUTPUT" -c copy -map 0:0 -map 1:1 -shortest out-ptx-sound.mp4
