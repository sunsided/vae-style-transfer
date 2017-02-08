#!/usr/bin/env bash

INPUT="out-Disclosure - Magnets ft. Lorde.mp4"
OUTPUT="~/Downloads/Disclosure - Magnets ft. Lorde.mp4"

ffmpeg -i "$INPUT" -i "$OUTPUT" -c copy -map 0:0 -map 1:1 -shortest out-disclosure-sound.mp4
