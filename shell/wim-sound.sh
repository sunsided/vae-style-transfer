#!/usr/bin/env bash

INPUT=out-Wim\ -\ See\ You\ Hurry.mp4
OUTPUT="~/Downloads/Wim - See You Hurry.mp4"

ffmpeg -i "$INPUT" -i "$OUTPUT" -c copy -map 0:0 -map 1:1 -shortest out-Wim-sound.mp4
