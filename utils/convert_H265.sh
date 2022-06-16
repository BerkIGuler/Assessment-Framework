#!/bin/bash

#below set of commands encodes mp4 files in test_pro directory in h.265 and saves them in h265_test_pro folder with the same names

for i in test_pro/*.mp4; do ffmpeg -i "$i" -c:v libx265 -vtag hvc1 "h265_test_pro/${i%.*}.mp4"; done












