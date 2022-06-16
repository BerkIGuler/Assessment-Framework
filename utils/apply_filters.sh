#!/bin/bash

#below set of commands applies filters to mp4 files in test directory and saves them in test_processed folder with the same names

#function of each filter is given in filters.txt file.

file_count=0

for i in test/*.mp4

do
	if [ $file_count -lt 77 ]
	then
		ffmpeg -i "$i" -vf curves=vintage "test_processed/${i%.*}.mp4"


	elif [ $file_count -lt 154 ]
	then
		ffmpeg -i "$i" -vf curves=darker "test_processed/${i%.*}.mp4"


	elif [ $file_count -lt 231 ]
	then
		ffmpeg -i "$i" -vf curves=increase_contrast "test_processed/${i%.*}.mp4"


	elif [ $file_count -lt 308 ]
	then
		ffmpeg -i "$i" -vf curves=lighter "test_processed/${i%.*}.mp4"

	elif [ $file_count -lt 385 ]
	then
		ffmpeg -i "$i" -vf hue="H=2*PI*t: s=sin(2*PI*t)+1" "test_processed/${i%.*}.mp4"

	elif [ $file_count -lt 462 ]
	then
		ffmpeg -i "$i" -vf noise=alls=20:allf=t+u "test_processed/${i%.*}.mp4"

	elif [ $file_count -lt 539 ]
	then
		ffmpeg -i "$i" -vf colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131 "test_processed/${i%.*}.mp4"


	elif [ $file_count -lt 616 ]
	then
		ffmpeg -i "$i" -vf colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3 "test_processed/${i%.*}.mp4"

	elif [ $file_count -lt 693 ]
	then
		ffmpeg -i "$i" -vf convolution="0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0" "test_processed/${i%.*}.mp4"

	else
		ffmpeg -i "$i" -vf gblur "test_processed/${i%.*}.mp4"
	fi

	let "file_count += 1"

done





