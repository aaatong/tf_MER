#!/bin/bash

# folder='./1/'
# class=1
# if ! test -e clips ; then
#     mkdir clips
# fi
# cd $folder

# count=0
# for song in $(ls) ; do
#     duration=$(soxi -D $song)
#     duration=${duration%.*}
#     # echo $duration
#     clip_num=$(expr $duration / 5)
#     # echo $clip_num 
#     for (( i=0; i<clip_num; i=i+1 )) ; do
        
#         outfile='../clips/'$class'_'$count
#         start=$(expr $i * 5)
#         sox $song $outfile trim $start 5 
#         count=$(expr $count + 1)
#     done
# done


cd data/processed_songs
for class in $(ls); do
    cd $class
    count=0
    for song in $(ls) ; do
        duration=$(soxi -D $song)
        duration=${duration%.*}
        # echo $duration
        clip_num=$(expr $duration / 5)
        # echo $clip_num 
        for (( i=0; i<clip_num; i=i+1 )) ; do
            
            outfile='../../clips/'$class'_'$count'.wav'
            start=$(expr $i \* 5)
            sox $song $outfile trim $start 5 
            count=$(expr $count + 1)
        done
    done
    cd ..
done
