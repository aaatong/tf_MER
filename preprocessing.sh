#!/bin/bash
#
# This script is written for converting .mp3 files to .wav files, and then converting them to mono
#
# Before running, put the script to the same lavel as your data folder
#


# if test -e temp ; then
#     rm -r temp
# fi

# if test -e output ; then
#     rm -r output
# fi

# mkdir temp output

# cd ./1/    # before running this script, modify the path here

# for song in *.mp3 ; do
#     newname=`echo ${song}|tr -d ' '`
#     mv $song './'$newname
#     outfile='../temp/'${newname/%'.mp3'/'.wav'}
#     mpg123 -w ${outfile} ${newname}
# done

# for wav in `ls ../temp/` ; do
#     infile='../temp/'$wav
#     outfile='../output/'$wav
#     sox $infile $outfile channels 1
# done

# rm -r ../temp/




if test -e temp ; then
    rm -r temp
fi

cd data/source_songs
for class in $(ls); do
    mkdir temp
    cd $class
    for song in *.mp3 ; do
        newname=${song//' '/'_'}
        # newname=`echo ${song}|tr -d ' '`
        mv "$song" "$newname"
        outfile='../temp/'${newname/%'.mp3'/'.wav'}
        mpg123 -w ${outfile} ${newname}
    done

    cd ..
    for wav in `ls temp` ; do
        infile='temp/'$wav
        outfile='../processed_songs/'$class'/'$wav
        if ! test -e ../processed_songs/$class; then
            mkdir ../processed_songs/$class
        fi
        sox $infile $outfile channels 1
    done

    rm -r temp
done
