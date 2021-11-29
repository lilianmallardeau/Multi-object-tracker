#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo Provide detector and tracker names to use for benchmark
    echo Use: $0 detector tracker [--no-tracking] [--no-benchmark]
    exit 1
fi

DETECTOR="$1"
TRACKER="$2"
EXTRA_OPTIONS="--filter-class person"
METRICS="HOTA CLEAR Identity VACE"
BENCHMARK="MOT17" # MOT15, MO16, MOT17 or MOT20

PERFORM_TRACKING=true
PERFORM_BENCHMARK=true
for arg in "$@"; do
    [ $arg = "--no-tracking" ] && PERFORM_TRACKING=false
    [ $arg = "--no-benchmark" ] && PERFORM_BENCHMARK=false
done

OUTPUT_FOLDER="output/${BENCHMARK}/${DETECTOR}_${TRACKER}"


if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

if [ $PERFORM_TRACKING = true ]; then
    [ -d data/ ] || mkdir data
    if [ ! -d data/${BENCHMARK} ]; then
        cd data
        wget https://motchallenge.net/data/${BENCHMARK}.zip
        unzip ${BENCHMARK}.zip
        rm ${BENCHMARK}.zip
        cd ..
    fi
    for example in data/${BENCHMARK}/train/*; do
        echo Performing tracking in ${example}...
        ./main.py track $DETECTOR $TRACKER --glob "${example}/img1/*" --output "$OUTPUT_FOLDER/$(basename $example).mp4" --export-csv "$OUTPUT_FOLDER/$(basename $example).txt" ${EXTRA_OPTIONS}
        echo
    done
    wait
fi

if [ $PERFORM_BENCHMARK = true ]; then
    [ -d TrackEval/ ] || git clone https://github.com/JonathonLuiten/TrackEval.git
    if [ ! -d TrackEval/data ]; then
        cd TrackEval
        wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
        unzip data.zip
        rm data.zip
        cd ..
    fi
    mkdir -p "TrackEval/data/trackers/mot_challenge/${BENCHMARK}-train/${DETECTOR}_${TRACKER}/data"
    cp ${OUTPUT_FOLDER}/*.txt "TrackEval/data/trackers/mot_challenge/${BENCHMARK}-train/${DETECTOR}_${TRACKER}/data"
    cd TrackEval
    python3 scripts/run_mot_challenge.py --BENCHMARK ${BENCHMARK} --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL "${DETECTOR}_${TRACKER}" --METRICS ${METRICS} --USE_PARALLEL True
fi