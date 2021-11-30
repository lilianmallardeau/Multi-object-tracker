#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo Provide detectors and trackers names to benchmark
    echo "Usage:    $0 detectors trackers [--no-tracking] [--no-benchmark]"
    echo "Example:  $0 \"yolo ssd\" \"naive-tracker kalman-tracker\""
    exit 1
fi

DETECTORS="$1"
TRACKERS="$2"
EXTRA_OPTIONS="--filter-class person"
METRICS="HOTA CLEAR Identity VACE"
BENCHMARK="MOT17" # MOT15, MO16, MOT17 or MOT20

PERFORM_TRACKING=true
PERFORM_BENCHMARK=true
for arg in "$@"; do
    [ "$arg" = "--no-tracking" ] && PERFORM_TRACKING=false
    [ "$arg" = "--no-benchmark" ] && PERFORM_BENCHMARK=false
done

OUTPUT_FOLDER="output/${BENCHMARK}"


if [ $PERFORM_TRACKING = true ]; then
    [ -d data/ ] || mkdir data
    if [ ! -d data/${BENCHMARK} ]; then
        cd data
        wget https://motchallenge.net/data/${BENCHMARK}.zip
        unzip ${BENCHMARK}.zip
        rm ${BENCHMARK}.zip
        cd ..
    fi
    for detector in ${DETECTORS}; do
        for tracker in ${TRACKERS}; do
            output_folder="${OUTPUT_FOLDER}/${detector}_${tracker}"
            [ -d "${output_folder}" ] || mkdir -p "${output_folder}"
            for example in data/${BENCHMARK}/train/*; do
                echo Performing tracking in ${example}...
                ./main.py track $detector $tracker --glob "${example}/img1/*" --output "$output_folder/$(basename $example).mp4" --export-csv "$output_folder/$(basename $example).txt" ${EXTRA_OPTIONS}
                echo
            done
            wait
        done
    done
fi

if [ $PERFORM_BENCHMARK = true ]; then
    [ -d TrackEval/ ] || git clone https://github.com/lilianmallardeau/TrackEval.git
    if [ ! -d TrackEval/data ]; then
        cd TrackEval
        wget https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
        unzip data.zip
        rm data.zip
        cd ..
    fi
    for detector in ${DETECTORS}; do
        for tracker in ${TRACKERS}; do
            output_folder="${OUTPUT_FOLDER}/${detector}_${tracker}"
            [ -d ${output_folder} ] || (echo \"${output_folder}\" not found; exit 2)
            mkdir -p "TrackEval/data/trackers/mot_challenge/${BENCHMARK}-train/${detector}_${tracker}/data"
            cp ${output_folder}/*.txt "TrackEval/data/trackers/mot_challenge/${BENCHMARK}-train/${detector}_${tracker}/data"
            trackers_to_eval="${trackers_to_eval} ${detector}_${tracker}"
        done
    done
    cd TrackEval
    python3 scripts/run_mot_challenge.py --BENCHMARK ${BENCHMARK} --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL ${trackers_to_eval} --METRICS ${METRICS} --USE_PARALLEL True
    python3 scripts/comparison_plots.py ${trackers_to_eval}
fi