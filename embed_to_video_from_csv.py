#!/usr/bin/env python3
import csv
import argparse
import numpy as np
import cv2

from videosource import VideoFile, GlobFrames

parser = argparse.ArgumentParser()
video_input = parser.add_mutually_exclusive_group()
video_input.add_argument("--input-filename", "-i", dest="input_filename", type=str, help="Input video file")
video_input.add_argument("--glob", type=str, help="Glob expression to select the video frames")
parser.add_argument("csv_file", type=str, help="CSV file with detections/tracks to show on video")
parser.add_argument("out", type=str, help="Output video file")
parser.add_argument("--image-coordinates", dest="image_coordinates", default=False, action='store_true', help="Specify this flag if coordinates in the CSV file are in image coordinates (origin at the bottom left corner). Otherwise, matrix coordinates are used (origin at the top left corner)")
parser.add_argument("--4c-codec", "--fourc-codec", "--codec", "--4cc", default="mp4v", dest="codec", type=str, help="Fourc codec for the output video")
parser.add_argument("--fps", type=float, default=25, help="FPS of the output video, if Glob input is used.")
args = parser.parse_args()


class BBox:
    def __init__(self, obj_id, left, top, width, height, confidence):
        self.obj_id = obj_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.confidence = confidence

frames = dict()
objects_color = dict()
with open(args.csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        frame = int(row[0])
        obj_id = int(row[1])
        left = int(row[2])
        top = int(row[3])
        width = int(row[4])
        height = int(row[5])
        confidence = float(row[6])

        if not frame in frames:
            frames[frame] = []
        if not obj_id in objects_color:
            objects_color[obj_id] = np.uint8(np.random.uniform(0, 255, (3,))).tolist()
        
        frames[frame].append(BBox(obj_id, left, top, width, height, confidence))


if args.glob:
    print("Reading pictures from glob...", end=' ', flush=True)
    video_source = GlobFrames(args.glob, fps=args.fps)
    print(f"Done, {video_source.nbr_frames} frames read")
else:
    video_source = VideoFile(args.input_filename)

width, height = video_source.video_size

codec = cv2.VideoWriter_fourcc(*args.codec)
out = cv2.VideoWriter(args.out, codec, video_source.fps, video_source.video_size)

n_frame = 1
try:
    while video_source.is_opened() and video_source.has_next_frame():
        print(f"\rProcessing frame {n_frame}/{video_source.nbr_frames}...", end='', flush=True)
        frame = video_source.get_next_frame()

        if n_frame in frames:
            for bbox in frames[n_frame]:
                if args.image_coordinates:
                    bbox.top = height - bbox.top
                frame = cv2.rectangle(frame, (bbox.left, bbox.top), (bbox.left + bbox.width, bbox.top + bbox.height), objects_color[bbox.obj_id], 2)
                text = f"{bbox.obj_id}: {bbox.confidence:.2%}"
                frame = cv2.putText(frame, text, (bbox.left, bbox.top), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

        out.write(frame)
        n_frame += 1
except KeyboardInterrupt:
    pass

print()
video_source.close()
out.release()
