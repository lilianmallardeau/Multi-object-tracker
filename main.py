#!/usr/bin/env python3
import argparse
import os
import numpy as np
import cv2

from videosource import VideoFile, GlobFrames, Camera
from boundingbox import BBox
from detector import YoloDetector, SSDDetector
from tracker import NaiveObjectTracker


# Arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["detect", "track"], help="Action to perform: detection [detect] or tracking [track]", type=str)
parser.add_argument("detector", type=str, help="Detector to use", choices=['yolo', 'yolotiny', 'ssd'])
parser.add_argument("tracker", type=str, help="Tracker to use", choices=['naive-tracker'])

video_input = parser.add_mutually_exclusive_group()
video_input.add_argument("--input-filename", "-i", dest="input_filename", type=str, help="Input file")
video_input.add_argument("--glob", type=str, help="Glob expression to select the video frames")
video_input.add_argument("--camera", type=int, help="Index of the camera to use")

parser.add_argument("--gpu", "--cuda", default=True, dest="gpu", help="Enable CUDA", action="store_true")
parser.add_argument("--show-perf", default=True, dest="show_perf", help="Show real time performance in the output video", action="store_true")
#parser.add_argument("--print-perf", default=False, dest="print_perf", help="Print real time performance in the console", action="store_true")
parser.add_argument("--conf-threshold", default=0.5, dest="conf_threshold", type=float, help="Confidence threshold for object detection")
parser.add_argument("--nms-threshold", default=0.4, dest="nms_threshold", type=float, help="Non maximum suppression threshold")

output = parser.add_mutually_exclusive_group()
output.add_argument("--output", "-o", default=None, dest="output_filename", type=str, help="Output file. An output filename is generated according to the chosen parameters if none is provided.")
output.add_argument("--no-save", default=True, dest="save_output", help="Don't save the output video to a file", action="store_false")
parser.add_argument("--4c-codec", "--fourc-codec", "--codec", "--4cc", default="mp4v", dest="codec", type=str, help="Fourc codec for the output video")
parser.add_argument("--show-output", default=False, dest="show_output", help="Show the real time output with OpenCV", action="store_true")
parser.add_argument("--export-csv", default=False, dest="export_csv", help="Export the tracking to a CSV file", type=str)
args = parser.parse_args()

if args.output_filename == None:
    action = "detection" if args.action == "detect" else "tracking"
    tracker = f"_{args.tracker}" if args.action == "track" else ""
    if args.input_filename: basename = os.path.splitext(os.path.basename(args.input_filename))
    elif args.glob: basename = "glob"
    elif args.camera: basename = f"camera{args.camera}"
    else: basename = ""
    args.output_filename = f"output/{action}_{args.detector}{tracker}_{basename}.mp4"
    try: os.makedirs("output/", exist_ok=True)
    except: exit('A file named "output" already exists')


# --------- Model parameters ---------
DETECTION_THRESHOLD = args.conf_threshold
NMS_THRESHOLD = args.nms_threshold

# Yolo
yolo_config = "detectors/yolov4/yolov4.cfg"
yolo_weights = "detectors/yolov4/yolov4.weights"
yolo_tiny_config = "detectors/yolov4-tiny/yolov4-tiny.cfg"
yolo_tiny_weights = "detectors/yolov4-tiny/yolov4-tiny.weights"
labels_file_yolo = "detectors/yolov4/coco.names"

# SSD
ssd_proto = "detectors/ssd/SSD_MobileNet_prototxt.txt"
ssd_weigths = "detectors/ssd/SSD_MobileNet.caffemodel"
labels_file_ssd = "detectors/ssd/ssd.names"


# --------- Loading detector ---------
if args.detector == 'yolo':
    detector = YoloDetector(
        yolo_config,
        yolo_weights,
        (416, 416),
        labels_file_yolo,
        DETECTION_THRESHOLD,
        NMS_THRESHOLD
    )
if args.detector == 'yolotiny':
    detector = YoloDetector(
        yolo_tiny_config,
        yolo_tiny_weights,
        (416, 416),
        labels_file_yolo,
        DETECTION_THRESHOLD,
        NMS_THRESHOLD
    )
if args.detector == 'ssd':
    detector = SSDDetector(
        ssd_proto,
        ssd_weigths,
        (600, 600),
        labels_file_ssd,
        DETECTION_THRESHOLD,
        NMS_THRESHOLD
    )

# GPU/CPU
if args.gpu:
    print("Enabling CUDA...", end=' ')
    detector.enableCuda()
    print("Done")

colors = np.uint8(np.random.uniform(0, 255, (detector.n_classes, 3)))


# --------- Loading tracker ---------
if args.tracker == 'naive-tracker':
    tracker = NaiveObjectTracker()


# -------------- Input --------------
if args.glob:
    print("Reading pictures from glob...", end=' ')
    video_source = GlobFrames(args.glob, fps=25)
    print(f"Done, {video_source.nbr_frames} frames read")
elif args.camera is not None:
    video_source = Camera(args.camera)
else:
    video_source = VideoFile(args.input_filename)


# -------------- Output --------------
if args.save_output:
    out_codec = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output_filename, out_codec, int(video_source.fps), video_source.video_size)


# ------------- Processing ------------
n_frame = 1
try:
    while video_source.is_opened() and video_source.has_next_frame():
        frame = video_source.get_next_frame()
        print(f"[{n_frame/video_source.nbr_frames:.0%}] Processing frame {n_frame} over {video_source.nbr_frames}...", end=' ')
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with the detector to get the bounding box predictions
        boxes = detector.detect(frame)
        print(f"{len(boxes)} bounding boxes detected")

        # --- Drawing --- #
        frame_annotated = frame  # Doesn't actually copy the frame but nvm

        # Drawing detections bounding boxes, if enabled
        if args.action == "detect":
            for box in boxes:
                frame_annotated = cv2.rectangle(frame_annotated, box.p1.as_tuple(), box.p2.as_tuple(), colors[box.class_id].tolist(), 2)
                frame_annotated = cv2.putText(frame_annotated, f"{box.label} {box.confidence*100:.2f}%", box.pos.as_tuple(), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,)*3)

        # Drawing the tracked objects on the frame
        if args.action == "track":
            objects = tracker.track(boxes)
            for obj in objects:
                frame_annotated = cv2.rectangle(frame_annotated, obj.last_bbox.p1.as_tuple(), obj.last_bbox.p2.as_tuple(), obj.color.tolist(), 2)
                frame_annotated = cv2.putText(frame_annotated, obj.repr(), obj.last_bbox.pos.as_tuple(), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,)*3)

        # Printing performance info in frame
        if args.show_perf:
            t, _ = detector.net.getPerfProfile()
            text = f"Inference time: {(t*1000 / cv2.getTickFrequency()):.2f} ms"
            fps = cv2.getTickFrequency() / t
            frame_annotated = cv2.putText(frame_annotated, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))
            frame_annotated = cv2.putText(frame_annotated, f"fps: {fps:.4f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))

        # Saving the frame in the output file
        if args.save_output:
            out.write(frame_annotated)

        # Showing the frame with cv2.imshow
        if args.show_output:
            cv2.imshow("Detection", frame_annotated)
            cv2.waitKey(1)
        
        n_frame += 1
except KeyboardInterrupt:
    pass

video_source.close()
if args.save_output:
    out.release()

if args.export_csv and args.action == "track":
    with open(args.export_csv, 'w') as f:
        f.write(tracker.to_csv())

cv2.destroyAllWindows()