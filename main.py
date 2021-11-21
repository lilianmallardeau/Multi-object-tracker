#!/usr/bin/env python3
import numpy as np
import cv2

from boundingbox import BBox
from detector import YoloDetector, SSDDetector
from tracker import NaiveObjectTracker


# ------------ PARAMETERS ------------
USE_GPU = True
SHOW_OUTPUT = False
SAVE_OUTPUT = True
PERF_INFO_IN_VIDEO = True
SHOW_DETECTIONS = False
SHOW_TRACKED_OBJECTS = True
INPUT_FILENAME = "/cluster/home/lilianma/test_pictures/highway_10sec.mp4" # 0 for webcam
OUTPUT_FILENAME = "output/highway_10sec.avi"
OUTPUT_4C_CODEC = 'MJPG'
DETECTION_THRESHOLD = 0.0
NMS_THRESHOLD = 0.4

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

# ------------------------------------

yolo_detector = YoloDetector(
    yolo_config,
    yolo_weights,
    (416, 416),
    labels_file_yolo,
    DETECTION_THRESHOLD,
    NMS_THRESHOLD
)
yolotiny_detector = YoloDetector(
    yolo_tiny_config,
    yolo_tiny_weights,
    (416, 416),
    labels_file_yolo,
    DETECTION_THRESHOLD,
    NMS_THRESHOLD
)
ssd_detector = SSDDetector(
    ssd_proto,
    ssd_weigths,
    (300, 300),
    labels_file_ssd,
    DETECTION_THRESHOLD,
    NMS_THRESHOLD
)
detector = ssd_detector
tracker = NaiveObjectTracker()


# GPU/CPU
if USE_GPU:
    detector.enableCuda()

colors = np.uint8(np.random.uniform(0, 255, (detector.n_classes, 3)))

video_source = cv2.VideoCapture(INPUT_FILENAME)
source_nbr_frames = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))
source_fps = video_source.get(cv2.CAP_PROP_FPS)
source_size = (
    int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

if SAVE_OUTPUT:
    out_codec = cv2.VideoWriter_fourcc(*OUTPUT_4C_CODEC)
    out = cv2.VideoWriter(OUTPUT_FILENAME, out_codec, int(source_fps), source_size)

n_frame = 1
try:
    while video_source.isOpened():
        has_frame, frame = video_source.read()
        if not has_frame:
            break
        print(f"[{n_frame/source_nbr_frames:.0%}] Processing frame {n_frame} over {source_nbr_frames}...", end=' ')
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with the detector to get the bounding box predictions
        boxes = detector.detect(frame)
        print(f"{len(boxes)} bounding boxes detected")
        objects = tracker.track(boxes)

        # --- Drawing --- #
        frame_annotated = frame  # Doesn't actually copy the frame but nvm

        # Drawing detections bounding boxes, if enabled
        if SHOW_DETECTIONS:
            for box in boxes:
                frame_annotated = cv2.rectangle(frame_annotated, box.p1.as_tuple(), box.p2.as_tuple(), colors[box.class_id].tolist(), 2)
                frame_annotated = cv2.putText(frame_annotated, f"{box.label} {box.confidence*100:.2f}%", box.pos.as_tuple(), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,)*3)

        # Drawing the tracked objects on the frame
        if SHOW_TRACKED_OBJECTS:
            for obj in objects:
                frame_annotated = cv2.rectangle(frame_annotated, obj.last_bbox.p1.as_tuple(), obj.last_bbox.p2.as_tuple(), obj.color.tolist(), 2)
                frame_annotated = cv2.putText(frame_annotated, obj.repr(), obj.last_bbox.pos.as_tuple(), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,)*3)

        # Printing performance info in frame
        if PERF_INFO_IN_VIDEO:
            t, _ = detector.net.getPerfProfile()
            text = f"Inference time: {(t*1000 / cv2.getTickFrequency()):.2f} ms"
            fps = cv2.getTickFrequency() / t
            frame_annotated = cv2.putText(frame_annotated, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))
            frame_annotated = cv2.putText(frame_annotated, f"fps: {fps:.4f}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))

        # Saving the frame in the output file
        if SAVE_OUTPUT:
            out.write(frame_annotated)

        # Showing the frame with cv2.imshow
        if SHOW_OUTPUT:
            cv2.imshow("Detection", frame_annotated)
            cv2.waitKey(1)
        
        n_frame += 1
except KeyboardInterrupt:
    pass

video_source.release()
if SAVE_OUTPUT:
    out.release()

cv2.destroyAllWindows()