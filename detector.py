import time
import numpy as np
import cv2

from boundingbox import Point, BBox


class AbstractDetector():
    def __init__(self, names_file: str, conf_threshold:float = 0.5, nms_threshold:float = 0.4):
        self.labels = open(names_file).read().lstrip('\n').rstrip('\n').split('\n')
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def detect(frame, perform_nms=True):
        raise NotImplementedError("You must override the `detect` method in the child detector class")
    
    def enableCuda(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def disableCuda(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def n_classes(self):
        return len(self.labels)
    
    def perform_nms(self, boxes):
        indices = cv2.dnn.NMSBoxes(
            [b.bbox for b in boxes],
            [b.confidence for b in boxes],
            self.conf_threshold,
            self.nms_threshold
        )
        return [boxes[i] for i in indices]


class YoloDetector(AbstractDetector):
    def __init__(self, cfg_file: str, weights_file: str, input_size: tuple, names_file: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        super(YoloDetector, self).__init__(names_file, conf_threshold, nms_threshold)
        self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.input_size = input_size[:2]
    
    def detect(self, frame, perform_nms=True):
        blob = cv2.dnn.blobFromImage(frame, 1/255, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        # t = time.time()
        detections = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # print(time.time()-t)
        boxes = []
        for detection in np.concatenate(detections):
            center_x, center_y, w, h = detection[:4] * np.array((frame.shape[1], frame.shape[0])*2)
            #score = detection[4]
            class_probabilities = detection[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]
            if confidence > self.conf_threshold:
                boxes.append(BBox(
                    Point(center_x - w/2, center_y - h/2),
                    (w, h),
                    confidence,
                    class_id=class_id,
                    label=self.labels[class_id]
                ))
        return self.perform_nms(boxes) if perform_nms else boxes