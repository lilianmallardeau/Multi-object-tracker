import uuid
import numpy as np
from boundingbox import BBox

from motpy import Detection, MultiObjectTracker


class TrackedObject():
    def __init__(self, id: int, bbox: BBox, frame: int, uuid=uuid.uuid4(), color=None):
        self.uuid = uuid
        self.id = id
        self.label = bbox.label
        self.color = color if color else np.uint8(np.random.uniform(0, 255, (3,)))
        self.first_frame = frame
        self.last_frame = frame

        bbox.object_id = id
        self._bboxes = [bbox]
        self._frames = [frame]
    
    @property
    def last_bbox(self):
        return self._bboxes[-1]
    
    def add_bbox(self, bbox: BBox, frame: int):
        bbox.object_id = self.id
        if bbox.label: self.label = bbox.label
        self._bboxes.append(bbox)
        self._frames.append(frame)
        self.last_frame = frame

    def repr(self):
        return f"{self.label} {self.id}: {self.last_bbox.confidence:.2%}"
    
    def to_csv(self, height):
        return "\n".join([f"{frame},{self.id},{','.join(str(p) for p in bbox.as_list_img_coord(height))},{bbox.confidence},-1,-1,-1" for frame, bbox in zip(self._frames, self._bboxes)])



class ObjectTracker():
    def __init__(self):
        self.tracked_objects = []
        self._last_detections = None
        self._frame_number = 0
        self._last_object_id = 0
    
    def get_new_id(self):
        self._last_object_id += 1
        return self._last_object_id

    @property
    def active_objects(self):
        return [obj for obj in self.tracked_objects if obj.last_frame == self._frame_number]
    
    @property
    def tracked_objects_by_id(self):
        return {obj.id: obj for obj in self.tracked_objects}
    
    @property
    def tracked_objects_by_uuid(self):
        return {obj.uuid: obj for obj in self.tracked_objects}

    def track(self, detections: list):
        self._frame_number += 1
        self._track(detections)
        self._last_detections = detections
        return self.active_objects

    def _track(self, detections: list):
        raise NotImplementedError("You must override the `_track` method in your child tracker class")
    
    def to_csv(self, height):
        return "\n".join([obj.to_csv(height) for obj in self.tracked_objects])



class NaiveObjectTracker(ObjectTracker):
    def __init__(self, distance_threshold=np.inf):
        super(NaiveObjectTracker, self).__init__()
        self.distance_threshold = distance_threshold
    
    def _track(self, detections: list):
        if self._last_detections is None:
            self.tracked_objects = [TrackedObject(self.get_new_id(), detection, self._frame_number) for detection in detections]
        
        else:
            labels = {detection.label for detection in detections}
            objects_by_class = {label: [d for d in detections if d.label == label] for label in labels}
            
            for label, new_detections in objects_by_class.items():
                previous_detections = [d for d in self._last_detections if d.label == label]

                bbox_distances = np.full((len(new_detections), len(previous_detections)), np.inf)
                for i, new_detection in enumerate(new_detections):
                    for j, previous_detection in enumerate(previous_detections):
                        bbox_distances[i, j] = new_detection.dist(previous_detection)
                
                while np.any(bbox_distances < self.distance_threshold):
                    argmin = np.unravel_index(np.argmin(bbox_distances), bbox_distances.shape)
                    bbox_distances = np.delete(bbox_distances, argmin[0], 0)
                    bbox_distances = np.delete(bbox_distances, argmin[1], 1)
                    self.tracked_objects_by_id[previous_detections.pop(argmin[1]).object_id].add_bbox(new_detections.pop(argmin[0]), self._frame_number)
                for new_bbox in new_detections:
                    self.tracked_objects.append(TrackedObject(self.get_new_id(), new_bbox, self._frame_number))


class KalmanObjectTracker(ObjectTracker):
    def __init__(self, class_labels: list):
        super(KalmanObjectTracker, self).__init__()
        self._tracker = MultiObjectTracker(dt=0.1)
        self._class_labels = class_labels

    def _track(self, detections):
        self._tracker.step(detections=[Detection(box=[b.p1.x, b.p1.y, b.p2.x, b.p2.y], score=b.confidence, class_id=b.class_id) for b in detections])
        tracks = self._tracker.active_tracks()

        for track in tracks:
            xmin, ymin, xmax, ymax = track.box
            bbox = BBox(
                pos=(xmin, ymin),
                size=(np.abs(xmax-xmin), np.abs(ymax-ymin)),
                confidence=track.score,
                class_id=track.class_id,
                label=self._class_labels[track.class_id]
            )
            if track.id in self.tracked_objects_by_uuid:
                self.tracked_objects_by_uuid[track.id].add_bbox(bbox, self._frame_number)
            else:
                self.tracked_objects.append(TrackedObject(self.get_new_id(), bbox, self._frame_number, uuid=track.id))