import numpy as np
from boundingbox import BBox


class TrackedObject():
    def __init__(self, id, bbox: BBox, frame: int, color=None):
        self.id = id
        self.label = bbox.label
        self.color = color if color else np.uint8(np.random.uniform(0, 255, (3,)))
        self.first_frame = frame
        self.last_frame = frame
        self.lost = False

        bbox.object_id = id
        self._bboxes = [bbox]
        self._frames = [frame]
    
    @property
    def last_bbox(self):
        return self._bboxes[-1]
    
    def add_bbox(self, bbox: BBox, frame: int):
        bbox.object_id = self.id
        self._bboxes.append(bbox)
        self._frames.append(frame)
        self.last_frame = frame

    def repr(self):
        return f"{self.label} {self.id}: {self.last_bbox.confidence:.2%}"
    
    def to_csv(self, height):
        return "\n".join([f"{frame},{self.id},{','.join(str(p) for p in bbox.as_list_img_coord(height))},{bbox.confidence},-1,-1,-1" for frame, bbox in zip(self._frames, self._bboxes)])



class ObjectTracker():
    def track(self, detections: list):
        raise NotImplementedError("You must override the `track` method in the child tracker class")
    
    def to_csv(self, height):
        return "\n".join([obj.to_csv(height) for obj in self.tracked_objects])


class NaiveObjectTracker(ObjectTracker):
    def __init__(self, distance_threshold=np.inf):
        self.distance_threshold = distance_threshold
        self.tracked_objects = []
        self.last_detections = None
        self.frame_number = 0

        self._last_object_id = -1
    
    def get_new_id(self):
        self._last_object_id += 1
        return self._last_object_id

    @property
    def currently_tracked_objects(self):
        return [obj for obj in self.tracked_objects if not obj.lost]
    
    @property
    def tracked_objects_as_dict(self):
        return {obj.id: obj for obj in self.tracked_objects}
        
    def track(self, detections: list):
        self.frame_number += 1

        if self.last_detections is None:
            self.tracked_objects = [TrackedObject(self.get_new_id(), detection, self.frame_number) for detection in detections]
        
        else:
            labels = {detection.label for detection in detections}
            objects_by_class = {label: [d for d in detections if d.label == label] for label in labels}
            
            for label, new_detections in objects_by_class.items():
                previous_detections = [d for d in self.last_detections if d.label == label]

                bbox_distances = np.full((len(new_detections), len(previous_detections)), np.inf)
                for i, new_detection in enumerate(new_detections):
                    for j, previous_detection in enumerate(previous_detections):
                        bbox_distances[i, j] = new_detection.dist(previous_detection)
                
                while np.any(bbox_distances < self.distance_threshold):
                    argmin = np.unravel_index(np.argmin(bbox_distances), bbox_distances.shape)
                    bbox_distances = np.delete(bbox_distances, argmin[0], 0)
                    bbox_distances = np.delete(bbox_distances, argmin[1], 1)
                    self.tracked_objects[previous_detections.pop(argmin[1]).object_id].add_bbox(new_detections.pop(argmin[0]), self.frame_number)
                for new_bbox in new_detections:
                    self.tracked_objects.append(TrackedObject(self.get_new_id(), new_bbox, self.frame_number))
                for lost_bbox in previous_detections:
                    self.tracked_objects[lost_bbox.object_id].lost = True
        
        for obj in self.tracked_objects:
            if obj.last_frame != self.frame_number:
                obj.lost = True

        self.last_detections = detections
        return self.currently_tracked_objects