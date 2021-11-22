import glob
import cv2



class VideoSource:
    def close(self):
        pass


class VideoFile(VideoSource):
    def __init__(self, filename: str):
        self.cv_capture = cv2.VideoCapture(filename)
        self.nbr_frames = int(self.cv_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cv_capture.get(cv2.CAP_PROP_FPS)
        self.video_size = (
            int(self.cv_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cv_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self._next_retrieved = False
    
    def has_next_frame(self):
        has_next, self.next_frame = self.cv_capture.read()
        self._next_retrieved = bool(has_next)
        return has_next
    
    def get_next_frame(self):
        if not self._next_retrieved:
            if not self.has_next_frame():
                raise RuntimeError("Video source has no next frame")
        self._next_retrieved = False
        return self.next_frame
    
    def is_opened(self):
        return self.cv_capture.isOpened()
    
    def close(self):
        self.cv_capture.release()


class GlobFrames(VideoSource):
    def __init__(self, glob_string: str, fps: int):
        self.frames = [cv2.imread(img) for img in sorted(glob.glob(glob_string))]
        self.nbr_frames = len(self.frames)
        self.fps = fps
        self.video_size = (
            self.frames[0].shape[1],
            self.frames[0].shape[0],
        )
        self.next_frame = 0
    
    def has_next_frame(self):
        return self.next_frame < self.nbr_frames
    
    def get_next_frame(self):
        self.next_frame += 1
        return self.frames[self.next_frame-1]
    
    def is_opened(self):
        return True


class Camera(VideoFile):
    pass