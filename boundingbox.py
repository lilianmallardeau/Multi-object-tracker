import math


class Point():
    def __init__(self, x, y=None):
        if y is None:
            if type(x) not in (tuple, list):
                raise TypeError('`Point` takes either two x, y arguments or one (x, y) argument as a tuple')
            x, y = x[:2]
        self.x = int(x)
        self.y = int(y)
    
    def as_tuple(self):
        return (self.x, self.y)
    
    def dist(self, p):
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"


def ensure_bbox_type(func):
    def wrapper(self, bbox):
        if type(bbox) is not BBox:
            raise TypeError("`bbox` should be a BBox object")
        return func(self, bbox)
    return wrapper


class BBox():
    def __init__(self, pos: Point, size: tuple, confidence: float, class_id: int=None, label: str=None, object_id: int=None):
        """
        Arguments:
            pos [Point]: top left corner of the bounding box
            size [tuple]: (width, height) tuple of the size of the bounding box
            class_id [int]: class id of detected object
            label [str]: optional, label of the detected object
            object_id [int]: optional, object id of the corresponding tracked object
        """
        self.pos = pos if type(pos) is Point else Point(pos)
        self.size = (int(size[0]), int(size[1]))
        self.confidence = confidence
        self.class_id = class_id
        self.label = label
        self.object_id = object_id
    
    @property
    def p1(self):
        """ Top left corner of the bounding box """
        return self.pos
    
    @property
    def p2(self):
        """ Bottom right corner of the bounding box """
        return Point(
            self.pos.x + self.size[0],
            self.pos.y + self.size[1]
        )
    
    @property
    def center(self):
        """ Center of the bounding box """
        return Point(
            self.pos.x + self.size[0]/2,
            self.pos.y + self.size[1]/2,
        )
    
    @property
    def width(self):
        """ Width of the bounding box """
        return self.size[0]

    @property
    def height(self):
        """ Height of the bounding box """
        return self.size[1]
    
    @property
    def area(self):
        return self.width * self.height
    
    @property
    def bbox(self):
        """ Bounding box as list as [top_left.x, top_left.y, width, height] """
        p1 = self.p1
        return [p1.x, p1.y, self.width, self.height]

    @ensure_bbox_type
    def dist(self, bbox):
        return self.center.dist(bbox.center)
    
    @ensure_bbox_type
    def intersect(self, bbox):
        return max((self.width+bbox.width)/2 - abs(self.center.x-bbox.center.x), 0) * max((self.height+bbox.height)/2 - abs(self.center.y-bbox.center.y), 0)
    
    @ensure_bbox_type
    def union(self, bbox):
        return self.area + bbox.area - self.intersect(bbox)
    
    @ensure_bbox_type
    def IoU(self, bbox):
        return self.intersect(bbox) / self.union(bbox)
    
    def __repr__(self):
        return f"BBox(center={self.center}, size={self.size})"