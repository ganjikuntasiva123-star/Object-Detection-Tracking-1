"""
Object detector wrapper using Ultralytics YOLO.

Supports YOLOv8, YOLOv9, YOLOv10, YOLO11 models.
Automatically downloads the model on first use.
"""

import numpy as np
from ultralytics import YOLO

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLODetector:
    """
    Wrapper around Ultralytics YOLO for object detection.

    Handles model loading, inference, and result parsing.
    Supports optional inference on a subset of classes.
    """

    def __init__(self, model_name='yolo11n.pt', conf_threshold=0.5,
                 iou_threshold=0.45, device='cpu', classes=None):
        """
        Initialize the YOLO detector.

        Args:
            model_name: YOLO model name or path (e.g., 'yolo11n.pt', 'yolov8s.pt')
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
            device: Device to run inference on ('cpu', 'cuda', 'mps', etc.)
            classes: List of class IDs to filter (None = all classes)
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.filter_classes = classes
        self.class_names = COCO_CLASSES

    def detect(self, image):
        """
        Run detection on an image frame.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV default)

        Returns:
            List of detections: [x1, y1, x2, y2, confidence, class_id]
            where coordinates are in pixel space of the input image
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            class_id = class_ids[i]

            # Skip classes not in filter list, if filtering is active
            if self.filter_classes is not None and class_id not in self.filter_classes:
                continue

            detection = [
                float(boxes[i][0]),  # x1
                float(boxes[i][1]),  # y1
                float(boxes[i][2]),  # x2
                float(boxes[i][3]),  # y2
                float(confidences[i]),  # confidence
                int(class_id)  # class_id
            ]
            detections.append(detection)

        return detections

    def get_class_name(self, class_id):
        """Get the class name for a given class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f'class_{class_id}'
