"""
SORT (Simple Online and Realtime Tracking) implementation.

Tracks objects across video frames using:
- Kalman Filter for state estimation
- Hungarian Algorithm for data association
- IoU (Intersection over Union) as the association metric
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes.
    State: [x, y, s, r, x', y', s']
    - x, y: center coordinates
    - s: scale (area)
    - r: aspect ratio (width/height) - assumed constant
    - x', y', s': velocities
    """
    count = 0

    def __init__(self, bbox, class_id=0):
        """
        Initialize tracker with a detection bounding box.

        Args:
            bbox: [x1, y1, x2, y2] in top-left bottom-right format
            class_id: Object class ID from detection
        """
        self.class_id = class_id

        # Convert to [cx, cy, s, r] where s=area, r=aspect ratio
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h  # scale is area
        r = w / h if h > 0 else 1.0  # aspect ratio

        # State transition matrix (constant velocity model)
        self.kf = np.eye(7, 7)
        # dt = 1 (frame-to-frame)
        dt = 1.0
        self.kf[0, 4] = dt
        self.kf[1, 5] = dt
        self.kf[2, 6] = dt

        # Measurement matrix: we observe [x, y, s, r]
        self.kf_measure = np.zeros((4, 7))
        self.kf_measure[0, 0] = 1.0
        self.kf_measure[1, 1] = 1.0
        self.kf_measure[2, 2] = 1.0
        self.kf_measure[3, 3] = 1.0

        # Process noise covariance (Q)
        self.kf_q = np.eye(7) * 0.01
        self.kf_q[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance (R)
        self.kf_r = np.eye(4) * 0.1

        # State estimate covariance (P)
        self.kf_p = np.eye(7) * 10.0
        self.kf_p[4:, 4:] *= 100.0  # Higher uncertainty for velocities

        # Initial state
        self.kf_x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)

        self.time_since_update = 0
        self.hits = 1  # Number of times this track has been matched
        self.hit_streak = 1  # Consecutive matches
        self.age = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Store history for visualization
        self.history = deque(maxlen=30)

    def update(self, bbox):
        """
        Update the tracker with a matched detection.

        Args:
            bbox: [x1, y1, x2, y2] in top-left bottom-right format
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Convert bbox to measurement [cx, cy, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = w / h if h > 0 else 1.0
        z = np.array([cx, cy, s, r], dtype=np.float32)

        # Kalman update
        # y = z - H * x
        y = z - self.kf_measure @ self.kf_x
        # S = H * P * H^T + R
        S = self.kf_measure @ self.kf_p @ self.kf_measure.T + self.kf_r
        # K = P * H^T * S^-1
        K = self.kf_p @ self.kf_measure.T @ np.linalg.inv(S)
        # x = x + K * y
        self.kf_x = self.kf_x + K @ y
        # P = (I - K*H) * P
        self.kf_p = (np.eye(7) - K @ self.kf_measure) @ self.kf_p

    def predict(self):
        """
        Advance the state forward by one time step.

        Returns:
            [x1, y1, x2, y2] predicted bounding box
        """
        # x = F * x (constant velocity prediction)
        self.kf_x = self.kf @ self.kf_x
        # P = F * P * F^T + Q
        self.kf_p = self.kf @ self.kf_p @ self.kf.T + self.kf_q

        self.age += 1
        self.time_since_update += 1
        self.history.append(self.get_state())

        return self.get_state()

    def get_state(self):
        """
        Get the current bounding box estimate.

        Returns:
            [x1, y1, x2, y2] bounding box
        """
        cx = self.kf_x[0]
        cy = self.kf_x[1]
        s = self.kf_x[2]
        r = self.kf_x[3]

        # Ensure valid values
        if s < 0:
            s = 0
        if r < 0:
            r = 0.1

        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_confidence(self):
        """Return a confidence score based on hit streak and recency."""
        if self.time_since_update > 0:
            return 0.0
        return min(1.0, self.hit_streak / 10.0)


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking.

    Manages multiple KalmanBoxTracker instances and handles
    data association between detections and existing tracks.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the SORT tracker.

        Args:
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IoU for a valid match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

    def reset(self):
        """Reset the tracker, clearing all tracks."""
        self.tracks = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_id] or
                       numpy array of shape (N, 6)

        Returns:
            List of tracked objects: [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        self.frame_count += 1

        # Convert to list if numpy array
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()

        # No detections — just predict existing tracks
        if len(detections) == 0:
            results = []
            for track in self.tracks:
                track.predict()
                if track.time_since_update <= self.max_age:
                    state = track.get_state()
                    results.append([
                        state[0], state[1], state[2], state[3],
                        track.id, 0,  # class_id unknown
                        track.get_confidence()
                    ])
            # Remove stale tracks
            self.tracks = [t for t in self.tracks
                           if t.time_since_update <= self.max_age]
            return results

        # Predict new locations for all existing tracks
        for track in self.tracks:
            track.predict()

        # Extract detection boxes [x1, y1, x2, y2]
        det_boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        det_confs = np.array([d[4] if len(d) > 4 else 1.0 for d in detections])
        det_classes = np.array([int(d[5]) if len(d) > 5 else 0 for d in detections])

        # Initialize association variables
        matched_track_indices = []
        matched_det_indices = []
        unmatched_track_indices = []
        unmatched_det_indices = []

        # Extract predicted boxes from tracks
        if len(self.tracks) > 0:
            trk_boxes = np.array([t.get_state() for t in self.tracks])

            # Compute IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, trk_box in enumerate(trk_boxes):
                for j, det_box in enumerate(det_boxes):
                    iou_matrix[i, j] = compute_iou(trk_box, det_box)

            # Hungarian algorithm for optimal assignment
            # Minimize cost = 1 - IoU
            cost_matrix = 1.0 - iou_matrix

            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(track_indices, det_indices):
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    matched_track_indices.append(t_idx)
                    matched_det_indices.append(d_idx)
                else:
                    unmatched_track_indices.append(t_idx)
                    unmatched_det_indices.append(d_idx)

            # Tracks not part of any assignment
            all_track_indices = set(range(len(self.tracks)))
            all_det_indices = set(range(len(detections)))
            matched_track_set = set(matched_track_indices)
            matched_det_set = set(matched_det_indices)
            unmatched_track_indices.extend(
                list(all_track_indices - matched_track_set)
            )
            unmatched_det_indices.extend(
                list(all_det_indices - matched_det_set)
            )
        else:
            # No existing tracks — all detections are new
            unmatched_det_indices = list(range(len(detections)))

        # Update matched tracks with detections
        for t_idx, d_idx in zip(matched_track_indices, matched_det_indices):
            self.tracks[t_idx].update(det_boxes[d_idx])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_det_indices:
            bbox = det_boxes[d_idx]
            new_track = KalmanBoxTracker(bbox, class_id=int(det_classes[d_idx]))
            self.tracks.append(new_track)

        # Remove stale tracks
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # Build output list
        results = []
        for track in self.tracks:
            if track.time_since_update == 0:  # Only include tracks updated this frame
                state = track.get_state()
                results.append([
                    state[0], state[1], state[2], state[3],
                    track.id,
                    track.class_id,
                    track.get_confidence()
                ])

        return results
