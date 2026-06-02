"""
Object Detection and Tracking System

Real-time object detection using YOLO and tracking using SORT.
Supports webcam input, video files, and image files.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from detector import YOLODetector, COCO_CLASSES
from tracker import SORTTracker

# Color palette for different tracked objects
COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (255, 128, 0),   # Orange
    (0, 128, 255),   # Light Blue
    (255, 0, 128),   # Pink
    (128, 255, 0),   # Lime
    (0, 255, 128),   # Teal
]


def get_color(track_id):
    """Get a consistent color for a given track ID."""
    return COLORS[track_id % len(COLORS)]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time Object Detection and Tracking'
    )
    parser.add_argument(
        '--source', '-s', type=str, default='0',
        help='Video source: 0 for webcam, or path to video/image file'
    )
    parser.add_argument(
        '--model', '-m', type=str, default='yolo11n.pt',
        help='YOLO model name or path (default: yolo11n.pt)'
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--iou', type=float, default=0.45,
        help='NMS IoU threshold (default: 0.45)'
    )
    parser.add_argument(
        '--track-iou', type=float, default=0.3,
        help='IoU threshold for tracking association (default: 0.3)'
    )
    parser.add_argument(
        '--max-age', type=int, default=30,
        help='Maximum frames to keep a track alive (default: 30)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device for inference: cpu, cuda, mps (default: cpu)'
    )
    parser.add_argument(
        '--classes', type=int, nargs='+', default=None,
        help='Class IDs to detect (e.g., 0 for person). Default: all classes'
    )
    parser.add_argument(
        '--show-fps', action='store_true', default=True,
        help='Show FPS counter (default: True)'
    )
    parser.add_argument(
        '--show-classes', action='store_true', default=True,
        help='Show class labels (default: True)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--resize', type=float, default=1.0,
        help='Resize factor for display (default: 1.0)'
    )
    return parser.parse_args()


def draw_detections(frame, tracked_objects, detector, show_classes=True):
    """
    Draw bounding boxes, labels, and tracking IDs on the frame.

    Args:
        frame: The video frame (numpy array)
        tracked_objects: List of [x1, y1, x2, y2, track_id, class_id, confidence]
        detector: YOLODetector instance (for class names)
        show_classes: Whether to show class labels

    Returns:
        Frame with visualizations drawn
    """
    display = frame.copy()
    h, w = display.shape[:2]

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, class_id, confidence = obj

        # Clamp to frame boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        color = get_color(track_id)

        # Draw bounding box with rounded corners effect (thicker line)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Draw corner accents for a more polished look
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        if corner_len > 5:
            # Top-left
            cv2.line(display, (x1, y1), (x1 + corner_len, y1), color, 2)
            cv2.line(display, (x1, y1), (x1, y1 + corner_len), color, 2)
            # Top-right
            cv2.line(display, (x2, y1), (x2 - corner_len, y1), color, 2)
            cv2.line(display, (x2, y1), (x2, y1 + corner_len), color, 2)
            # Bottom-left
            cv2.line(display, (x1, y2), (x1 + corner_len, y2), color, 2)
            cv2.line(display, (x1, y2), (x1, y2 - corner_len), color, 2)
            # Bottom-right
            cv2.line(display, (x2, y2), (x2 - corner_len, y2), color, 2)
            cv2.line(display, (x2, y2), (x2, y2 - corner_len), color, 2)

        # Build label text
        label_parts = [f'ID: {track_id}']
        if show_classes:
            class_name = detector.get_class_name(class_id)
            label_parts.append(class_name)
        label = ' | '.join(label_parts)

        # Draw label background and text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        label_y = y1 - 10 if y1 > 25 else y2 + 25

        # Background rectangle for label
        cv2.rectangle(
            display,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 10, label_y + 5),
            color,
            -1  # Filled
        )

        # Text color (white or black based on background brightness)
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            display, label,
            (x1 + 5, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, text_color, 2
        )

    return display


def draw_info_panel(frame, fps, num_tracks, frame_count):
    """
    Draw an information overlay on the frame.

    Args:
        frame: The video frame
        fps: Current frames per second
        num_tracks: Number of active tracks
        frame_count: Current frame number
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Info text
    info_text = f'FPS: {fps:.1f}  |  Tracks: {num_tracks}  |  Frame: {frame_count}'
    cv2.putText(
        frame, info_text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    # Legend on the right side
    legend_x = w - 220
    cv2.rectangle(frame, (legend_x, 5), (legend_x + 210, 35), (0, 0, 0), -1)

    cv2.putText(
        frame, 'Controls: [Q]uit  [P]ause  [R]eset',
        (legend_x + 10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (200, 200, 200), 1
    )

    return frame


def main():
    """Main entry point for the object detection and tracking system."""
    args = parse_args()

    print("=" * 60)
    print("   Object Detection and Tracking System")
    print("=" * 60)
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Confidence threshold: {args.conf}")
    print("=" * 60)
    print()

    # Initialize detector
    print("[INFO] Loading YOLO model...")
    try:
        detector = YOLODetector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            classes=args.classes
        )
        print(f"[INFO] Model loaded successfully: {args.model}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Initialize tracker
    tracker = SORTTracker(
        max_age=args.max_age,
        min_hits=3,
        iou_threshold=args.track_iou
    )

    # Open video source
    if args.source == '0':
        source = 0  # Webcam index
        print("[INFO] Opening webcam (press 'Q' to quit)...")
    else:
        source = args.source
        if not os.path.exists(source):
            print(f"[ERROR] Video source not found: {source}")
            sys.exit(1)
        print(f"[INFO] Opening video file: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {source}")
        sys.exit(1)

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Apply resize
    display_width = int(orig_width * args.resize)
    display_height = int(orig_height * args.resize)

    print(f"[INFO] Video: {orig_width}x{orig_height} @ {fps_source:.1f} FPS")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Display: {display_width}x{display_height}")

    # Video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps_source,
                              (display_width, display_height))
        print(f"[INFO] Recording output to: {args.output}")

    # Main loop
    frame_count = 0
    paused = False
    fps_history = []
    fps_update_interval = 15  # Update FPS display every N frames
    start_time = time.time()
    last_frame_time = time.time()

    print("[INFO] Running... (press 'Q' to quit, 'P' to pause, 'R' to reset)")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream.")
                    break

                frame_count += 1

                # Resize frame for display
                if args.resize != 1.0:
                    frame = cv2.resize(frame, (display_width, display_height))

                # Run detection
                detections = detector.detect(frame)

                # Update tracker
                tracked_objects = tracker.update(detections)

                # Draw results
                display = draw_detections(
                    frame, tracked_objects, detector,
                    show_classes=args.show_classes
                )

                # Calculate and display FPS (rolling average)
                now = time.time()
                fps = 1.0 / (now - last_frame_time) if (now - last_frame_time) > 0 else 0
                last_frame_time = now
                fps_history.append(fps)

                if len(fps_history) >= fps_update_interval:
                    avg_fps = sum(fps_history[-fps_update_interval:]) / fps_update_interval
                elif frame_count > 1:
                    avg_fps = sum(fps_history) / len(fps_history)
                else:
                    avg_fps = fps

                display = draw_info_panel(
                    display, avg_fps, len(tracked_objects), frame_count
                )

                # Record output if requested
                if out:
                    out.write(display)

                # Show frame
                cv2.imshow('Object Detection & Tracking', display)

                # Write frame count progress for video files
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    sys.stdout.write(f"\r[PROGRESS] Frame {frame_count}/{total_frames}"
                                     f" ({progress:.1f}%)")
                    sys.stdout.flush()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('p') or key == ord('P'):
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"\n[INFO] {status}")
            elif key == ord('r') or key == ord('R'):
                print("\n[INFO] Resetting tracker...")
                tracker.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    # Summary
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"   Session Summary")
    print(f"{'=' * 60}")
    print(f"   Frames processed: {frame_count}")
    print(f"   Elapsed time: {total_elapsed:.1f}s")
    print(f"   Average FPS: {frame_count / total_elapsed:.1f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
