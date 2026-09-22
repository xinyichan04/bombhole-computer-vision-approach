#!/usr/bin/env python3
"""
ball_tracker.py
---------------
Tracks a ball in GoPro throw clips using background subtraction + Kalman filter.
No training data needed.

Usage:
    python3 src/ball_tracker.py clip.MP4
    python3 src/ball_tracker.py clip.MP4 --show          # display live visualization
    python3 src/ball_tracker.py clip.MP4 --save          # save annotated video
    python3 src/ball_tracker.py clip.MP4 --show --save   # both

Output:
    - Prints detected ball positions per frame
    - Optionally saves annotated video to ./tracked/
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


# ─────────────────────────────────────────────
# Kalman Filter Setup
# State:       [x, y, dx, dy]
# Measurement: [x, y]
# ─────────────────────────────────────────────
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


def find_ball_candidate(fg_mask, min_area=30, max_area=3000, min_circularity=0.3):
    """
    Find the most ball-like contour in the foreground mask.
    Returns (cx, cy, radius) or None.
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Score: prefer more circular and larger blobs
        score = circularity * area
        if score > best_score:
            best_score = score
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            best = (int(x), int(y), int(radius))

    return best


def track_ball(video_path, show=False, save=False, 
               process_width=540, process_height=960):
    """
    Main tracking function.
    
    Args:
        video_path:      Path to input MP4
        show:            Display live window
        save:            Save annotated video
        process_width:   Width to process at (smaller = faster)
        process_height:  Height to process at
    
    Returns:
        List of dicts: {frame, time_sec, x, y, radius, detected}
        x, y are in ORIGINAL video coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale factors to map processed coords → original coords
    scale_x = orig_w / process_width
    scale_y = orig_h / process_height

    print(f"\n  Video:      {Path(video_path).name}")
    print(f"  Original:   {orig_w}x{orig_h} | {fps:.1f}fps | {total} frames")
    print(f"  Processing: {process_width}x{process_height}")

    # Background subtractor
    # history=30: learns background from first ~30 frames
    # varThreshold=20: lower = more sensitive to motion
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=30,
        varThreshold=20,
        detectShadows=False
    )

    kf = create_kalman()
    kalman_initialized = False

    # Morphological kernel for cleaning up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    results = []
    trajectory = []       # list of (x, y) in original coords for drawing
    missed_frames = 0
    MAX_MISSED = 10       # reset Kalman if ball lost for this many frames

    # Video writer setup
    writer = None
    if save:
        os.makedirs("test/results/tracked", exist_ok=True)
        out_name = f"test/results/tracked/{Path(video_path).stem}_tracked.MP4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_name, fourcc, fps, (orig_w, orig_h))
        print(f"  Saving to:  {out_name}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / fps

        # Resize for processing
        small = cv2.resize(frame, (process_width, process_height))

        # Background subtraction
        fg_mask = bg_sub.apply(small)

        # Morphological cleanup: remove noise, fill holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find ball candidate in mask
        candidate = find_ball_candidate(fg_mask)

        detected = False
        pos_orig = None

        if candidate is not None:
            cx, cy, radius = candidate

            # Map back to original resolution
            ox = int(cx * scale_x)
            oy = int(cy * scale_y)
            or_ = int(radius * max(scale_x, scale_y))

            # Update Kalman
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not kalman_initialized:
                kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                kalman_initialized = True

            kf.correct(measurement)
            predicted = kf.predict()

            detected = True
            missed_frames = 0
            pos_orig = (ox, oy, or_)
            trajectory.append((ox, oy))

        else:
            # Ball not found — use Kalman prediction if initialized
            missed_frames += 1
            if kalman_initialized and missed_frames < MAX_MISSED:
                predicted = kf.predict()
                px = int(predicted[0] * scale_x)
                py = int(predicted[1] * scale_y)
                pos_orig = (px, py, 15)  # estimated radius
                trajectory.append((px, py))
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                trajectory = []

        # Store result
        result = {
            "frame":    frame_idx,
            "time_sec": round(time_sec, 3),
            "detected": detected,
            "x":        pos_orig[0] if pos_orig else None,
            "y":        pos_orig[1] if pos_orig else None,
            "radius":   pos_orig[2] if pos_orig else None,
        }
        results.append(result)

        # ── Visualization ──────────────────────────────────────────
        if show or save:
            vis = frame.copy()

            # Draw trajectory trail
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(vis, trajectory[i-1], trajectory[i], color, 2)

            # Draw ball position
            if pos_orig:
                ox, oy, or_ = pos_orig
                if detected:
                    cv2.circle(vis, (ox, oy), or_, (0, 255, 0), 2)       # green = detected
                    cv2.circle(vis, (ox, oy), 4,   (0, 255, 0), -1)
                else:
                    cv2.circle(vis, (ox, oy), or_, (0, 165, 255), 2)     # orange = predicted
                    cv2.circle(vis, (ox, oy), 4,   (0, 165, 255), -1)

            # HUD
            cv2.putText(vis, f"Frame: {frame_idx}  Time: {time_sec:.2f}s",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            status = "DETECTED" if detected else ("PREDICTED" if pos_orig else "LOST")
            color  = (0,255,0)  if detected else ((0,165,255) if pos_orig else (0,0,255))
            cv2.putText(vis, status, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            if save and writer:
                writer.write(vis)

            if show:
                # Downscale for display
                disp = cv2.resize(vis, (540, 960))
                cv2.imshow("Ball Tracker", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # Summary
    detected_count = sum(1 for r in results if r["detected"])
    print(f"\n  Results: {detected_count}/{len(results)} frames with ball detected")
    print(f"  Detection rate: {detected_count/max(len(results),1)*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Track ball in throw clips.")
    parser.add_argument("input",  help="Input video file")
    parser.add_argument("--show", action="store_true", help="Display live visualization")
    parser.add_argument("--save", action="store_true", help="Save annotated video to ./tracked/")
    parser.add_argument("--width",  type=int, default=540,  help="Processing width  (default: 540)")
    parser.add_argument("--height", type=int, default=960,  help="Processing height (default: 960)")
    args = parser.parse_args()

    results = track_ball(
        args.input,
        show=args.show,
        save=args.save,
        process_width=args.width,
        process_height=args.height,
    )

    # Print detections
    print("\n  Frame-by-frame detections:")
    for r in results:
        if r["detected"]:
            print(f"    Frame {r['frame']:3d} ({r['time_sec']:.2f}s): "
                  f"x={r['x']}, y={r['y']}, r={r['radius']}")


if __name__ == "__main__":
    main()