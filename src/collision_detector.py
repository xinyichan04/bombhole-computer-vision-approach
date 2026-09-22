#!/usr/bin/env python3
"""
collision_detector.py
---------------------
Detects:
  1. Ball collision + spark on PEDAL targets (labeled 20/40/50)
  2. Ball entering HOLE targets (dark circular openings)

Setup: Draw ROIs on the first frame interactively.
  - Phase 1: Draw rectangles around each PEDAL, enter score
  - Phase 2: Draw circles around each HOLE, enter score

Usage:
    python3 src/collision_detector.py clip.MP4 --save --save-config config.json
    python3 src/collision_detector.py clip.MP4 --save --config config.json
    python3 src/collision_detector.py clip.MP4 --save --config config.json --output-dir test/results/scoring

Controls during setup:
    Click + drag  = draw rectangle (pedal phase) or circle (hole phase)
    Enter score   = type in terminal after drawing
    Z             = undo last region
    Q             = done with current phase / move to next
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path


# ─────────────────────────────────────────────
# Kalman Filter
# ─────────────────────────────────────────────
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


def find_ball_candidate(fg_mask, min_area=30, max_area=3000, min_circularity=0.3):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
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
        score = circularity * area
        if score > best_score:
            best_score = score
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            best = (int(x), int(y), int(radius))
    return best


# ─────────────────────────────────────────────
# Spark Detector
# ─────────────────────────────────────────────
def detect_spark(frame, roi, threshold=0.04):
    """Detect yellow/orange spark in pedal ROI (x, y, w, h)."""
    x, y, w, h = roi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,  100, 150])
    upper_yellow = np.array([45,  255, 255])
    lower_white  = np.array([0,   0,   220])
    upper_white  = np.array([180, 40,  255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_white,  upper_white)
    )
    intensity = cv2.countNonZero(mask) / max((x2-x1)*(y2-y1), 1)
    return intensity > threshold, round(intensity, 4)


# ─────────────────────────────────────────────
# Hole Detector
# ─────────────────────────────────────────────
def detect_hole_entry(frame, hole, ball_pos):
    """
    Detect if ball has entered a hole.
    Strategy: ball center is within the hole circle radius.
    hole: {cx, cy, r, score, name} in original coords.
    Returns (in_hole: bool, dist: float)
    """
    if ball_pos is None:
        return False, 0.0

    bx, by, _ = ball_pos
    hcx, hcy, hr = hole["cx"], hole["cy"], hole["r"]

    dist = float(np.sqrt((bx - hcx)**2 + (by - hcy)**2))
    TOLERANCE = 20  # extra pixels beyond drawn radius to account for tracker imprecision
    in_hole = dist <= (hr + TOLERANCE)

    return in_hole, round(dist, 1)


# ─────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────
PEDAL_COLORS = [
    (0,   255, 255),
    (255, 0,   255),
    (0,   165, 255),
    (255, 255, 0  ),
    (128, 0,   255),
    (0,   255, 128),
]
HOLE_COLORS = [
    (0,   80,  255),
    (255, 80,  0  ),
    (80,  255, 80 ),
    (200, 0,   200),
    (0,   200, 200),
    (255, 200, 0  ),
]


# ─────────────────────────────────────────────
# Interactive Setup
# ─────────────────────────────────────────────
_drawing   = False
_start_pt  = None
_cur_pt    = None
_temp_rect = None


def _mouse_cb(event, x, y, flags, param):
    global _drawing, _start_pt, _cur_pt, _temp_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing  = True
        _start_pt = (x, y)
        _cur_pt   = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _cur_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing  = False
        _cur_pt   = (x, y)
        x1 = min(_start_pt[0], _cur_pt[0])
        y1 = min(_start_pt[1], _cur_pt[1])
        x2 = max(_start_pt[0], _cur_pt[0])
        y2 = max(_start_pt[1], _cur_pt[1])
        _temp_rect = (x1, y1, x2 - x1, y2 - y1)


def _reset_mouse():
    global _drawing, _start_pt, _cur_pt, _temp_rect
    _drawing = False
    _start_pt = None
    _cur_pt = None
    _temp_rect = None


def _draw_all(base, pedals, holes, sx, sy, phase):
    vis = base.copy()
    for i, p in enumerate(pedals):
        c = PEDAL_COLORS[i % len(PEDAL_COLORS)]
        dx, dy = int(p["x"]/sx), int(p["y"]/sy)
        dw, dh = int(p["w"]/sx), int(p["h"]/sy)
        cv2.rectangle(vis, (dx, dy), (dx+dw, dy+dh), c, 2)
        cv2.putText(vis, f"P{i+1}:{p['score']}pt", (dx, max(dy-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    for i, h in enumerate(holes):
        c = HOLE_COLORS[i % len(HOLE_COLORS)]
        dcx, dcy = int(h["cx"]/sx), int(h["cy"]/sy)
        dr = int(h["r"] / max(sx, sy))
        cv2.circle(vis, (dcx, dcy), max(dr,1), c, 2)
        cv2.putText(vis, f"H{i+1}:{h['score']}pt", (dcx-20, max(dcy-dr-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    label = "PHASE 1: Draw PEDALS (rectangles)" if phase == "pedal" else "PHASE 2: Draw HOLES (circles)"
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return vis


def setup_regions(video_path, disp_w=540, disp_h=960):
    global _temp_rect

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame")

    orig_h, orig_w = first_frame.shape[:2]
    sx = orig_w / disp_w
    sy = orig_h / disp_h
    display = cv2.resize(first_frame, (disp_w, disp_h))

    win = "Setup — Pedals & Holes"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse_cb)

    pedals, holes = [], []

    # ── Phase 1: Pedals ──
    print("\n┌──────────────────────────────────────────────┐")
    print("│  PHASE 1 — PEDALS                            │")
    print("│  Click + drag rectangles around each pedal  │")
    print("│  Type score in terminal after each draw      │")
    print("│  Z = undo last   |   Q = done with pedals   │")
    print("└──────────────────────────────────────────────┘\n")
    _reset_mouse()

    while True:
        vis = _draw_all(display, pedals, holes, sx, sy, "pedal")
        if _drawing and _start_pt and _cur_pt:
            c = PEDAL_COLORS[len(pedals) % len(PEDAL_COLORS)]
            cv2.rectangle(vis, _start_pt, _cur_pt, c, 2)
        elif not _drawing and _temp_rect:
            c = PEDAL_COLORS[len(pedals) % len(PEDAL_COLORS)]
            x, y, w, h = _temp_rect
            cv2.rectangle(vis, (x, y), (x+w, y+h), c, 2)
            cv2.putText(vis, "check terminal", (x, max(y-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(vis, f"Pedals:{len(pedals)}  Z=undo  Q=next",
                    (10, disp_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if _temp_rect and not _drawing:
            x, y, w, h = _temp_rect
            if w > 5 and h > 5:
                ox, oy = int(x*sx), int(y*sy)
                ow, oh = int(w*sx), int(h*sy)
                print(f"  Pedal {len(pedals)+1} — Enter score: ", end="", flush=True)
                try:
                    score = int(input().strip())
                except ValueError:
                    score = 1
                pedals.append({"name": f"Pedal_{len(pedals)+1}",
                                "x": ox, "y": oy, "w": ow, "h": oh, "score": score})
                print(f"  ✓ Pedal_{len(pedals)} score:{score} ROI:({ox},{oy},{ow},{oh})")
                _temp_rect = None
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("z"), ord("Z")) and pedals:
            print(f"  Undid {pedals.pop()['name']}")

    # ── Phase 2: Holes ──
    print("\n┌──────────────────────────────────────────────┐")
    print("│  PHASE 2 — HOLES                             │")
    print("│  Click + drag a circle over each hole        │")
    print("│  (drag from edge to edge of the hole)        │")
    print("│  Type score in terminal after each draw      │")
    print("│  Z = undo last   |   Q = finish setup       │")
    print("└──────────────────────────────────────────────┘\n")
    _reset_mouse()

    while True:
        vis = _draw_all(display, pedals, holes, sx, sy, "hole")
        if _drawing and _start_pt and _cur_pt:
            c = HOLE_COLORS[len(holes) % len(HOLE_COLORS)]
            cx_d = (_start_pt[0] + _cur_pt[0]) // 2
            cy_d = (_start_pt[1] + _cur_pt[1]) // 2
            r_d  = int(np.sqrt((_cur_pt[0]-_start_pt[0])**2 +
                               (_cur_pt[1]-_start_pt[1])**2) / 2)
            cv2.circle(vis, (cx_d, cy_d), max(r_d, 1), c, 2)
        elif not _drawing and _temp_rect:
            c = HOLE_COLORS[len(holes) % len(HOLE_COLORS)]
            x, y, w, h = _temp_rect
            cx_d = x + w//2
            cy_d = y + h//2
            r_d  = int(np.sqrt(w**2 + h**2) / 2)
            cv2.circle(vis, (cx_d, cy_d), max(r_d, 1), c, 2)
            cv2.putText(vis, "check terminal", (cx_d-40, max(cy_d-r_d-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(vis, f"Holes:{len(holes)}  Z=undo  Q=finish",
                    (10, disp_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if _temp_rect and not _drawing:
            x, y, w, h = _temp_rect
            if w > 5 and h > 5:
                cx_d = x + w//2
                cy_d = y + h//2
                r_d  = int(np.sqrt(w**2 + h**2) / 2)
                ocx  = int(cx_d * sx)
                ocy  = int(cy_d * sy)
                or_  = int(r_d  * max(sx, sy))
                print(f"  Hole {len(holes)+1} — Enter score: ", end="", flush=True)
                try:
                    score = int(input().strip())
                except ValueError:
                    score = 1
                holes.append({"name": f"Hole_{len(holes)+1}",
                               "cx": ocx, "cy": ocy, "r": or_, "score": score})
                print(f"  ✓ Hole_{len(holes)} score:{score} center:({ocx},{ocy}) r={or_}")
                _temp_rect = None
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("z"), ord("Z")) and holes:
            print(f"  Undid {holes.pop()['name']}")

    cv2.destroyAllWindows()
    print(f"\n  Setup complete: {len(pedals)} pedal(s), {len(holes)} hole(s)")
    return pedals, holes


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run_pipeline(video_path, pedals, holes, save=False,
                 process_w=540, process_h=960,
                 spark_confirm_frames=3,
                 hole_confirm_frames=1,
                 output_dir="test/results/tracked"):

    cap = cv2.VideoCapture(video_path)
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    sx = orig_w / process_w
    sy = orig_h / process_h

    print(f"\n  Running: {Path(video_path).name}  ({orig_w}x{orig_h} @ {fps:.1f}fps)")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames = 0
    MAX_MISSED = 10
    trajectory = []

    pedal_state = [{
        "spark_streak":     0,
        "collision_logged": False,
        "collision_frame":  -1,
        "spark_logged":     False,
        "total_score":      0,
    } for _ in pedals]

    hole_state = [{
        "entry_streak":   0,
        "entry_logged":   False,
        "total_score":    0,
        "was_in_hole":    False,   # was ball inside on last frame
        "throw_count":    0,       # how many times logged for this hole
        "last_entry_frame": -999,  # frame of last confirmed entry (for cooldown)
    } for _ in holes]

    events      = []
    total_score = 0

    writer = None
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"{output_dir}/{Path(video_path).stem}_scored.MP4"
        writer   = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"avc1"),
                                   fps, (orig_w, orig_h))
        print(f"  Saving to: {out_name}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        # ── Ball Tracking ──────────────────────────────
        small   = cv2.resize(frame, (process_w, process_h))
        fg_mask = bg_sub.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidate = find_ball_candidate(fg_mask)
        ball_pos  = None
        detected  = False

        if candidate:
            cx, cy, r = candidate
            ox, oy = int(cx*sx), int(cy*sy)
            or_    = int(r * max(sx, sy))
            meas   = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not kalman_initialized:
                kf.statePre = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_initialized = True
            kf.correct(meas)
            kf.predict()
            detected      = True
            missed_frames = 0
            ball_pos      = (ox, oy, or_)
            trajectory.append((ox, oy))
        else:
            missed_frames += 1
            if kalman_initialized and missed_frames < MAX_MISSED:
                pred = kf.predict()
                px = int(float(pred[0]) * sx)
                py = int(float(pred[1]) * sy)
                ball_pos = (px, py, 15)
                trajectory.append((px, py))
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                trajectory = []
                for s in pedal_state:
                    s["collision_logged"] = False
                    s["spark_logged"]     = False
                # Reset holes so next throw can score again
                for s in hole_state:
                    s["entry_logged"] = False
                    s["entry_streak"] = 0
                    s["was_in_hole"]  = False

        frame_events = []

        # ── Pedal Detection ────────────────────────────
        for i, (p, state) in enumerate(zip(pedals, pedal_state)):
            roi = (p["x"], p["y"], p["w"], p["h"])
            ball_in_roi = False
            if ball_pos:
                bx, by, _ = ball_pos
                ball_in_roi = (p["x"] <= bx <= p["x"]+p["w"] and
                               p["y"] <= by <= p["y"]+p["h"])

            if ball_in_roi and not state["collision_logged"]:
                state["collision_logged"] = True
                state["collision_frame"]  = frame_idx
                ev = {"frame": frame_idx, "time_sec": round(time_sec,3),
                      "target": p["name"], "event": "BALL_COLLISION", "score": 0}
                frame_events.append(ev); events.append(ev)
                print(f"  🎯 {time_sec:.2f}s  BALL HIT {p['name']}")

            if state["collision_logged"] and not state["spark_logged"]:
                spark_ok, _ = detect_spark(frame, roi)
                state["spark_streak"] = state["spark_streak"]+1 if spark_ok else 0
                if state["spark_streak"] >= spark_confirm_frames:
                    state["spark_logged"]  = True
                    state["total_score"]  += p["score"]
                    total_score           += p["score"]
                    ev = {"frame": frame_idx, "time_sec": round(time_sec,3),
                          "target": p["name"], "event": "SPARK_CONFIRMED", "score": p["score"]}
                    frame_events.append(ev); events.append(ev)
                    print(f"  ✨ {time_sec:.2f}s  SPARK {p['name']} +{p['score']}pts (total:{total_score})")
            else:
                if not state["collision_logged"]:
                    state["spark_streak"] = 0

        # ── Hole Detection ─────────────────────────────
        # Logic: ball center enters hole circle for >= 2 consecutive frames.
        # entry_logged resets when ball is lost (missed_frames >= MAX_MISSED),
        # so a second throw into the same hole will score again.
        for i, (h, state) in enumerate(zip(holes, hole_state)):
            in_hole, dist = detect_hole_entry(frame, h, ball_pos)

            if in_hole:
                state["entry_streak"] += 1
            else:
                # Reset streak AND entry_logged when ball leaves hole region
                # so a second throw can be detected
                if not in_hole and state["was_in_hole"]:
                    state["entry_logged"] = False
                state["entry_streak"] = 0

            state["was_in_hole"] = in_hole

            # Confirm: ball inside for 2 consecutive frames, not already logged this entry
            COOLDOWN_FRAMES = 30  # min frames between two hole entries (~1 second)
            cooldown_ok = (frame_idx - state["last_entry_frame"]) > COOLDOWN_FRAMES
            if state["entry_streak"] >= hole_confirm_frames and not state["entry_logged"] and cooldown_ok:
                state["entry_logged"]  = True
                state["last_entry_frame"] = frame_idx
                state["throw_count"]  += 1
                state["total_score"]  += h["score"]
                total_score           += h["score"]
                ev = {"frame": frame_idx, "time_sec": round(time_sec,3),
                      "target": h["name"], "event": "HOLE_ENTRY", "score": h["score"]}
                frame_events.append(ev); events.append(ev)
                print(f"  🕳️  {time_sec:.2f}s  HOLE ENTRY {h['name']} +{h['score']}pts (total:{total_score})")

        # ── Visualization ──────────────────────────────
        if save and writer:
            vis = frame.copy()

            # Trajectory
            for j in range(1, len(trajectory)):
                alpha = j / len(trajectory)
                cv2.line(vis, trajectory[j-1], trajectory[j],
                         (0, int(255*alpha), int(255*(1-alpha))), 2)

            # Ball
            if ball_pos:
                bx, by, br = ball_pos
                col = (0,255,0) if detected else (0,165,255)
                cv2.circle(vis, (bx, by), max(br, 12), col, 3)
                cv2.circle(vis, (bx, by), 5, col, -1)
                cv2.putText(vis, "DETECTED" if detected else "PREDICTED",
                            (bx+16, by), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

            # Pedal boxes
            for i, (p, state) in enumerate(zip(pedals, pedal_state)):
                color = PEDAL_COLORS[i % len(PEDAL_COLORS)]
                x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                if state["spark_streak"] > 0:
                    ov = vis.copy()
                    cv2.rectangle(ov, (x,y), (x+w,y+h), (255,255,255), -1)
                    cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,255,255), 4)
                elif state["collision_logged"] and not state["spark_logged"]:
                    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
                else:
                    cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
                lbl = f"{p['name']} [{p['score']}pt]"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis, (x, y-th-10), (x+tw+6, y), (0,0,0), -1)
                cv2.putText(vis, lbl, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if state["spark_logged"]:
                    cv2.putText(vis, f"SPARK +{p['score']}", (x, y+h+28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
                elif state["collision_logged"]:
                    cv2.putText(vis, "COLLISION!", (x, y+h+28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,200,255), 2)

            # Hole circles
            for i, (h, state) in enumerate(zip(holes, hole_state)):
                color = HOLE_COLORS[i % len(HOLE_COLORS)]
                hcx, hcy, hr = h["cx"], h["cy"], h["r"]
                thickness = 4 if state["entry_streak"] > 0 else 2
                cv2.circle(vis, (hcx, hcy), hr, color, thickness)
                lbl = f"{h['name']} [{h['score']}pt]"
                cv2.putText(vis, lbl, (hcx-30, hcy-hr-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if state["entry_logged"]:
                    cv2.putText(vis, f"HOLE IN +{h['score']}", (hcx-30, hcy+hr+28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100,255,100), 2)

            # Top HUD bar
            cv2.rectangle(vis, (0,0), (orig_w, 130), (0,0,0), -1)
            cv2.putText(vis, f"SCORE: {total_score}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0,255,255), 5)
            ts = f"{time_sec:.2f}s | Frame {frame_idx}"
            (tw,_),_ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(vis, ts, (orig_w-tw-30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180,180,180), 2)
            bs  = "TRACKING" if detected else ("PREDICTING" if ball_pos else "LOST")
            bsc = (0,255,0) if detected else ((0,165,255) if ball_pos else (0,0,255))
            cv2.putText(vis, bs, (30, 122), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bsc, 2)

            # Bottom event banner
            if frame_events:
                ev = frame_events[-1]
                if ev["event"] == "SPARK_CONFIRMED":
                    banner, bcol = f"+{ev['score']}pts  {ev['target']} SPARK!", (0,255,255)
                elif ev["event"] == "HOLE_ENTRY":
                    banner, bcol = f"+{ev['score']}pts  {ev['target']} HOLE IN!", (100,255,100)
                else:
                    banner, bcol = f"COLLISION  {ev['target']}", (0,200,255)
                cv2.rectangle(vis, (0, orig_h-80), (orig_w, orig_h), (0,0,0), -1)
                (tw,_),_ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                cv2.putText(vis, banner, ((orig_w-tw)//2, orig_h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, bcol, 3)

            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    return events, total_score


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ball collision + spark + hole detector.")
    parser.add_argument("input",         help="Input video clip")
    parser.add_argument("--save",        action="store_true", help="Save annotated video")
    parser.add_argument("--config",      default=None,  help="Load saved config JSON")
    parser.add_argument("--save-config", default=None,  help="Save config to JSON")
    parser.add_argument("--output-dir",  default="test/results/tracked",
                        help="Output folder (default: test/results/tracked)")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
        pedals = cfg.get("pedals", [])
        holes  = cfg.get("holes",  [])
        print(f"  Loaded config: {len(pedals)} pedals, {len(holes)} holes")
        for p in pedals:
            print(f"    {p['name']}: score={p['score']}  ROI=({p['x']},{p['y']},{p['w']},{p['h']})")
        for h in holes:
            print(f"    {h['name']}: score={h['score']}  center=({h['cx']},{h['cy']}) r={h['r']}")
    else:
        pedals, holes = setup_regions(args.input)
        if args.save_config:
            with open(args.save_config, "w") as f:
                json.dump({"pedals": pedals, "holes": holes}, f, indent=2)
            print(f"\n  Config saved to {args.save_config}")

    events, total_score = run_pipeline(
        args.input, pedals, holes,
        save=args.save, output_dir=args.output_dir
    )

    print(f"\n{'─'*50}")
    print(f"  FINAL SCORE: {total_score} pts")
    print(f"  Events ({len(events)}):")
    icons = {"BALL_COLLISION": "🎯", "SPARK_CONFIRMED": "✨", "HOLE_ENTRY": "🕳️ "}
    for ev in events:
        print(f"    {icons.get(ev['event'],'•')} {ev['time_sec']:.2f}s — "
              f"{ev['target']} — {ev['event']} (+{ev['score']})")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()