#!/usr/bin/env python3
"""
pipeline_debug.py  (v3)
------------------------
Mirrors confusion_matrix_v3.py exactly and saves 8-panel debug PNGs.

v3 fixes reflected in debug output:
  Fix 1: threshold 0.04->0.02, full-frame spark suppression
  Fix 2: collision_logged preserved on ball loss
  Fix 3: Kalman frozen while collision pending spark confirmation

Panel layout:
  Row 1: raw | resized+ROIs | MOG2 raw | morph
  Row 2: detect | kalman | ROI check | spark confirm

Panel changes from v1:
  Panel 7 (kalman): now shows spark_active + freeze status
  Panel 8 (ROI check): now shows collision_logged + freeze state per pedal
  Panel 9 (spark confirm): now shows spark intensity + streak per pedal

Usage:
    python3 src/debug/debugPipeline.py --config config.json --clip path/to/clip.MP4
    python3 src/debug/debugPipeline.py --config config.json --clip path/to/clip.MP4 --events-only
    python3 src/debug/debugPipeline.py --config config.json --clip path/to/clip.MP4 --every-n 3
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path


# ================================================================
#  Constants — must match confusion_matrix_v3.py
# ================================================================

SPARK_SUPPRESS_THRESHOLD = 0.02   # v3: was 0.04
SPARK_CONFIRM_FRAMES_DEFAULT = 1
COLLISION_FREEZE_FRAMES = 15
MAX_MISSED = 10


# ================================================================
#  Pipeline helpers — identical to v3
# ================================================================

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
    all_candidates = []
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
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        all_candidates.append((int(x), int(y), int(radius), round(circularity, 3), round(area)))
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(radius))
    return best, all_candidates


def detect_spark(frame, roi, threshold=0.02):   # v3: was 0.04
    x, y, w, h = roi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False, 0.0, None
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
    vis = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    vis[y1:y2, x1:x2] = mask
    return intensity > threshold, round(intensity, 4), vis


def detect_hole_entry(frame, hole, ball_pos):
    if ball_pos is None:
        return False, 0.0
    bx, by, _ = ball_pos
    dist = float(np.sqrt((bx - hole["cx"])**2 + (by - hole["cy"])**2))
    return dist <= (hole["r"] + 20), round(dist, 1)


def global_spark_intensity(frame):
    """Full frame — v3 fix (was bottom-50% only)."""
    region = frame
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,  100, 150])
    upper_yellow = np.array([45,  255, 255])
    lower_white  = np.array([0,   0,   220])
    upper_white  = np.array([180, 40,  255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_white,  upper_white)
    )
    return cv2.countNonZero(mask) / max(region.shape[0] * region.shape[1], 1)


# ================================================================
#  Visualization helpers
# ================================================================

PEDAL_COLORS = [
    (0,255,255),(255,0,255),(0,165,255),
    (255,255,0),(128,0,255),(0,255,128),
]
HOLE_COLORS = [
    (0,80,255),(255,80,0),(80,255,80),(200,0,200),
]

PANEL_W, PANEL_H = 270, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX


def txt(img, text, pos, scale=0.38, color=(255,255,255), thick=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thick+1)
    cv2.putText(img, text, pos, FONT, scale, color, thick)


def make_panel(src, label, w=PANEL_W, h=PANEL_H):
    if len(src.shape) == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(src, (w, h))
    txt(panel, label, (4, 14), scale=0.4, color=(200,255,200))
    return panel


def draw_pedals_holes(panel, pedals, pedal_state, holes, hole_state,
                      orig_w, orig_h, w=PANEL_W, h=PANEL_H):
    sx = w / orig_w
    sy = h / orig_h
    for i, (p, st) in enumerate(zip(pedals, pedal_state)):
        col = PEDAL_COLORS[i % len(PEDAL_COLORS)]
        x1 = int(p["x"]*sx); y1 = int(p["y"]*sy)
        x2 = int((p["x"]+p["w"])*sx); y2 = int((p["y"]+p["h"])*sy)
        thick = 3 if st["collision_logged"] else 1
        cv2.rectangle(panel, (x1,y1), (x2,y2), col, thick)
        tag = ""
        if st["spark_logged"]:        tag = "SPARK"
        elif st["collision_logged"]:  tag = "HIT"
        if tag:
            txt(panel, tag, (x1, y2-4), color=(0,255,255))
        txt(panel, p["name"], (x1, max(y1-4,10)), color=col)
    for i, (h_cfg, st) in enumerate(zip(holes, hole_state)):
        col = HOLE_COLORS[i % len(HOLE_COLORS)]
        cx = int(h_cfg["cx"]*sx); cy = int(h_cfg["cy"]*sy)
        r  = int((h_cfg["r"]+20)*min(sx,sy))
        cv2.circle(panel, (cx,cy), r, col, 1)
        if st["entry_logged"]:
            txt(panel, "IN!", (cx-10, cy), color=(100,255,100))


# ================================================================
#  8-panel frame saver — v3 layout
# ================================================================

def save_step_frame(
    frame_idx, fps,
    raw_frame, small_frame,
    fg_raw, fg_morph,
    all_candidates,
    ball_pos, ball_source, kalman_initialized,
    pedals, pedal_state, holes, hole_state,
    spark_intensities,
    spark_active, global_spark_val,
    any_collision_pending,
    out_dir, clip_label, event_tag,
):
    orig_h, orig_w = raw_frame.shape[:2]
    proc_h, proc_w = small_frame.shape[:2]
    sx = orig_w / proc_w

    # ── Panel 1: raw frame ────────────────────────────────────
    p1 = make_panel(raw_frame, "raw  F%d %.2fs" % (frame_idx, frame_idx/fps))

    # ── Panel 3: resized + ROIs ───────────────────────────────
    p3 = make_panel(small_frame, "resize %dx%d" % (proc_w, proc_h))
    draw_pedals_holes(p3, pedals, pedal_state, holes, hole_state,
                      proc_w, proc_h, PANEL_W, PANEL_H)

    # ── Panel 4: MOG2 raw ─────────────────────────────────────
    p4 = make_panel(fg_raw, "MOG2 raw")
    gsi_col = (0,200,255) if spark_active else (180,180,180)
    txt(p4, "spark_active=%s (%.3f)" % (spark_active, global_spark_val),
        (4, PANEL_H-8), color=gsi_col)

    # ── Panel 5: morph ────────────────────────────────────────
    p5 = make_panel(fg_morph, "morph (open+close)")

    # ── Panel 6: detect candidates ────────────────────────────
    p6 = make_panel(fg_morph, "detect candidates")
    for item in (all_candidates or []):
        bx, by, br, circ, area = item
        cv2.circle(p6, (bx, by), max(br,4), (0,200,255), 1)
        txt(p6, "c=%.2f" % circ, (bx+3, by), scale=0.33, color=(0,200,255))
    if all_candidates:
        wx, wy, wr = all_candidates[0][0], all_candidates[0][1], all_candidates[0][2]
        cv2.circle(p6, (wx, wy), max(wr,4), (0,255,0), 2)
        txt(p6, "BEST", (wx+3, wy-6), color=(0,255,0))
    else:
        freeze_reason = ""
        if spark_active:          freeze_reason = "(spark_active)"
        elif any_collision_pending: freeze_reason = "(collision freeze)"
        txt(p6, "NO CANDIDATE %s" % freeze_reason,
            (4, PANEL_H//2), color=(0,0,255))

    # ── Panel 7: kalman — shows v3 freeze status ──────────────
    p7 = make_panel(raw_frame, "kalman  src=%s" % ball_source)
    draw_pedals_holes(p7, pedals, pedal_state, holes, hole_state,
                      orig_w, orig_h, PANEL_W, PANEL_H)
    dsx = PANEL_W / orig_w
    dsy = PANEL_H / orig_h
    if ball_pos:
        bx = int(ball_pos[0]*dsx); by = int(ball_pos[1]*dsy)
        br = max(int(ball_pos[2]*min(dsx,dsy)), 6)
        if ball_source == "detected":    col = (0,255,0)
        elif ball_source == "predicted": col = (0,165,255)
        else:                            col = (0,0,200)
        cv2.circle(p7, (bx,by), br, col, 2)
        cv2.circle(p7, (bx,by), 3, col, -1)
        txt(p7, ball_source, (bx+5, by), color=col)
    # v3 status line
    status_parts = []
    if spark_active:            status_parts.append("SPARK_SUPPRESS")
    if any_collision_pending:   status_parts.append("FREEZE")
    status_col = (0,200,255) if status_parts else (180,180,180)
    txt(p7, " | ".join(status_parts) if status_parts else "tracking",
        (4, PANEL_H-8), color=status_col)

    # ── Panel 8: ROI check — v3 collision state ───────────────
    p8 = make_panel(raw_frame, "ROI check (v3)")
    draw_pedals_holes(p8, pedals, pedal_state, holes, hole_state,
                      orig_w, orig_h, PANEL_W, PANEL_H)
    if ball_pos:
        bx = int(ball_pos[0]*dsx); by = int(ball_pos[1]*dsy)
        br = max(int(ball_pos[2]*min(dsx,dsy)), 6)
        cv2.circle(p8, (bx,by), br, (0,255,0), 2)
        # highlight active ROI
        for i, (p, st) in enumerate(zip(pedals, pedal_state)):
            x1=int(p["x"]*dsx); y1=int(p["y"]*dsy)
            x2=int((p["x"]+p["w"])*dsx); y2=int((p["y"]+p["h"])*dsy)
            in_roi = (x1 <= bx <= x2 and y1 <= by <= y2)
            if in_roi and not st["collision_logged"]:
                cv2.rectangle(p8, (x1,y1),(x2,y2),(0,255,0),3)
                txt(p8, "IN ROI", (x1,y1-2), color=(0,255,0))
    y_off = 28
    for i, (p, st) in enumerate(zip(pedals, pedal_state)):
        col = (0,255,255) if st["collision_logged"] else (180,180,180)
        flags = []
        if st["collision_logged"]: flags.append("HIT")
        if st["spark_logged"]:     flags.append("SPARK")
        txt(p8, "%s: %s" % (p["name"], ",".join(flags) if flags else "-"),
            (4, y_off), color=col)
        y_off += 14

    # ── Panel 9: spark confirm ────────────────────────────────
    spark_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for p_cfg in pedals:
        roi = (p_cfg["x"], p_cfg["y"], p_cfg["w"], p_cfg["h"])
        _, _, smask = detect_spark(raw_frame, roi)
        if smask is not None:
            spark_mask_combined = cv2.bitwise_or(spark_mask_combined, smask)
    spark_overlay = raw_frame.copy()
    spark_overlay[spark_mask_combined > 0] = [0, 200, 255]
    p9 = make_panel(spark_overlay, "spark confirm (v3 thr=0.02)")
    draw_pedals_holes(p9, pedals, pedal_state, holes, hole_state,
                      orig_w, orig_h, PANEL_W, PANEL_H)
    y_off = 28
    for i, p_cfg in enumerate(pedals):
        val    = spark_intensities.get(p_cfg["name"], 0.0)
        streak = pedal_state[i]["spark_streak"]
        col_hit = pedal_state[i]["collision_logged"]
        col = (0,255,255) if val > 0.02 else (180,180,180)
        active_marker = ">" if col_hit else " "
        txt(p9, "%s%s: s=%.3f str=%d" % (active_marker, p_cfg["name"], val, streak),
            (4, y_off), color=col)
        y_off += 16

    # ── Assemble grid ─────────────────────────────────────────
    row1 = np.hstack([p1, p3, p4, p5])
    row2 = np.hstack([p6, p7, p8, p9])
    grid = np.vstack([row1, row2])

    # Header bar
    hbar = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    parts = []
    for i, p_cfg in enumerate(pedals):
        val = spark_intensities.get(p_cfg["name"], 0.0)
        st  = pedal_state[i]
        flag = "*" if st["spark_logged"] else ("H" if st["collision_logged"] else "-")
        parts.append("%s:%s%.3f/s%d" % (p_cfg["name"], flag, val, st["spark_streak"]))
    freeze_tag = " [FREEZE]" if any_collision_pending else ""
    spark_tag  = " [SPARK_SUP]" if spark_active else ""
    txt(hbar,
        "F%04d [%s]%s%s  %s" % (
            frame_idx, event_tag, freeze_tag, spark_tag, "  ".join(parts)),
        (6, 18), scale=0.38, color=(255,255,200))
    grid = np.vstack([hbar, grid])

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, "%s_F%04d_%s.png" % (clip_label, frame_idx, event_tag))
    cv2.imwrite(fname, grid)
    return fname


# ================================================================
#  Main pipeline loop — v3 logic
# ================================================================

def run_pipeline_debug(video_path, pedals, holes,
                       process_w=540, process_h=960,
                       spark_confirm_frames=1,
                       hole_confirm_frames=1,
                       out_dir="./step_debug",
                       every_n_frames=1,
                       save_all=True,
                       silent=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  ERROR: cannot open %s" % video_path)
        return "miss"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_label = Path(video_path).stem
    sx = orig_w / process_w
    sy = orig_h / process_h

    if not silent:
        print("  Clip   : %s" % video_path)
        print("  Size   : %dx%d  FPS=%.1f  frames~%d" % (orig_w, orig_h, fps, total))
        print("  Output : %s/" % out_dir)
        print("  v3 fixes: thr=0.02, full-frame suppress, collision freeze, "
              "preserve collision_logged on ball loss")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames      = 0
    ball_pos           = None
    ball_source        = "none"

    pedal_state = [{
        "spark_streak":     0,
        "collision_logged": False,
        "collision_frame":  -1,
        "spark_logged":     False,
        "freeze_remaining": 0,
    } for _ in pedals]

    hole_state = [{
        "entry_streak":     0,
        "entry_logged":     False,
        "was_in_hole":      False,
        "throw_count":      0,
        "last_entry_frame": -999,
    } for _ in holes]

    scoring_events    = []
    spark_intensities = {}
    frame_idx         = 0
    saved_count       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── spark suppression (v3: full frame, threshold 0.02) ─
        global_spark_val = global_spark_intensity(frame)
        spark_active     = global_spark_val > SPARK_SUPPRESS_THRESHOLD

        # ── bg sub + morph ─────────────────────────────────────
        small    = cv2.resize(frame, (process_w, process_h))
        fg_raw   = bg_sub.apply(small)
        fg_morph = cv2.morphologyEx(fg_raw,   cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_morph = cv2.morphologyEx(fg_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ── v3 fix 3: freeze Kalman if collision pending ───────
        any_collision_pending = any(
            s["collision_logged"] and not s["spark_logged"]
            for s in pedal_state
        )

        candidate, all_candidates = find_ball_candidate(fg_morph) \
            if not spark_active and not any_collision_pending else (None, [])
        ball_pos    = None
        ball_source = "none"

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
            missed_frames = 0
            ball_pos      = (ox, oy, or_)
            ball_source   = "detected"
        else:
            if not spark_active:
                missed_frames += 1
            if kalman_initialized and (missed_frames < MAX_MISSED or spark_active):
                pred    = kf.predict()
                px = int(float(pred[0].flat[0]) * sx)
                py = int(float(pred[1].flat[0]) * sy)
                ball_pos    = (px, py, 15)
                ball_source = "predicted"
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                ball_pos    = None
                ball_source = "lost"
                # v3 fix 2: preserve collision_logged on ball loss
                for s in pedal_state:
                    if not s["collision_logged"]:
                        s["spark_streak"] = 0
                for s in hole_state:
                    s["entry_logged"] = False
                    s["entry_streak"] = 0
                    s["was_in_hole"]  = False

        # ── pedal detection ────────────────────────────────────
        event_tags = []
        for i, (p, state) in enumerate(zip(pedals, pedal_state)):
            ball_in_roi = False
            if ball_pos:
                bx, by, _ = ball_pos
                ball_in_roi = (p["x"] <= bx <= p["x"]+p["w"] and
                               p["y"] <= by <= p["y"]+p["h"])

            if ball_in_roi and not state["collision_logged"]:
                state["collision_logged"] = True
                state["collision_frame"]  = frame_idx
                event_tags.append("COLLISION_%s" % p["name"])

            if state["collision_logged"] and not state["spark_logged"]:
                roi = (p["x"], p["y"], p["w"], p["h"])
                spark_ok, spark_val, _ = detect_spark(frame, roi)
                spark_intensities[p["name"]] = spark_val
                state["spark_streak"] = state["spark_streak"]+1 if spark_ok else 0
                if state["spark_streak"] >= spark_confirm_frames:
                    state["spark_logged"] = True
                    scoring_events.append((frame_idx, p["name"]))
                    event_tags.append("SPARK_%s" % p["name"])
            else:
                if not state["collision_logged"]:
                    state["spark_streak"] = 0

        # ── hole detection ─────────────────────────────────────
        for i, (h, state) in enumerate(zip(holes, hole_state)):
            in_hole, _ = detect_hole_entry(frame, h, ball_pos)
            if in_hole:
                state["entry_streak"] += 1
            else:
                if state["was_in_hole"]:
                    state["entry_logged"] = False
                state["entry_streak"] = 0
            state["was_in_hole"] = in_hole
            COOLDOWN_FRAMES = 30
            cooldown_ok = (frame_idx - state["last_entry_frame"]) > COOLDOWN_FRAMES
            if (state["entry_streak"] >= hole_confirm_frames
                    and not state["entry_logged"] and cooldown_ok):
                state["entry_logged"]     = True
                state["last_entry_frame"] = frame_idx
                state["throw_count"]     += 1
                event_tags.append("HOLE_%s" % h["name"])

        # ── save frame ─────────────────────────────────────────
        is_event   = len(event_tags) > 0
        should_save = is_event or (save_all and frame_idx % every_n_frames == 0)

        if should_save:
            tag = "_".join(event_tags) if event_tags else "frame"
            fname = save_step_frame(
                frame_idx, fps,
                frame, small,
                fg_raw, fg_morph,
                all_candidates,
                ball_pos, ball_source, kalman_initialized,
                pedals, pedal_state, holes, hole_state,
                spark_intensities,
                spark_active, global_spark_val,
                any_collision_pending,
                out_dir, clip_label, tag,
            )
            saved_count += 1
            if is_event and not silent:
                print("    F%04d  EVENT=%s  -> %s" % (
                    frame_idx, tag, Path(fname).name))

        frame_idx += 1

    cap.release()

    result = "miss"
    if scoring_events:
        scoring_events.sort(key=lambda e: e[0])
        result = scoring_events[0][1]

    if not silent:
        print("\n  Pipeline result : %s" % result)
        print("  Saved %d debug frames to %s/" % (saved_count, out_dir))
        print("  Scoring events  : %s" % scoring_events)
    return result


# ================================================================
#  Entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-step pipeline debug (v3) for ONE clip.")
    parser.add_argument("--config",       default="config.json")
    parser.add_argument("--clip",         required=True)
    parser.add_argument("--out-dir",      default="./test/debug/step_debug")
    parser.add_argument("--spark-frames", type=int, default=1)
    parser.add_argument("--hole-frames",  type=int, default=1)
    parser.add_argument("--every-n",      type=int, default=1)
    parser.add_argument("--events-only",  action="store_true",
                        help="Only save frames where an event fires")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    pedals = cfg.get("pedals", [])
    holes  = cfg.get("holes",  [])

    print("\n  Config : %d pedal(s), %d hole(s)" % (len(pedals), len(holes)))
    print("  v3: SPARK_SUPPRESS_THRESHOLD=%.2f  detect_spark_threshold=0.02  "
          "COLLISION_FREEZE=%d" % (SPARK_SUPPRESS_THRESHOLD, COLLISION_FREEZE_FRAMES))

    run_pipeline_debug(
        video_path=args.clip,
        pedals=pedals,
        holes=holes,
        spark_confirm_frames=args.spark_frames,
        hole_confirm_frames=args.hole_frames,
        out_dir=args.out_dir,
        every_n_frames=args.every_n,
        save_all=not args.events_only,
    )


if __name__ == "__main__":
    main()