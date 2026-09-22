#!/usr/bin/env python3
"""
confusion_matrix_v2.py
----------------------
Same evaluation harness as confusion_matrix.py but with a rewritten
pedal detection strategy: SPARK-FIRST.

Old strategy (broken):
    ball enters ROI → collision_logged → spark detected in ROI → scored
    Problem: spark blob hijacks ball tracker at impact, corrupting position
             before collision_logged ever fires.

New strategy:
    spark detected in any pedal ROI → candidate pedals collected
    → if 1 candidate  : score it
    → if 2+ candidates: score the one closest to last_reliable_ball_pos
    Ball tracking is kept for:
      (a) hole entry detection (unchanged)
      (b) tiebreaker when spark bleeds across neighboring ROIs

State machine per throw:
    IDLE
      ball visible in frame (tracking live)
      spark appears in pedal ROI above threshold → PEDAL_SCORED
      ball enters hole → HOLE_SCORED
      spark_cooldown = SPARK_COOLDOWN_FRAMES, pedal states reset
      back to IDLE

Usage:
    python3 src/eval/confusion_matrix_v2.py --config config.json \
        --classified-dir ./classifiedFlash
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from glob import glob


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

SPARK_THRESHOLD        = 0.01   # min spark intensity to consider a pedal hit
SPARK_CONFIRM_FRAMES   = 3      # consecutive frames spark must appear to confirm
SPARK_COOLDOWN_FRAMES  = 15     # frames to suppress spark detection after hole entry
HOLE_CONFIRM_FRAMES    = 1      # frames ball must be in hole to confirm entry
HOLE_COOLDOWN_FRAMES   = 30     # frames between hole entries (existing logic)
MAX_MISSED_FRAMES      = 10     # Kalman prediction frames before ball declared lost


# ═══════════════════════════════════════════════════════════════
#  Pipeline helpers
# ═══════════════════════════════════════════════════════════════

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


def measure_spark_intensity(frame, roi):
    """
    Measure yellow/white spark pixel ratio inside a pedal ROI.
    Returns intensity float in [0, 1].
    """
    x, y, w, h = roi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,  100, 150])
    upper_yellow = np.array([45,  255, 255])
    lower_white  = np.array([0,   0,   220])
    upper_white  = np.array([180, 40,  255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_white,  upper_white)
    )
    return cv2.countNonZero(mask) / max((x2-x1)*(y2-y1), 1)


def detect_hole_entry(hole, ball_pos):
    """Returns (in_hole: bool, dist: float)"""
    if ball_pos is None:
        return False, 0.0
    bx, by, _ = ball_pos
    dist = float(np.sqrt((bx - hole["cx"])**2 + (by - hole["cy"])**2))
    return dist <= (hole["r"] + 20), round(dist, 1)


def pedal_center(p):
    return (p["x"] + p["w"] / 2, p["y"] + p["h"] / 2)


def dist2d(ax, ay, bx, by):
    return float(np.sqrt((ax - bx)**2 + (ay - by)**2))


# ═══════════════════════════════════════════════════════════════
#  Core per-throw state
# ═══════════════════════════════════════════════════════════════

def make_pedal_state():
    return {
        "spark_streak":  0,
        "spark_logged":  False,   # scored this throw
    }

def make_hole_state():
    return {
        "entry_streak":     0,
        "entry_logged":     False,
        "was_in_hole":      False,
        "throw_count":      0,
        "last_entry_frame": -999,
    }

def reset_pedal_states(pedal_state):
    """Reset all pedals for the next throw."""
    for s in pedal_state:
        s["spark_streak"] = 0
        s["spark_logged"] = False

def reset_hole_entry_flags(hole_state):
    """Allow hole re-entry detection for next throw."""
    for s in hole_state:
        s["entry_logged"] = False
        s["entry_streak"] = 0
        s["was_in_hole"]  = False


# ═══════════════════════════════════════════════════════════════
#  Main classifier
# ═══════════════════════════════════════════════════════════════

def classify_clip_full(video_path, pedals, holes,
                       process_w=540, process_h=960,
                       debug=False):
    """
    Spark-first pedal detection pipeline.

    Scoring events collected: (frame_idx, pedal_name)
    Returns the name of the FIRST pedal scored, or "miss".

    Pedal scoring (NEW):
        Each frame, measure spark intensity in every pedal ROI.
        Collect pedals with intensity > SPARK_THRESHOLD for
        SPARK_CONFIRM_FRAMES consecutive frames.
        If multiple pedals trigger simultaneously, pick the one
        whose center is closest to last_reliable_ball_pos.

    Hole scoring (UNCHANGED):
        Ball center within hole circle + tolerance for
        HOLE_CONFIRM_FRAMES frames, with HOLE_COOLDOWN_FRAMES
        between entries.
        After each hole entry: reset pedal states + set
        spark_cooldown so residue from impact doesn't bleed
        into the next throw.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "miss"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sx     = orig_w / process_w
    sy     = orig_h / process_h

    # ── Ball tracking ─────────────────────────────────────────
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized  = False
    missed_frames       = 0
    ball_pos            = None   # current frame ball pos (orig space)

    # Last position where ball was DIRECTLY detected (not predicted/lost).
    # Used as tiebreaker when multiple pedals spark simultaneously.
    last_reliable_ball_pos = None

    # ── Per-throw state ───────────────────────────────────────
    pedal_state  = [make_pedal_state() for _ in pedals]
    hole_state   = [make_hole_state()  for _ in holes]

    # Frames remaining to suppress spark detection (post hole-entry cooldown)
    spark_cooldown = 0

    # All pedal scoring events this clip: (frame_idx, pedal_name)
    scoring_events = []

    if debug:
        pnames = [p["name"] for p in pedals]
        header = f"  {'F':>5}  {'ball_pos':<18}  {'cooldown':>8}  " + \
                 "  ".join(f"{n}(s/str)" for n in pnames) + "  event"
        print(f"\n  {'─'*len(header)}")
        print(header)
        print(f"  {'─'*len(header)}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── ① Ball tracking ───────────────────────────────────
        small   = cv2.resize(frame, (process_w, process_h))
        fg_mask = bg_sub.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidate = find_ball_candidate(fg_mask)
        ball_pos  = None

        if candidate:
            cx, cy, r  = candidate
            ox, oy     = int(cx * sx), int(cy * sy)
            or_        = int(r * max(sx, sy))
            meas       = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not kalman_initialized:
                kf.statePre     = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_initialized = True
            kf.correct(meas)
            kf.predict()
            missed_frames = 0
            ball_pos      = (ox, oy, or_)
            # Only trust directly detected positions as tiebreaker reference
            last_reliable_ball_pos = ball_pos

        else:
            missed_frames += 1
            if kalman_initialized and missed_frames < MAX_MISSED_FRAMES:
                pred     = kf.predict()
                px       = int(float(pred[0].flat[0]) * sx)
                py       = int(float(pred[1].flat[0]) * sy)
                ball_pos = (px, py, 15)
            elif missed_frames >= MAX_MISSED_FRAMES:
                kalman_initialized = False
                ball_pos           = None
                # Ball truly lost — reset hole entry flags so next throw
                # can re-enter the same hole. Do NOT touch pedal spark state
                # (spark-first doesn't depend on ball being in ROI).
                reset_hole_entry_flags(hole_state)

        # ── ② Spark-first pedal detection ────────────────────
        spark_vals = {p["name"]: 0.0 for p in pedals}
        if spark_cooldown > 0:
            spark_cooldown -= 1
        else:
            # Measure spark intensity in every pedal ROI
            spark_vals = {}
            for p in pedals:
                roi = (p["x"], p["y"], p["w"], p["h"])
                spark_vals[p["name"]] = round(measure_spark_intensity(frame, roi), 4)

            # Update streak for each pedal
            candidates_this_frame = []   # pedals confirmed this frame
            for i, (p, state) in enumerate(zip(pedals, pedal_state)):
                if state["spark_logged"]:
                    # Already scored this throw — ignore
                    state["spark_streak"] = 0
                    continue

                if spark_vals[p["name"]] >= SPARK_THRESHOLD:
                    state["spark_streak"] += 1
                else:
                    state["spark_streak"] = 0

                if state["spark_streak"] >= SPARK_CONFIRM_FRAMES:
                    candidates_this_frame.append(i)
                    state["spark_streak"] = 0  # reset so it doesn't re-trigger next frame

            if candidates_this_frame:
                if len(candidates_this_frame) == 1:
                    winner_idx = candidates_this_frame[0]
                else:
                    if last_reliable_ball_pos is not None:
                        rbx, rby, _ = last_reliable_ball_pos
                        winner_idx = min(
                            candidates_this_frame,
                            key=lambda i: dist2d(
                                rbx, rby,
                                *pedal_center(pedals[i])
                            )
                        )
                    else:
                        winner_idx = max(
                            candidates_this_frame,
                            key=lambda i: spark_vals[pedals[i]["name"]]
                        )

                winner_name = pedals[winner_idx]["name"]
                pedal_state[winner_idx]["spark_logged"] = True
                scoring_events.append((frame_idx, winner_name))

                # Zero out all streaks to prevent residue re-triggering
                for state in pedal_state:
                    state["spark_streak"] = 0

        # ── ③ Hole detection (unchanged logic) ────────────────
        for h, state in zip(holes, hole_state):
            in_hole, _ = detect_hole_entry(h, ball_pos)

            if in_hole:
                state["entry_streak"] += 1
            else:
                if state["was_in_hole"]:
                    state["entry_logged"] = False
                state["entry_streak"] = 0
            state["was_in_hole"] = in_hole

            cooldown_ok = (frame_idx - state["last_entry_frame"]) > HOLE_COOLDOWN_FRAMES
            if (state["entry_streak"] >= HOLE_CONFIRM_FRAMES
                    and not state["entry_logged"]
                    and cooldown_ok):
                state["entry_logged"]     = True
                state["last_entry_frame"] = frame_idx
                state["throw_count"]     += 1
                # Throw is over — reset pedal states for next throw
                # and suppress spark for SPARK_COOLDOWN_FRAMES frames
                reset_pedal_states(pedal_state)
                spark_cooldown = SPARK_COOLDOWN_FRAMES
                # intentionally NOT added to scoring_events
                # (hole scoring is separate from pedal classification)

        # ── Debug print ───────────────────────────────────────
        if debug:
            # Ball position string
            if ball_pos:
                bstr = f"({ball_pos[0]:4d},{ball_pos[1]:4d})"
            else:
                bstr = "None"

            # Per-pedal spark intensity + streak
            pedal_cols = "  ".join(
                f"{spark_vals.get(p['name'], 0.0):.3f}/{pedal_state[i]['spark_streak']}"
                for i, p in enumerate(pedals)
            ) if spark_cooldown == 0 else "  ".join(
                f"{'---':>9}" for _ in pedals
            )

            # Event string
            events_this_frame = [e[1] for e in scoring_events if e[0] == frame_idx]
            hole_events = []
            for h, hs in zip(holes, hole_state):
                if hs["last_entry_frame"] == frame_idx:
                    hole_events.append(f"HOLE_{h['name']}")
            event_str = "  ".join(events_this_frame + hole_events) or ""

            # Highlight lines with events or high spark
            any_high = any(
                spark_vals.get(p["name"], 0.0) >= SPARK_THRESHOLD
                for p in pedals
            ) if spark_cooldown == 0 else False
            marker = "★" if events_this_frame else ("·" if any_high else " ")

            print(f"  {marker} F{frame_idx:04d}  {bstr:<18}  cd={spark_cooldown:>3}  "
                  f"{pedal_cols}  {event_str}")

        frame_idx += 1

    cap.release()

    if not scoring_events:
        return "miss"

    scoring_events.sort(key=lambda e: e[0])
    return scoring_events[0][1]


# ═══════════════════════════════════════════════════════════════
#  Confusion Matrix Printer (unchanged)
# ═══════════════════════════════════════════════════════════════

def print_confusion_matrix(matrix, classes):
    col_w   = 12
    label_w = 14

    print("\n  Confusion Matrix  (rows = Predicted, cols = Actual)\n")
    header = " " * label_w + "".join(f"{c:>{col_w}}" for c in classes)
    print("  " + header)
    print("  " + "─" * len(header))

    for i, row_cls in enumerate(classes):
        row = f"  {row_cls:<{label_w}}"
        for j in range(len(classes)):
            val  = matrix[i][j]
            cell = f"[{val}]" if i == j else f" {val} "
            row += f"{cell:>{col_w}}"
        print(row)

    print("  " + "─" * len(header))

    print("\n  Per-class metrics:")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "─" * 54)
    n = len(classes)
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[i][j] for j in range(n) if j != i)
        fn = sum(matrix[j][i] for j in range(n) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = tp + fn
        print(f"  {cls:<14} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")

    total = sum(matrix[i][i] for i in range(n))
    grand = sum(matrix[i][j] for i in range(n) for j in range(n))
    print(f"\n  Overall accuracy: {total}/{grand} = {total/max(grand,1)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix — spark-first pedal detection pipeline.")
    parser.add_argument("--config",         default="config.json")
    parser.add_argument("--classified-dir", default="./test/results/classified")
    parser.add_argument("--debug-clip",     default=None,
                        help="Print per-frame spark trace for this clip name "
                             "(e.g. GX019500_throw01.MP4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    pedals  = cfg.get("pedals", [])
    holes   = cfg.get("holes",  [])
    classes = [p["name"] for p in pedals] + ["miss"]

    print(f"\n  Config loaded: {len(pedals)} pedal(s), {len(holes)} hole(s)")
    print(f"  Classes ({len(classes)}): {classes}")
    print(f"\n  Pipeline: SPARK-FIRST")
    print(f"  SPARK_THRESHOLD      = {SPARK_THRESHOLD}")
    print(f"  SPARK_CONFIRM_FRAMES = {SPARK_CONFIRM_FRAMES}")
    print(f"  SPARK_COOLDOWN_FRAMES= {SPARK_COOLDOWN_FRAMES}")

    # ── Collect clips ────────────────────────────────────────
    all_clips = []
    for cls in classes:
        folder = os.path.join(args.classified_dir, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue
        clips = sorted(
            glob(os.path.join(folder, "*.MP4")) +
            glob(os.path.join(folder, "*.mp4"))
        )
        for c in clips:
            all_clips.append((c, cls))
        print(f"  {cls:<16s}: {len(clips)} clips")

    if not all_clips:
        print("\n  No clips found. Check --classified-dir.")
        return

    print(f"\n  Running on {len(all_clips)} clips...\n")

    cls_idx = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    matrix  = [[0]*n for _ in range(n)]
    wrong   = []

    for idx, (video_path, true_label) in enumerate(all_clips):
        name = Path(video_path).name
        print(f"  [{idx+1:3d}/{len(all_clips)}] {name}  (true={true_label}) ...",
              end=" ", flush=True)

        do_debug = args.debug_clip and (args.debug_clip == name)
        if do_debug:
            print(f"\n  ── DEBUG TRACE: {name}  (true={true_label}) ──")
        predicted = classify_clip_full(video_path, pedals, holes, debug=do_debug)

        ok = predicted == true_label
        print(f"pred={predicted}" + (" ✓" if ok else " ✗"))

        row = cls_idx.get(predicted, cls_idx["miss"])
        col = cls_idx.get(true_label, cls_idx["miss"])
        matrix[row][col] += 1

        if not ok:
            wrong.append((name, true_label, predicted))

    print_confusion_matrix(matrix, classes)

    if wrong:
        print(f"\n  Misclassified clips ({len(wrong)}):")
        for name, true, pred in wrong:
            print(f"    {name}  true={true}  pred={pred}")
    else:
        print("\n  All clips classified correctly!")


if __name__ == "__main__":
    main()