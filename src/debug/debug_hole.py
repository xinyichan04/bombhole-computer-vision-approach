#!/usr/bin/env python3
"""
debug_hole.py - prints ball position vs hole position every frame
Usage: python3 src/debug/debug_hole.py source/clips/GX019507_throw57.MP4 --config config.json --hole Hole_3
"""
import cv2
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--config", default="config.json")
parser.add_argument("--hole", default=None)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)
holes = cfg.get("holes", [])
if args.hole:
    holes = [h for h in holes if h["name"] == args.hole]

cap = cv2.VideoCapture(args.input)
fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
process_w, process_h = 540, 960
sx = orig_w / process_w
sy = orig_h / process_h

bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def find_ball(fg_mask):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 3000: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circ = 4 * np.pi * area / (perimeter**2)
        if circ < 0.3: continue
        score = circ * area
        if score > best_score:
            best_score = score
            (x,y), r = cv2.minEnclosingCircle(cnt)
            best = (int(x*sx), int(y*sy))
    return best

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    small = cv2.resize(frame, (process_w, process_h))
    fg = bg_sub.apply(small)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    ball = find_ball(fg)

    for h in holes:
        if ball:
            bx, by = ball
            dist = np.sqrt((bx-h["cx"])**2 + (by-h["cy"])**2)
            in_hole = dist <= h["r"]
            print(f"  Frame {frame_idx:3d} ({frame_idx/fps:.2f}s)  ball=({bx},{by})  "
                  f"{h['name']} center=({h['cx']},{h['cy']}) r={h['r']}  "
                  f"dist={dist:.1f}  IN={'YES ✓' if in_hole else 'no'}")
        else:
            print(f"  Frame {frame_idx:3d} ({frame_idx/fps:.2f}s)  ball=None")
    frame_idx += 1

cap.release()