# Bombhole — Computer Vision Approach

Ball tracking, collision detection and scoring for a skee-ball style game ("bombhole"),
from GoPro footage. All scripts are standalone CLIs — run them from the repository root.

## Where the data lives

| Drive folder | Local path | Size | Files | Contents |
|---|---|---:|---:|---|
| `source/raw` | `source/raw/` | 1.5 GB | 7 | Uncut GoPro recordings, `GX019499`–`GX019506` — [[click me](https://drive.google.com/drive/folders/1yhA9hurvCbpeMlqRg2qLiuNw9V36ZAtW?usp=drive_link)] |
| `source/clips` | `source/clips/` | 2.4 GB | 175 | Per-throw clips cut from the raw recordings — [[click me](https://drive.google.com/drive/folders/1UdVGtl5nDi8U-7G3oinbU9TPhVUouv3t?usp=drive_link)] |
| `test/data` | `test/data/` | 61 MB | 5 | Clips used for ad-hoc testing — [[click me](https://drive.google.com/drive/folders/1zHcfz4qdboYuLSFhSWhpr5asR_Gafq0i?usp=drive_link)] |
| `test/results` | `test/results/` | 2.3 GB | 802 | Classified clips, annotated videos, extracted frames — [[click me](https://drive.google.com/drive/folders/1rfuW5oyEOKi9bHKBOQ7edp9CNw-HXY_M?usp=drive_link)] |
| `test/debug` | `test/debug/` | 526 MB | 672 | Frame-level debug dumps — [[click me](https://drive.google.com/drive/folders/17Ek7LaTwJRhuEOOawzS_u4W9bWqyip3t?usp=drive_link)] |

To work with the pipeline, download those folders back to the same paths inside your
clone. They are listed in `.gitignore`, so they will never be committed by accident.

The small text results **are** in the repo: `test/results/confusion_matrix/*.txt` and
`test/results/hole_results.csv`.

## Layout

```
config.json          pedal + hole geometry and scores (the live config)
pedals.json          older pedal-only calibration, kept for reference

source/              raw video input — nothing here is generated   [Google Drive]
  raw/                 uncut GoPro recordings
  clips/               single-throw clips cut out of raw/ by segment_throws.py

src/                 all code                                      [in git]
  segment_throws.py    raw recording  ->  per-throw clips
  ball_tracker.py      MOG2 background subtraction + contour ball tracking
  collision_detector.py  ball tracking + pedal/hole collision + scoring overlay
  classify_clips.py    label each clip Pedal_1..Pedal_5 / miss
  classify_holes.py    label each clip Hole_1..Hole_4
  visualize_hole.py    render hole zones over a clip for calibration
  eval/                confusion matrices (v1 -> v3, plus the hole variant)
  debug/               frame-by-frame pipeline dumps
  prototypes/          scratch experiments, not part of the pipeline

test/                test inputs and everything generated from them
  data/                clips used for ad-hoc testing                [Google Drive]
  results/                                                          [Google Drive]
    classified/          classify_clips.py output, by predicted pedal
    classifiedFlash/     same, flash-based detection run
    classified_holes/    classify_holes.py output, by predicted hole
    tracked/             ball_tracker.py annotated videos
    collision/           collision overlays
    scoring/             scored videos
    holedetection/       hole-scored videos
    hole_viz/            visualize_hole.py calibration renders
    frames/              extracted frames, per clip
    debug_output/        one-off scored videos
    confusion_matrix/    saved confusion matrix reports             [in git]
    hole_results.csv     classify_holes.py summary                  [in git]
  debug/               frame-level debug dumps                      [Google Drive]
    misclassified/       frames for clips the classifier got wrong
    v2/ v3/              per-pipeline-version dumps
    step_debug/          debugPipeline.py step-by-step frames

docs/                research write-ups, experiment plans, notes    [in git]
```

## Pipeline

```bash
# 1. cut a raw recording into per-throw clips
python3 src/segment_throws.py source/raw/GX019505.MP4 --output-dir ./source/clips

# 2. classify every clip against the pedal zones
python3 src/classify_clips.py source/clips/*.MP4 --config config.json \
    --copy-to ./test/results/classified --output test/results/results.csv

# 3. score the classifier against the hand-sorted ground truth
python3 src/eval/confusion_matrix_v3.py --config config.json \
    --classified-dir ./test/results/classified

# hole variant
python3 src/classify_holes.py source/clips/*.MP4 --config config.json \
    --copy-to ./test/results/classified_holes --output test/results/hole_results.csv
python3 src/eval/confusion_matrix_holes.py --config config.json \
    --classified-dir ./test/results/classified_holes
```

Single-clip inspection:

```bash
python3 src/collision_detector.py source/clips/GX019507_throw47.MP4 \
    --config config.json --save --output-dir test/results/scoring
python3 src/visualize_hole.py source/clips/GX019502_throw16.MP4 --config config.json
python3 src/debug/debugPipeline.py --config config.json --clip source/clips/GX019505_throw10.MP4
```

Output-directory defaults in every script already point into `test/`, so running them
from the repository root writes to the right place without extra flags.

## Requirements

Python 3 with `opencv-python` and `numpy`.

```bash
pip3 install opencv-python numpy
```
