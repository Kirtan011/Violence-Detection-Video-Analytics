"""
fight_detection_AB.py

Run fight detection on a video using:
 - YOLO person detector (yolo11*.pt) for per-frame detections
 - Simple IOU-based tracker per person
 - R3D-18 fight vs non-fight classifier
 - Motion-based activity filter (Solution B)
 - 2-second continuity rule (Solution A)
 - Slightly enlarged bounding boxes for better visual coverage

Paths assume project root:
    D:/violence-detection

Usage (from project root):
    python -m inference.fight_detection_AB --video D:/violence-detection/Test-Video.mp4

Output:
    workspace/inference_outputs/<video_name>_fight_AB.mp4
"""

import argparse
from pathlib import Path
from collections import deque
from math import sqrt

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from ultralytics import YOLO


# ========= CONFIG: PATHS =========
ROOT = Path("D:/violence-detection").resolve()
WORKSPACE = ROOT / "workspace"
MODELS_DIR = WORKSPACE / "models"
OUT_DIR = WORKSPACE / "inference_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= MODEL / CLIP CONFIG =========
CLIP_LEN = 16
IMG_SIZE = 112
NUM_CLASSES = 2
FIGHT_PROB_THRESH = 0.5   # min probability to consider a fight candidate

# Motion activity thresholds
MOTION_WINDOW = 10
STAND_SPEED_THRESH = 0.5   # < 0.5 px/frame => idle
WALK_SPEED_THRESH = 3.0    # < 3 px/frame => walk, else active

# Tracking
MAX_AGE = 30
IOU_MATCH_THRESH = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet mean/std for preprocessing
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)


# ========= UTILS =========
def iou_xyxy(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


def expand_box(box, scale, frame_w, frame_h):
    """
    Expand box around its center by 'scale' factor.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    new_w = w * scale
    new_h = h * scale
    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(frame_w, cx + new_w / 2))
    ny2 = int(min(frame_h, cy + new_h / 2))
    return [nx1, ny1, nx2, ny2]


def preprocess_clip(frames_list):
    """
    frames_list: list of RGB frames (H,W,3), length = CLIP_LEN
    Returns: tensor (1,3,T,H,W) on DEVICE
    """
    arr = np.stack(frames_list, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    arr = np.transpose(arr, (3,0,1,2))  # (3,T,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,3,T,H,W)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.to(DEVICE)


def classify_clip(model, frames_list):
    """
    Runs classifier on a sequence of frames.
    Returns: label (0/1), fight_prob (float)
    """
    with torch.no_grad():
        x = preprocess_clip(frames_list)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        fight_prob = float(probs[1].cpu().item())
        label = int(torch.argmax(probs).cpu().item())
    return label, fight_prob


# ========= MAIN INFERENCE PIPELINE =========
def run_inference(video_path: Path, output_path: Path):
    print("Using video:", video_path)
    print("Output video:", output_path)
    print("Device:", DEVICE)

    # ------ Load YOLO ------
    yolo_weights = list(ROOT.glob("yolo11*.pt"))
    assert len(yolo_weights) > 0, f"No YOLO weights (yolo11*.pt) found in {ROOT}"
    YOLO_WEIGHTS = yolo_weights[0]
    print("Using YOLO weights:", YOLO_WEIGHTS)
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    # ------ Load classifier ------
    best_model_path = MODELS_DIR / "best_r3d18_fight_classifier.pth"
    assert best_model_path.exists(), f"Best classifier not found at {best_model_path}"

    model = video_models.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("Loaded classifier:", best_model_path)

    # ------ Open video ------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    print("FPS:", fps)
    min_fight_frames = int(2 * fps)   # 2-second rule
    print("Min fight frames (2s):", min_fight_frames)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # ------ Tracking state ------
    NEXT_TRACK_ID = 0
    TRACKS = {}  # tid -> dict

    FRAME_SKIP = 1
    CONF_THRESH = 0.25
    PERSON_CLASS = 0  # YOLO person class

    def update_tracks(detections, frame, frame_idx):
        """
        detections: list of [x1,y1,x2,y2,conf]
        Updates TRACKS and returns it.
        """
        nonlocal NEXT_TRACK_ID, TRACKS

        # age & delete old
        for tid in list(TRACKS.keys()):
            TRACKS[tid]["age"] += 1
            if TRACKS[tid]["age"] > MAX_AGE:
                del TRACKS[tid]

        # match detections
        for det in detections:
            box = det[:4]
            best_i = 0.0
            best_tid = None
            for tid, tr in TRACKS.items():
                i = iou_xyxy(box, tr["bbox"])
                if i > best_i:
                    best_i = i
                    best_tid = tid

            if best_i > IOU_MATCH_THRESH and best_tid is not None:
                tid = best_tid
            else:
                tid = NEXT_TRACK_ID
                NEXT_TRACK_ID += 1
                TRACKS[tid] = {
                    "bbox": box,
                    "age": 0,
                    "buffer": deque(maxlen=CLIP_LEN),
                    "centers": deque(maxlen=MOTION_WINDOW),
                    "activity": "idle",
                    "label": None,
                    "prob": 0.0,
                    "fight_start_frame": None,
                    "is_fight_confirmed": False,
                }

            TRACKS[tid]["bbox"] = box
            TRACKS[tid]["age"] = 0

            # motion centers
            cx, cy = center_of_box(box)
            TRACKS[tid]["centers"].append((cx, cy))

            speeds = []
            centers = list(TRACKS[tid]["centers"])
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                speeds.append(sqrt(dx*dx + dy*dy))
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

            if avg_speed < STAND_SPEED_THRESH:
                TRACKS[tid]["activity"] = "idle"
            elif avg_speed < WALK_SPEED_THRESH:
                TRACKS[tid]["activity"] = "walk"
            else:
                TRACKS[tid]["activity"] = "active"

            # crop for classifier
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            TRACKS[tid]["buffer"].append(crop)

            # classify when we have CLIP_LEN
            if len(TRACKS[tid]["buffer"]) == CLIP_LEN:
                frames_list = list(TRACKS[tid]["buffer"])
                label, prob = classify_clip(model, frames_list)
                TRACKS[tid]["label"] = label
                TRACKS[tid]["prob"] = prob

                # Solutions A+B: fight if:
                # - classifier says fight,
                # - prob >= threshold,
                # - activity = active
                if (
                    label == 1
                    and prob >= FIGHT_PROB_THRESH
                    and TRACKS[tid]["activity"] == "active"
                ):
                    if TRACKS[tid]["fight_start_frame"] is None:
                        TRACKS[tid]["fight_start_frame"] = max(0, frame_idx - CLIP_LEN + 1)

                    duration = frame_idx - TRACKS[tid]["fight_start_frame"] + 1
                    if duration >= min_fight_frames:
                        TRACKS[tid]["is_fight_confirmed"] = True
                else:
                    TRACKS[tid]["fight_start_frame"] = None
                    TRACKS[tid]["is_fight_confirmed"] = False

        return TRACKS

    print("Processing video with A+B logic (motion + 2s rule + enlarged boxes)...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        # YOLO detection
        results = yolo_model.predict(
            source=frame,
            conf=CONF_THRESH,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    cls_arr = b.cls.cpu().numpy().reshape(-1)[0]
                    cls_id = int(cls_arr)
                    if cls_id != PERSON_CLASS:
                        continue
                    xyxy = b.xyxy.cpu().numpy().reshape(-1)[:4].tolist()
                    conf = float(b.conf.cpu().numpy().reshape(-1)[0])
                    detections.append(xyxy + [conf])

        tracks = update_tracks(detections, frame, frame_idx)

        # Draw only confirmed fights (with enlarged boxes)
        for tid, tr in tracks.items():
            if not tr.get("is_fight_confirmed", False):
                continue

            x1, y1, x2, y2 = map(int, tr["bbox"])
            prob = tr.get("prob", 0.0)
            activity = tr.get("activity", "idle")

            # enlarge box visually
            ex1, ey1, ex2, ey2 = expand_box(
                [x1, y1, x2, y2], scale=1.4, frame_w=width, frame_h=height
            )

            color = (0, 0, 255)  # red
            text = f"FIGHT {prob*100:.1f}% ({activity})"

            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color, 2)
            cv2.putText(
                frame,
                text,
                (ex1, max(0, ey1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Done! Output saved to:", output_path)


# ========= CLI =========
def parse_args():
    parser = argparse.ArgumentParser(description="Fight detection A+B on a video.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video file (optional). If not provided, "
             "it will be saved under workspace/inference_outputs/."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if args.output is not None:
        output_path = Path(args.output).resolve()
    else:
        output_path = OUT_DIR / f"{video_path.stem}_fight_AB.mp4"

    run_inference(video_path, output_path)
