import cv2
import time
import uuid
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from torchreid.models import build_model
import torch

# -----------------------------
# Model setup
# -----------------------------
# Load the trained YOLO model for luggage/person detection
yolo_model = YOLO("C:/Users/kumar/OneDrive/Desktop/Mini_project/runs/yolov8_training6/weights/best.pt")

# Load OSNet for person ReID (appearance-based tracking)
reid_model = build_model(name='osnet_x0_25', num_classes=1000, pretrained=True)
reid_model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = reid_model.to(device)

# -----------------------------
# Parameters and thresholds
# -----------------------------
DETECTION_CONFIDENCE = 0.5     # YOLO confidence threshold

# Abandoned luggage rules
ABANDONED_TIME_THRESHOLD = 5.0    # seconds before bag is considered abandoned
NEAR_DISTANCE_MULTIPLIER = 1.5    # how far a person can be to count as "near" luggage

# Resize for speed
FRAME_RESIZE_FACTOR = 0.5

# Your dataset classes
LUGGAGE_CLASS = 0
PERSON_CLASS  = 1

# ReID & tracking parameters
SIMILARITY_THRESHOLD = 0.75       # minimum cosine similarity to match the same person
PERSON_MAX_AGE = 2.0              # how long to keep a person track if they disappear
LUGGAGE_MAX_AGE = 2.0             # how long to keep a luggage track if it disappears
FEATURE_MOMENTUM = 0.7            # smoothing factor for appearance features
SPATIAL_MAX_RATIO = 2.0           # tolerance for spatial drift of tracked persons

DEBUG = True  # set False to disable logs

# -----------------------------
# Tracking state
# -----------------------------
# persons: { id: {bbox, feature, last_seen, first_seen} }
tracked_persons = {}
# luggage: { id: {bbox, center, person_nearby, abandoned_since, last_seen, first_seen} }
tracked_luggage = {}

# -----------------------------
# Utility functions
# -----------------------------
def log_debug(msg):
    if DEBUG:
        print(msg)

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def calculate_distance(c1, c2):
    return np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

def _prep_person_crop_for_reid(img_bgr):
    """
    Convert cropped person image to the format OSNet expects:
    - BGR → RGB
    - scale to [0,1]
    - resize to 128x256 (done before calling)
    - HWC → NCHW tensor
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    chw = img_rgb.transpose(2, 0, 1)
    chw = np.expand_dims(chw, axis=0)
    tensor = torch.from_numpy(chw).to(device)
    return tensor

def extract_person_features(frame, bbox):
    """Crop a person from the frame and get their ReID embedding."""
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 512), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (128, 256))
    tensor = _prep_person_crop_for_reid(crop)

    with torch.no_grad():
        feat = reid_model(tensor)  # shape [1, D]
    feat = feat.cpu().numpy()

    # normalize so cosine similarity is stable
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
    feat = feat / norm
    return feat

def iou_xyxy(a, b):
    """Compute IoU between two boxes in (x1,y1,x2,y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter + 1e-12
    return inter / union

# -----------------------------
# Person tracking with ReID
# -----------------------------
def match_persons_to_tracks(person_detections, frame, current_time):
    """
    Match detected persons to existing tracks using:
    - ReID cosine similarity
    - spatial proximity (center distance)
    Unmatched persons start new tracks, and old tracks are kept alive briefly.
    """
    global tracked_persons

    # Only keep tracks that are still recent
    alive_tracks = {
        pid: pdata for pid, pdata in tracked_persons.items()
        if (current_time - pdata['last_seen']) <= PERSON_MAX_AGE
    }

    # Compute embeddings for all detected persons
    det_items = []
    for det in person_detections:
        bbox = tuple(det.xyxy[0])
        feat = extract_person_features(frame, bbox)
        center = get_center(bbox)
        det_items.append({'bbox': bbox, 'feature': feat, 'center': center})

    assigned_tracks = set()
    new_tracked = {}

    # Try to match each detection to an existing track
    for det in det_items:
        best_id = None
        best_sim = -1.0

        # Use bbox size to define how far a track can drift
        x1, y1, x2, y2 = det['bbox']
        diag = calculate_distance((x1, y1), (x2, y2))
        spatial_gate = SPATIAL_MAX_RATIO * max(10.0, diag)

        for pid, pdata in alive_tracks.items():
            if pid in assigned_tracks:
                continue

            sim = cosine_similarity(det['feature'], pdata['feature'])[0][0]
            d = calculate_distance(det['center'], get_center(pdata['bbox']))
            overlap = iou_xyxy(det['bbox'], pdata['bbox'])

            if sim >= SIMILARITY_THRESHOLD and d <= spatial_gate:
                score = sim + 0.1 * overlap
                if score > best_sim:
                    best_sim = score
                    best_id = pid

        if best_id is not None:
            # Update track with EMA-smoothed features
            old_feat = tracked_persons[best_id]['feature']
            new_feat = FEATURE_MOMENTUM * old_feat + (1.0 - FEATURE_MOMENTUM) * det['feature']
            norm = np.linalg.norm(new_feat, axis=1, keepdims=True) + 1e-12
            new_feat = new_feat / norm

            new_tracked[best_id] = {
                'bbox': det['bbox'],
                'feature': new_feat,
                'last_seen': current_time,
                'first_seen': tracked_persons[best_id]['first_seen'],
            }
            assigned_tracks.add(best_id)
        else:
            # Start a new track
            pid = str(uuid.uuid4())
            new_tracked[pid] = {
                'bbox': det['bbox'],
                'feature': det['feature'],
                'last_seen': current_time,
                'first_seen': current_time,
            }
            assigned_tracks.add(pid)
            log_debug(f"New person track: {pid}")

    # Keep unmatched old tracks alive for a short while
    for pid, pdata in tracked_persons.items():
        if pid in new_tracked:
            continue
        age = current_time - pdata['last_seen']
        if age <= PERSON_MAX_AGE:
            new_tracked[pid] = pdata

    tracked_persons = new_tracked

# -----------------------------
# Luggage tracking
# -----------------------------
def match_luggage_to_tracks(luggage_detections, frame, current_time):
    """
    Track luggage using center distance and box similarity.
    Mark as abandoned if no person is nearby for too long.
    """
    global tracked_luggage

    alive_tracks = {
        lid: ldata for lid, ldata in tracked_luggage.items()
        if (current_time - ldata['last_seen']) <= LUGGAGE_MAX_AGE
    }

    new_tracked = {}
    assigned = set()

    for det in luggage_detections:
        bbox = tuple(det.xyxy[0])
        center = get_center(bbox)
        diag = calculate_distance((bbox[0], bbox[1]), (bbox[2], bbox[3]))
        proximity_threshold = NEAR_DISTANCE_MULTIPLIER * max(10.0, diag)

        best_id, best_score = None, 1e18
        for lid, ldata in alive_tracks.items():
            if lid in assigned:
                continue

            d = calculate_distance(center, ldata['center'])
            bbox_distance = np.mean(np.abs(np.array(bbox) - np.array(ldata['bbox'])))

            score = d + 0.01 * bbox_distance
            if d <= proximity_threshold and score < best_score:
                best_score = score
                best_id = lid

        if best_id is None:
            # New luggage track
            lid = str(uuid.uuid4())
            new_tracked[lid] = {
                'bbox': bbox,
                'center': center,
                'last_seen': current_time,
                'first_seen': current_time,
                'person_nearby': False,
                'abandoned_since': None
            }
            assigned.add(lid)
            log_debug(f"New luggage track: {lid}")
        else:
            prev = tracked_luggage[best_id]

            # Check if any tracked person is close
            is_near_person = False
            for pid, pdata in tracked_persons.items():
                pd_center = get_center(pdata['bbox'])
                if calculate_distance(center, pd_center) <= proximity_threshold:
                    is_near_person = True
                    break

            # Update abandoned timer
            if is_near_person:
                abandoned_since = None
            else:
                abandoned_since = prev['abandoned_since']
                if abandoned_since is None:
                    abandoned_since = current_time

            new_tracked[best_id] = {
                'bbox': bbox,
                'center': center,
                'last_seen': current_time,
                'first_seen': prev['first_seen'],
                'person_nearby': is_near_person,
                'abandoned_since': abandoned_since
            }
            assigned.add(best_id)

    # Keep unmatched luggage tracks for a short while
    for lid, ldata in tracked_luggage.items():
        if lid in new_tracked:
            continue
        age = current_time - ldata['last_seen']
        if age <= LUGGAGE_MAX_AGE:
            new_tracked[lid] = ldata

    tracked_luggage = new_tracked

# -----------------------------
# Frame processing
# -----------------------------
def process_frame(frame):
    global tracked_persons, tracked_luggage

    if FRAME_RESIZE_FACTOR != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * FRAME_RESIZE_FACTOR), int(h * FRAME_RESIZE_FACTOR)))

    # Run YOLO detection
    results = yolo_model(frame)[0]
    dets = results.boxes.cpu().numpy()

    # Filter detections by class
    luggage_detections = [det for det in dets if int(det.cls[0]) == LUGGAGE_CLASS and det.conf[0] >= DETECTION_CONFIDENCE]
    person_detections = [det for det in dets if int(det.cls[0]) == PERSON_CLASS and det.conf[0] >= DETECTION_CONFIDENCE]

    log_debug(f"Detected {len(luggage_detections)} luggage, {len(person_detections)} persons")

    # Update tracks
    current_time = time.time()
    match_persons_to_tracks(person_detections, frame, current_time)
    match_luggage_to_tracks(luggage_detections, frame, current_time)

    # Draw luggage
    for lid, ldata in tracked_luggage.items():
        x1, y1, x2, y2 = map(int, ldata['bbox'])
        box_color = (255, 0, 0)
        label = "Luggage"

        # Mark as abandoned if timer exceeds threshold
        if not ldata['person_nearby'] and ldata['abandoned_since'] is not None:
            t_stationary = current_time - ldata['abandoned_since']
            if t_stationary >= ABANDONED_TIME_THRESHOLD:
                label = f"Abandoned ({int(t_stationary)}s)"
                box_color = (0, 0, 255)
                log_debug(f"Luggage {lid} abandoned for {t_stationary:.2f}s")
            else:
                label = f"Luggage ({int(t_stationary)}s)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Draw persons
    for pid, pdata in tracked_persons.items():
        x1, y1, x2, y2 = map(int, pdata['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(frame, f"Person {pid[:8]}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2)

    return frame

# -----------------------------
# Main loop
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame)
            cv2.imshow("Abandoned Object Detection with ReID (OSNet)", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam session ended.")

if __name__ == "__main__":
    main()
