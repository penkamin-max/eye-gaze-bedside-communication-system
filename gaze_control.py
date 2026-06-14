"""
PATIENT-SIDE EYE-GAZE BEDSIDE COMMUNICATION SYSTEM

Purpose:
- Patient uses eye gaze to send requests to nurse/caregiver.
- Main communication: Firebase Realtime Database.
- Backup communication: local requests.json if Firebase fails.
- Patient-side screen can read caregiver feedback status:
  Pending -> Acknowledged -> Completed.

Controls:
Q = quit
P = pause
R = reset START
F = fullscreen/windowed
C = recalibrate center
X = invert left/right cursor
Y = invert up/down cursor

Backup demo keys:
E = send Emergency
W = send I need water
N = send Call Nurse

Interaction:
START screen:
- Look at START and hold 1.36s

Main screen:
- LEFT  = send current patient request
- RIGHT = next request
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import math
import time
import json
import os
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception as e:
    print("[REQUESTS IMPORT ERROR]", e)
    REQUESTS_AVAILABLE = False


# ============================================================
# FIREBASE IMPORT
# ============================================================

try:
    from firebase_writer import send_request_to_firebase
    FIREBASE_AVAILABLE = True
except Exception as e:
    print("[FIREBASE IMPORT ERROR]", e)
    FIREBASE_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME = "PATIENT GAZE COMMUNICATION"
CAMERA_PREVIEW_WINDOW = "Webcam Preview"

CAMERA_INDEX = 0

CANVAS_W = 1280
CANVAS_H = 720

START_DWELL_TIME_SECONDS = 1.36
PAGE_DWELL_TIME_SECONDS = 1.36
ACTION_COOLDOWN = 1.8

SMOOTH_WINDOW = 10
ZONE_HISTORY_LEN = 8
ZONE_MIN_VOTES = 5
ZONE_CONFIDENCE_THRESHOLD = 3.8

CURSOR_GAIN_X = 8.0
CURSOR_GAIN_Y = 5.0

INVERT_X_DEFAULT = False
INVERT_Y_DEFAULT = False

FULLSCREEN_ON_START = True
SHOW_MINI_PREVIEW = True
SHOW_SEPARATE_PREVIEW = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUESTS_FILE = os.path.join(BASE_DIR, "requests.json")

FIREBASE_DATABASE_URL = (
    "https://eye-gaze-nurse-call-default-rtdb.asia-southeast1.firebasedatabase.app"
)


# ============================================================
# REQUEST DATA
# ============================================================

REQUEST_PAGES = [
    {
        "name": "Call Nurse",
        "subtitle": "Goi y ta",
        "priority": "High",
        "icon": "nurse",
    },
    {
        "name": "Emergency",
        "subtitle": "Khan cap",
        "priority": "High",
        "icon": "emergency",
    },
    {
        "name": "I am in pain",
        "subtitle": "Toi bi dau",
        "priority": "High",
        "icon": "pain",
    },
    {
        "name": "I need water",
        "subtitle": "Toi can nuoc",
        "priority": "Low",
        "icon": "water",
    },
    {
        "name": "I need restroom assistance",
        "subtitle": "Toi can ho tro di ve sinh",
        "priority": "Low",
        "icon": "restroom",
    },
    {
        "name": "I feel uncomfortable",
        "subtitle": "Toi thay kho chiu",
        "priority": "Low",
        "icon": "uncomfortable",
    },
]


PRIORITY_MAP = {
    "Call Nurse": "High",
    "Emergency": "High",
    "I am in pain": "High",
    "I need water": "Low",
    "I need restroom assistance": "Low",
    "I feel uncomfortable": "Low",
}


# ============================================================
# GAZE PROFILE DATA
# ============================================================

GAZE_PROFILES = {
    "CENTER": {
        "x": 0.492857,
        "y": 0.657872,
        "std_x": 0.020279,
        "std_y": 0.084422
    },
    "LEFT_BOTTOM": {
        "x": 0.519941,
        "y": 0.671203,
        "std_x": 0.048168,
        "std_y": 0.088579
    },
    "LEFT_TOP": {
        "x": 0.522546,
        "y": 0.616839,
        "std_x": 0.035206,
        "std_y": 0.069304
    },
    "RIGHT_BOTTOM": {
        "x": 0.458045,
        "y": 0.676935,
        "std_x": 0.026032,
        "std_y": 0.087910
    },
    "RIGHT_TOP": {
        "x": 0.482287,
        "y": 0.614277,
        "std_x": 0.038854,
        "std_y": 0.077592
    },
}


# ============================================================
# COLORS - BGR
# ============================================================

COLOR_BG = (12, 4, 14)
COLOR_PANEL = (30, 18, 35)

COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 120)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_BLUE = (255, 120, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_GRAY = (120, 120, 120)

PRIORITY_COLOR_MAP = {
    "High": COLOR_RED,
    "Low": COLOR_BLUE,
}


# ============================================================
# FIREBASE REST STATUS FUNCTIONS
# ============================================================

def firebase_url(path):
    path = path.strip("/")
    return f"{FIREBASE_DATABASE_URL}/{path}.json"


def get_current_request_from_firebase():
    if not REQUESTS_AVAILABLE:
        return None

    response = requests.get(firebase_url("current_request"), timeout=5)

    if response.status_code != 200:
        raise RuntimeError(f"Cannot read current_request: {response.text}")

    return response.json()


def get_current_status_from_firebase(expected_request_id=None):
    data = get_current_request_from_firebase()

    if not data:
        return None

    if expected_request_id is not None:
        try:
            firebase_id = int(data.get("request_id", -1))
            expected_id = int(expected_request_id)

            if firebase_id != expected_id:
                return None

        except Exception:
            return None

    return data.get("status", None)


def clear_current_request_from_firebase():
    if not REQUESTS_AVAILABLE:
        return False

    response = requests.delete(firebase_url("current_request"), timeout=5)

    if response.status_code != 200:
        raise RuntimeError(f"Cannot clear current_request: {response.text}")

    print("[FIREBASE] current_request cleared")
    return True


# ============================================================
# LOCAL JSON BACKUP FUNCTIONS
# ============================================================

def load_requests():
    if not os.path.exists(REQUESTS_FILE):
        return []

    try:
        with open(REQUESTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception as e:
        print("[LOCAL JSON LOAD ERROR]", e)
        return []


def save_requests(requests_list):
    try:
        with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
            json.dump(requests_list, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print("[LOCAL JSON SAVE ERROR]", e)


def append_patient_request(request_type):
    """
    Backup local writer.
    This writes the request into requests.json if Firebase is unavailable.
    """
    requests_list = load_requests()

    if len(requests_list) == 0:
        next_id = 1
    else:
        ids = []
        for req in requests_list:
            try:
                ids.append(int(req.get("request_id", 0)))
            except Exception:
                pass

        next_id = max(ids) + 1 if ids else 1

    new_request = {
        "request_id": next_id,
        "type": request_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "priority": PRIORITY_MAP.get(request_type, "Low"),
        "status": "Pending"
    }

    requests_list.append(new_request)
    save_requests(requests_list)

    print("[LOCAL BACKUP REQUEST SENT]")
    print(new_request)

    return new_request


def send_patient_request(request_type):
    """
    Main request sender.
    Priority:
    1. Send to Firebase Realtime Database.
    2. If Firebase fails, save to local requests.json.
    """
    if FIREBASE_AVAILABLE:
        try:
            sent_request = send_request_to_firebase(request_type)
            print("[REQUEST SENT TO FIREBASE]")
            return sent_request

        except Exception as e:
            print("[FIREBASE SEND ERROR]", e)
            print("[BACKUP MODE] Saving request to local requests.json")

    return append_patient_request(request_type)


# ============================================================
# GEOMETRY
# ============================================================

def clamp(v, low, high):
    return max(low, min(high, v))


def get_start_circle():
    return CANVAS_W // 2, CANVAS_H // 2 + 20, 115


def get_left_button_rect():
    return 80, 210, 590, 560


def get_next_button_rect():
    return 690, 210, 1200, 560


def point_in_rect(px, py, rect, padding=35):
    x1, y1, x2, y2 = rect
    return (
        x1 - padding <= px <= x2 + padding and
        y1 - padding <= py <= y2 + padding
    )


def point_in_circle(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2


# ============================================================
# DRAW UTILS
# ============================================================

def draw_centered_text(
    img,
    text,
    y,
    font_scale,
    color,
    thickness,
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (CANVAS_W - text_size[0]) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def draw_grid_bg(img):
    img[:] = COLOR_BG

    for x in range(0, CANVAS_W, 80):
        cv2.line(img, (x, 0), (x, CANVAS_H), (22, 12, 25), 1)

    for y in range(0, CANVAS_H, 80):
        cv2.line(img, (0, y), (CANVAS_W, y), (22, 12, 25), 1)

    for y in range(0, CANVAS_H, 6):
        cv2.line(img, (0, y), (CANVAS_W, y), (16, 8, 18), 1)


def draw_glow_rect(img, rect, color, active=False):
    x1, y1, x2, y2 = rect

    if active:
        cv2.rectangle(img, (x1 - 12, y1 - 12), (x2 + 12, y2 + 12), color, 1)
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), color, 2)
        fill = tuple(int(c * 0.22) for c in color)
    else:
        fill = COLOR_PANEL

    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    length = 55

    cv2.line(img, (x1, y1), (x1 + length, y1), color, 5)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, 5)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, 5)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, 5)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, 5)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, 5)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, 5)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, 5)


def draw_progress_bar(img, rect, progress, color):
    x1, y1, x2, y2 = rect

    bar_x1 = x1 + 35
    bar_x2 = x2 - 35
    bar_y1 = y2 - 42
    bar_y2 = y2 - 22

    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), color, 2)

    fill_w = int((bar_x2 - bar_x1) * clamp(progress, 0.0, 1.0))

    cv2.rectangle(
        img,
        (bar_x1 + 2, bar_y1 + 2),
        (bar_x1 + fill_w, bar_y2 - 2),
        color,
        -1
    )


def draw_gaze_cursor(img, x, y, face_detected=True):
    if not face_detected:
        return

    x = int(clamp(x, 0, CANVAS_W - 1))
    y = int(clamp(y, 0, CANVAS_H - 1))

    cv2.circle(img, (x, y), 30, COLOR_GREEN, 1)
    cv2.circle(img, (x, y), 19, COLOR_CYAN, 2)
    cv2.circle(img, (x, y), 6, COLOR_YELLOW, -1)

    cv2.line(img, (x - 38, y), (x + 38, y), COLOR_CYAN, 1)
    cv2.line(img, (x, y - 38), (x, y + 38), COLOR_CYAN, 1)


# ============================================================
# SIMPLE ICONS
# ============================================================

def draw_icon_nurse(img, cx, cy, size, color):
    cv2.circle(img, (cx, cy - 20), size // 2, color, 3)
    cv2.rectangle(img, (cx - size, cy + 5), (cx + size, cy + size), color, 3)
    cv2.line(img, (cx - 18, cy - 45), (cx + 18, cy - 45), color, 5)
    cv2.line(img, (cx, cy - 63), (cx, cy - 27), color, 5)


def draw_icon_emergency(img, cx, cy, size, color):
    pts = np.array([
        [cx, cy - size],
        [cx - size, cy + size],
        [cx + size, cy + size],
    ], np.int32)
    cv2.polylines(img, [pts], True, color, 4)
    cv2.line(img, (cx, cy - 30), (cx, cy + 25), color, 6)
    cv2.circle(img, (cx, cy + 55), 6, color, -1)


def draw_icon_pain(img, cx, cy, size, color):
    cv2.circle(img, (cx, cy), size, color, 3)

    cv2.line(
        img,
        (cx - 25, cy - 15),
        (cx - 5, cy - 5),
        color,
        3
    )

    cv2.line(
        img,
        (cx + 25, cy - 15),
        (cx + 5, cy - 5),
        color,
        3
    )

    cv2.ellipse(
        img,
        (cx, cy + 25),
        (30, 18),
        0,
        200,
        340,
        color,
        3
    )


def draw_icon_water(img, cx, cy, size, color):
    pts = np.array([
        [cx, cy - size],
        [cx - size // 2, cy],
        [cx, cy + size],
        [cx + size // 2, cy],
    ], np.int32)
    cv2.polylines(img, [pts], True, color, 4)


def draw_icon_restroom(img, cx, cy, size, color):
    cv2.circle(img, (cx - 30, cy - 40), 15, color, 3)
    cv2.line(img, (cx - 30, cy - 20), (cx - 30, cy + 45), color, 4)
    cv2.line(img, (cx - 55, cy + 5), (cx - 5, cy + 5), color, 4)

    cv2.circle(img, (cx + 35, cy - 40), 15, color, 3)
    cv2.rectangle(img, (cx + 15, cy - 15), (cx + 55, cy + 45), color, 3)


def draw_icon_uncomfortable(img, cx, cy, size, color):
    cv2.circle(img, (cx, cy), size, color, 3)
    cv2.line(img, (cx - 35, cy - 15), (cx - 15, cy - 15), color, 3)
    cv2.line(img, (cx + 15, cy - 15), (cx + 35, cy - 15), color, 3)
    cv2.line(img, (cx - 30, cy + 25), (cx + 30, cy + 25), color, 3)


def draw_icon_next(img, cx, cy, size, color):
    pts = np.array([
        [cx - size // 2, cy - size],
        [cx - size // 2, cy + size],
        [cx + size, cy],
    ], np.int32)

    cv2.fillPoly(img, [pts], color)
    cv2.line(img, (cx + size + 10, cy - size), (cx + size + 10, cy + size), color, 8)


def draw_icon(img, icon, cx, cy, size, color):
    if icon == "nurse":
        draw_icon_nurse(img, cx, cy, size, color)
    elif icon == "emergency":
        draw_icon_emergency(img, cx, cy, size, color)
    elif icon == "pain":
        draw_icon_pain(img, cx, cy, size, color)
    elif icon == "water":
        draw_icon_water(img, cx, cy, size, color)
    elif icon == "restroom":
        draw_icon_restroom(img, cx, cy, size, color)
    elif icon == "uncomfortable":
        draw_icon_uncomfortable(img, cx, cy, size, color)


# ============================================================
# GAZE CONTROLLER
# ============================================================

class GazeController:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        self.raw_x = 0.5
        self.raw_y = 0.5
        self.smooth_x = 0.5
        self.smooth_y = 0.5

        self.face_detected = False
        self.head_status = "NO_FACE"

        self.gaze_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.zone_history = deque(maxlen=ZONE_HISTORY_LEN)

        self.current_zone = None
        self.stable_zone = None
        self.zone_score = 999.0

        self.baseline_x = 0.5
        self.baseline_y = 0.5
        self.baseline_ready = False

        self.invert_x = INVERT_X_DEFAULT
        self.invert_y = INVERT_Y_DEFAULT

        self.last_frame = None

    def is_open(self):
        return self.cap.isOpened()

    def read_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return False, None

        frame = cv2.flip(frame, 1)
        self.last_frame = frame.copy()

        return True, frame

    def process(self, frame):
        if frame is None:
            self.face_detected = False
            return

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self.face_detected = False
            self.head_status = "NO_FACE"
            self.current_zone = None
            self.stable_zone = None
            return

        self.face_detected = True
        landmarks = results.multi_face_landmarks[0].landmark

        raw = self.calculate_raw_gaze(landmarks, w, h)

        if raw is not None:
            self.raw_x, self.raw_y = raw
            self.gaze_buffer.append((self.raw_x, self.raw_y))

        if len(self.gaze_buffer) > 0:
            self.smooth_x = float(np.mean([p[0] for p in self.gaze_buffer]))
            self.smooth_y = float(np.mean([p[1] for p in self.gaze_buffer]))

        self.update_head_status(landmarks)
        self.detect_profile_zone()

    def calculate_raw_gaze(self, landmarks, frame_w, frame_h):
        try:
            left_iris = [468, 469, 470, 471]
            right_iris = [473, 474, 475, 476]

            def center(ids):
                xs = [landmarks[i].x * frame_w for i in ids]
                ys = [landmarks[i].y * frame_h for i in ids]
                return sum(xs) / len(xs), sum(ys) / len(ys)

            left_ix, left_iy = center(left_iris)
            right_ix, right_iy = center(right_iris)

            left_corner_a = landmarks[33].x * frame_w
            left_corner_b = landmarks[133].x * frame_w

            right_corner_a = landmarks[362].x * frame_w
            right_corner_b = landmarks[263].x * frame_w

            def ratio_x(iris_x, a, b):
                mn = min(a, b)
                mx = max(a, b)

                if abs(mx - mn) < 1:
                    return 0.5

                return (iris_x - mn) / (mx - mn)

            left_ratio = ratio_x(left_ix, left_corner_a, left_corner_b)
            right_ratio = ratio_x(right_ix, right_corner_a, right_corner_b)

            raw_x = (left_ratio + right_ratio) / 2.0
            raw_y = ((left_iy + right_iy) / 2.0) / frame_h

            raw_x = clamp(raw_x, 0.0, 1.0)
            raw_y = clamp(raw_y, 0.0, 1.0)

            return raw_x, raw_y

        except Exception:
            return None

    def update_head_status(self, landmarks):
        try:
            nose_x = landmarks[1].x

            if 0.25 <= nose_x <= 0.75:
                self.head_status = "OK"
            else:
                self.head_status = "TURNED"

        except Exception:
            self.head_status = "UNKNOWN"

    def detect_profile_zone(self):
        if not self.face_detected:
            self.current_zone = None
            self.stable_zone = None
            return

        best_zone = None
        best_score = 999.0

        for zone_name, profile in GAZE_PROFILES.items():
            dx = (self.smooth_x - profile["x"]) / max(profile["std_x"], 0.01)
            dy = (self.smooth_y - profile["y"]) / max(profile["std_y"], 0.01)

            score = math.sqrt(dx * dx + dy * dy)

            if score < best_score:
                best_score = score
                best_zone = zone_name

        self.zone_score = best_score

        if best_score <= ZONE_CONFIDENCE_THRESHOLD:
            self.current_zone = best_zone
            self.zone_history.append(best_zone)
        else:
            self.current_zone = None
            self.zone_history.append(None)

        if len(self.zone_history) >= ZONE_MIN_VOTES:
            counter = Counter(self.zone_history)
            zone, count = counter.most_common(1)[0]

            if zone is not None and count >= ZONE_MIN_VOTES:
                self.stable_zone = zone
            else:
                self.stable_zone = None

    def calibrate_center(self):
        self.baseline_x = self.smooth_x
        self.baseline_y = self.smooth_y
        self.baseline_ready = True

        print(f"[CALIBRATE] baseline = ({self.baseline_x:.3f}, {self.baseline_y:.3f})")

    def get_cursor(self):
        if not self.face_detected:
            return CANVAS_W // 2, CANVAS_H // 2

        if not self.baseline_ready:
            nx = self.smooth_x
            ny = self.smooth_y
        else:
            dx = self.smooth_x - self.baseline_x
            dy = self.smooth_y - self.baseline_y

            if self.invert_x:
                dx = -dx

            if self.invert_y:
                dy = -dy

            nx = 0.5 + dx * CURSOR_GAIN_X
            ny = 0.5 + dy * CURSOR_GAIN_Y

        px = int(clamp(nx, 0.02, 0.98) * CANVAS_W)
        py = int(clamp(ny, 0.02, 0.98) * CANVAS_H)

        return px, py

    def toggle_invert_x(self):
        self.invert_x = not self.invert_x
        print(f"[TOGGLE] invert_x = {self.invert_x}")

    def toggle_invert_y(self):
        self.invert_y = not self.invert_y
        print(f"[TOGGLE] invert_y = {self.invert_y}")

    def release(self):
        if self.cap is not None:
            self.cap.release()

        self.face_mesh.close()


# ============================================================
# PATIENT APP
# ============================================================

class PatientGazeApp:
    def __init__(self):
        self.controller = GazeController()

        self.state = "START"
        self.paused = False
        self.fullscreen = FULLSCREEN_ON_START

        self.page_index = 0

        self.hover_id = None
        self.hover_start_time = None
        self.active_dwell_time = START_DWELL_TIME_SECONDS

        self.last_action_time = 0

        self.last_sent_request = None
        self.last_sent_request_id = None

        self.confirm_screen_start = None
        self.confirm_screen_duration = 10.0

        self.patient_feedback_status = "Pending"
        self.last_status_poll_time = 0.0
        self.status_poll_interval = 1.0

    def reset_dwell(self):
        self.hover_id = None
        self.hover_start_time = None

    def update_dwell(self, target_id, dwell_time):
        self.active_dwell_time = dwell_time

        if target_id is None:
            self.reset_dwell()
            return 0.0, False

        if self.hover_id != target_id:
            self.hover_id = target_id
            self.hover_start_time = time.time()
            return 0.0, False

        elapsed = time.time() - self.hover_start_time
        progress = elapsed / dwell_time

        if elapsed >= dwell_time:
            self.reset_dwell()
            return 1.0, True

        return clamp(progress, 0.0, 1.0), False

    def can_send(self):
        return time.time() - self.last_action_time >= ACTION_COOLDOWN

    def record_send(self):
        self.last_action_time = time.time()

    def register_sent_request(self, sent_request):
        self.last_sent_request = sent_request
        self.last_sent_request_id = sent_request.get("request_id")
        self.patient_feedback_status = sent_request.get("status", "Pending")
        self.last_status_poll_time = 0.0

        self.confirm_screen_start = time.time()
        self.state = "CONFIRM"

        self.record_send()
        self.reset_dwell()

    def poll_patient_feedback_status(self):
        if not FIREBASE_AVAILABLE:
            return

        if self.last_sent_request_id is None:
            return

        now = time.time()

        if now - self.last_status_poll_time < self.status_poll_interval:
            return

        self.last_status_poll_time = now

        try:
            status = get_current_status_from_firebase(self.last_sent_request_id)

            if status:
                self.patient_feedback_status = status

                if self.last_sent_request is not None:
                    self.last_sent_request["status"] = status

                print("[PATIENT FEEDBACK STATUS]", status)

        except Exception as e:
            print("[STATUS POLL ERROR]", e)

    def get_start_hover(self, cursor_x, cursor_y):
        cx, cy, r = get_start_circle()

        if self.controller.face_detected and point_in_circle(cursor_x, cursor_y, cx, cy, r + 90):
            return "START"

        return None

    def get_page_hover(self, cursor_x, cursor_y):
        left_rect = get_left_button_rect()
        next_rect = get_next_button_rect()

        if point_in_rect(cursor_x, cursor_y, left_rect, padding=55):
            return "SEND_REQUEST"

        if point_in_rect(cursor_x, cursor_y, next_rect, padding=55):
            return "NEXT"

        z = self.controller.stable_zone

        if z in ["LEFT_TOP", "LEFT_BOTTOM"]:
            return "SEND_REQUEST"

        if z in ["RIGHT_TOP", "RIGHT_BOTTOM"]:
            return "NEXT"

        return None

    def draw_debug(self, canvas):
        c = self.controller
        current_request = REQUEST_PAGES[self.page_index]["name"]

        firebase_status = "ON" if FIREBASE_AVAILABLE else "OFF"

        lines = [
            f"RAW: {c.raw_x:.3f}, {c.raw_y:.3f}",
            f"SMOOTH: {c.smooth_x:.3f}, {c.smooth_y:.3f}",
            f"ZONE: {c.current_zone}",
            f"STABLE: {c.stable_zone}",
            f"SCORE: {c.zone_score:.2f}",
            f"HEAD: {c.head_status}",
            f"FACE: {'YES' if c.face_detected else 'NO'}",
            f"CAM: {CAMERA_INDEX} ({c.frame_w}x{c.frame_h})",
            f"PAGE: {self.page_index + 1}/{len(REQUEST_PAGES)} {current_request}",
            f"DWELL: {self.active_dwell_time:.2f}s",
            f"INV_X: {c.invert_x}  INV_Y: {c.invert_y}",
            f"FIREBASE: {firebase_status}",
            f"PATIENT STATUS: {self.patient_feedback_status}",
            f"BACKUP: {REQUESTS_FILE}",
        ]

        color = COLOR_CYAN if c.face_detected else COLOR_RED

        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (12, 25 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    def draw_mini_preview(self, canvas):
        if not SHOW_MINI_PREVIEW:
            return

        frame = self.controller.last_frame

        if frame is None:
            return

        pw, ph = 200, 140
        px = CANVAS_W - pw - 20
        py = CANVAS_H - ph - 20

        small = cv2.resize(frame, (pw, ph))

        canvas[py:py + ph, px:px + pw] = small

        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_CYAN, 2)

        cv2.putText(
            canvas,
            "CAM PREVIEW",
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_CYAN,
            1,
        )

    def draw_start(self, canvas, cursor_x, cursor_y, dwell_progress):
        draw_grid_bg(canvas)

        draw_centered_text(
            canvas,
            "EYE-GAZE BEDSIDE",
            105,
            2.1,
            COLOR_CYAN,
            4
        )

        draw_centered_text(
            canvas,
            "COMMUNICATION SYSTEM",
            165,
            1.65,
            COLOR_MAGENTA,
            4
        )

        cx, cy, r = get_start_circle()

        hovered = self.hover_id == "START"
        color = COLOR_YELLOW if hovered else COLOR_CYAN

        if hovered:
            cv2.circle(canvas, (cx, cy), r + 25, COLOR_YELLOW, 1)
            cv2.circle(canvas, (cx, cy), r + 15, COLOR_YELLOW, 2)

        cv2.circle(canvas, (cx, cy), r, color, 3)
        cv2.circle(canvas, (cx, cy), r - 10, color, 1)

        cv2.putText(
            canvas,
            "START",
            (cx - 88, cy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            COLOR_YELLOW,
            4,
        )

        bar_rect = (cx - 190, cy + r + 45, cx + 190, cy + r + 70)
        x1, y1, x2, y2 = bar_rect

        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CYAN, 2)

        fill_w = int((x2 - x1) * dwell_progress)

        cv2.rectangle(
            canvas,
            (x1 + 2, y1 + 2),
            (x1 + fill_w, y2 - 2),
            COLOR_GREEN,
            -1
        )

        draw_centered_text(
            canvas,
            "Look at START and hold 1.36 seconds",
            CANVAS_H - 95,
            0.75,
            COLOR_CYAN,
            2
        )

        draw_centered_text(
            canvas,
            "Q Quit | D Clear | E Emergency | W Water | N Nurse | F Fullscreen",
            CANVAS_H - 55,
            0.55,
            COLOR_GRAY,
            1
        )

        self.draw_debug(canvas)
        self.draw_mini_preview(canvas)
        draw_gaze_cursor(canvas, cursor_x, cursor_y, self.controller.face_detected)

    def draw_page(self, canvas, cursor_x, cursor_y, hover_id, dwell_progress):
        draw_grid_bg(canvas)

        request = REQUEST_PAGES[self.page_index]
        request_color = PRIORITY_COLOR_MAP.get(request["priority"], COLOR_BLUE)

        draw_centered_text(
            canvas,
            f"REQUEST {self.page_index + 1}/{len(REQUEST_PAGES)}",
            60,
            1.1,
            COLOR_CYAN,
            3
        )

        draw_centered_text(
            canvas,
            "LOOK LEFT TO SEND REQUEST  |  LOOK RIGHT FOR NEXT",
            105,
            0.75,
            COLOR_MAGENTA,
            2
        )

        left_rect = get_left_button_rect()
        next_rect = get_next_button_rect()

        left_active = hover_id == "SEND_REQUEST"
        next_active = hover_id == "NEXT"

        draw_glow_rect(canvas, left_rect, request_color, active=left_active)

        x1, y1, x2, y2 = left_rect

        draw_icon(
            canvas,
            request["icon"],
            (x1 + x2) // 2,
            y1 + 100,
            58,
            request_color
        )

        text_scale = 1.25
        if len(request["name"]) > 18:
            text_scale = 0.9

        cv2.putText(
            canvas,
            request["name"],
            (x1 + 45, y1 + 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            COLOR_YELLOW if left_active else COLOR_WHITE,
            3,
        )

        cv2.putText(
            canvas,
            request["subtitle"],
            (x1 + 55, y1 + 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_CYAN,
            2,
        )

        cv2.putText(
            canvas,
            f"PRIORITY: {request['priority'].upper()}",
            (x1 + 105, y1 + 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            request_color,
            2,
        )

        if left_active:
            draw_progress_bar(canvas, left_rect, dwell_progress, COLOR_GREEN)

        draw_glow_rect(canvas, next_rect, COLOR_CYAN, active=next_active)

        nx1, ny1, nx2, ny2 = next_rect

        draw_icon_next(
            canvas,
            (nx1 + nx2) // 2 - 20,
            ny1 + 120,
            55,
            COLOR_CYAN
        )

        cv2.putText(
            canvas,
            "NEXT",
            (nx1 + 165, ny1 + 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            COLOR_YELLOW if next_active else COLOR_WHITE,
            4,
        )

        next_request = REQUEST_PAGES[(self.page_index + 1) % len(REQUEST_PAGES)]["name"]

        next_scale = 0.75
        if len(next_request) > 20:
            next_scale = 0.55

        cv2.putText(
            canvas,
            f"Next: {next_request}",
            (nx1 + 80, ny1 + 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            next_scale,
            COLOR_CYAN,
            2,
        )

        if next_active:
            draw_progress_bar(canvas, next_rect, dwell_progress, COLOR_GREEN)

        cv2.putText(
            canvas,
            "Q Quit | P Pause | R Reset | D Clear | E Emergency | W Water | N Nurse",
            (35, CANVAS_H - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_CYAN,
            1,
        )

        self.draw_debug(canvas)
        self.draw_mini_preview(canvas)
        draw_gaze_cursor(canvas, cursor_x, cursor_y, self.controller.face_detected)

    def draw_confirmation(self, canvas, cursor_x, cursor_y):
        draw_grid_bg(canvas)

        if self.last_sent_request is None:
            return

        req = self.last_sent_request
        priority = req.get("priority", "Low")
        request_color = PRIORITY_COLOR_MAP.get(priority, COLOR_BLUE)

        status = req.get("status", self.patient_feedback_status)

        status_color = COLOR_CYAN
        feedback_text = "Waiting for caregiver..."

        if status == "Acknowledged":
            status_color = COLOR_YELLOW
            feedback_text = "Caregiver acknowledged your request"

        elif status == "Completed":
            status_color = COLOR_GREEN
            feedback_text = "Request completed"

        elif status == "Pending":
            status_color = COLOR_CYAN
            feedback_text = "Waiting for caregiver..."

        draw_centered_text(
            canvas,
            "REQUEST SENT",
            105,
            2.0,
            COLOR_GREEN,
            4
        )

        draw_centered_text(
            canvas,
            req.get("type", "Unknown request"),
            210,
            1.35,
            request_color,
            3
        )

        draw_centered_text(
            canvas,
            f"Priority: {priority}",
            285,
            0.9,
            COLOR_YELLOW,
            2
        )

        draw_centered_text(
            canvas,
            f"Status: {status}",
            345,
            1.05,
            status_color,
            3
        )

        draw_centered_text(
            canvas,
            feedback_text,
            405,
            0.85,
            status_color,
            2
        )

        draw_centered_text(
            canvas,
            f"Time: {req.get('timestamp', '')}",
            465,
            0.75,
            COLOR_WHITE,
            2
        )

        save_message = "Sent to Firebase Realtime Database"
        if not FIREBASE_AVAILABLE:
            save_message = f"Saved to local backup: {REQUESTS_FILE}"

        draw_centered_text(
            canvas,
            save_message,
            525,
            0.7,
            COLOR_GRAY,
            2
        )

        draw_centered_text(
            canvas,
            "Returning to request menu...",
            CANVAS_H - 85,
            0.7,
            COLOR_CYAN,
            2
        )

        self.draw_debug(canvas)
        self.draw_mini_preview(canvas)
        draw_gaze_cursor(canvas, cursor_x, cursor_y, self.controller.face_detected)

    def send_backup_request_by_key(self, request_type):
        """
        Backup demo shortcut.
        This does not replace gaze control; it is only for safe demo recovery.
        """
        try:
            sent = send_patient_request(request_type)
            self.register_sent_request(sent)
            print(f"[BACKUP KEY] {request_type} sent")

        except Exception as e:
            print("[BACKUP KEY ERROR]", e)

    def run(self):
        if not self.controller.is_open():
            print(f"Cannot open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 1 or 2.")
            return

        print("Camera opened.")
        print("Patient-side gaze communication system started.")

        if FIREBASE_AVAILABLE:
            print("Request mode: Firebase Realtime Database")
        else:
            print("Request mode: Local backup requests.json only")

        print("Local backup file:", REQUESTS_FILE)
        print("Q=quit | P=pause | R=reset | F=fullscreen | X/Y=invert cursor | C=recalibrate")
        print("Backup keys: D=Clear current request | E=Emergency | W=Water | N=Call Nurse")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

        if SHOW_SEPARATE_PREVIEW:
            cv2.namedWindow(CAMERA_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(CAMERA_PREVIEW_WINDOW, 360, 220)

        while True:
            ret, frame = self.controller.read_frame()

            if not ret:
                print("Cannot read camera frame.")
                break

            if SHOW_SEPARATE_PREVIEW:
                preview = cv2.resize(frame, (360, 220))
                cv2.imshow(CAMERA_PREVIEW_WINDOW, preview)

            if not self.paused:
                self.controller.process(frame)

            cursor_x, cursor_y = self.controller.get_cursor()

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            if self.state == "START":
                start_target = self.get_start_hover(cursor_x, cursor_y)
                dwell_progress, activated = self.update_dwell(
                    start_target,
                    START_DWELL_TIME_SECONDS
                )

                if activated:
                    self.controller.calibrate_center()
                    self.state = "PAGE"
                    self.reset_dwell()
                    print("Entered request menu.")

                self.draw_start(canvas, cursor_x, cursor_y, dwell_progress)

            elif self.state == "CONFIRM":
                self.poll_patient_feedback_status()
                self.draw_confirmation(canvas, cursor_x, cursor_y)

                if self.confirm_screen_start is not None:
                    elapsed = time.time() - self.confirm_screen_start

                    if elapsed >= self.confirm_screen_duration:
                        self.state = "PAGE"
                        self.confirm_screen_start = None
                        self.reset_dwell()

            else:
                hover_id = self.get_page_hover(cursor_x, cursor_y)
                dwell_progress, activated = self.update_dwell(
                    hover_id,
                    PAGE_DWELL_TIME_SECONDS
                )

                if activated:
                    if hover_id == "SEND_REQUEST":
                        if self.can_send():
                            request = REQUEST_PAGES[self.page_index]

                            sent = send_patient_request(request["name"])
                            self.register_sent_request(sent)

                    elif hover_id == "NEXT":
                        self.page_index = (self.page_index + 1) % len(REQUEST_PAGES)
                        print(f"Next request: {REQUEST_PAGES[self.page_index]['name']}")
                        self.reset_dwell()

                self.draw_page(canvas, cursor_x, cursor_y, hover_id, dwell_progress)

            if self.paused:
                cv2.putText(
                    canvas,
                    "PAUSED",
                    (CANVAS_W // 2 - 120, CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.7,
                    COLOR_RED,
                    4,
                )

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q")]:
                break

            if key in [ord("p"), ord("P")]:
                self.paused = not self.paused
                print(f"Paused: {self.paused}")

            if key in [ord("r"), ord("R")]:
                self.state = "START"
                self.page_index = 0
                self.controller.baseline_ready = False
                self.last_sent_request = None
                self.last_sent_request_id = None
                self.patient_feedback_status = "Pending"
                self.confirm_screen_start = None
                self.reset_dwell()
                print("Reset to START.")

            if key in [ord("m"), ord("M")]:
                self.state = "PAGE"
                self.controller.calibrate_center()
                self.reset_dwell()
                print("Go to request menu.")

            if key in [ord("c"), ord("C")]:
                self.controller.calibrate_center()
                self.reset_dwell()

            if key in [ord("x"), ord("X")]:
                self.controller.toggle_invert_x()
                self.reset_dwell()

            if key in [ord("y"), ord("Y")]:
                self.controller.toggle_invert_y()
                self.reset_dwell()

            if key in [ord("e"), ord("E")]:
                self.send_backup_request_by_key("Emergency")

            if key in [ord("w"), ord("W")]:
                self.send_backup_request_by_key("I need water")

            if key in [ord("n"), ord("N")]:
                self.send_backup_request_by_key("Call Nurse")

            if key in [ord("d"), ord("D")]:
                try:
                    if FIREBASE_AVAILABLE:
                        clear_current_request_from_firebase()
                        self.last_sent_request = None
                        self.last_sent_request_id = None
                        self.patient_feedback_status = "Pending"
                        self.confirm_screen_start = None
                        self.state = "PAGE"
                        self.reset_dwell()
                        print("[CLEAR KEY] Firebase current_request cleared")
                    else:
                        print("[CLEAR ERROR] Firebase is not available")

                except Exception as e:
                    print("[CLEAR ERROR]", e)

            if key in [ord("f"), ord("F")]:
                self.fullscreen = not self.fullscreen

                if self.fullscreen:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN
                    )
                else:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

        self.controller.release()
        cv2.destroyAllWindows()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    app = PatientGazeApp()
    app.run()