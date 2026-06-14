import cv2
import csv
import time
import os
import math
import mediapipe as mp
from collections import deque
from datetime import datetime


# =========================
# CONFIG
# =========================

SCREEN_W = 1280
SCREEN_H = 720

# SUA SO NAY THEO check_camera.py
# 0 = webcam laptop
# 1 hoac 2 = webcam ngoai
CAMERA_INDEX = 1

DATA_DIR = "gaze_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

TARGETS = [
    "CENTER",
    "LEFT_TOP",
    "LEFT_BOTTOM",
    "RIGHT_TOP",
    "RIGHT_BOTTOM",
]

TARGET_POSITIONS = {
    "CENTER": (SCREEN_W // 2, SCREEN_H // 2),
    "LEFT_TOP": (220, 190),
    "LEFT_BOTTOM": (220, 530),
    "RIGHT_TOP": (1060, 190),
    "RIGHT_BOTTOM": (1060, 530),
}

TARGET_RADIUS = 55

TRIALS_PER_TARGET = 5
COUNTDOWN_SEC = 1.5
RECORD_SEC = 2.0

SMOOTHING_WINDOW = 10


# =========================
# MEDIAPIPE SETUP
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================
# LANDMARK INDEXES
# =========================

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133

RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454


# =========================
# HELPER FUNCTIONS
# =========================

def landmark_to_point(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def euclidean(p1, p2):
    return math.dist(p1, p2)


def calculate_ear(landmarks, indexes, frame_w, frame_h):
    p = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in indexes]

    vertical_1 = euclidean(p[1], p[5])
    vertical_2 = euclidean(p[2], p[4])
    horizontal = euclidean(p[0], p[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def iris_center(landmarks, iris_indexes, frame_w, frame_h):
    points = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in iris_indexes]
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y


def calculate_eye_ratio_x(
    landmarks,
    iris_indexes,
    left_corner_idx,
    right_corner_idx,
    frame_w,
    frame_h
):
    iris_x, _ = iris_center(landmarks, iris_indexes, frame_w, frame_h)

    left_corner = landmark_to_point(landmarks[left_corner_idx], frame_w, frame_h)
    right_corner = landmark_to_point(landmarks[right_corner_idx], frame_w, frame_h)

    eye_width = right_corner[0] - left_corner[0]

    if abs(eye_width) < 1:
        return 0.5

    ratio_x = (iris_x - left_corner[0]) / eye_width
    return ratio_x


def calculate_gaze_raw(landmarks, frame_w, frame_h):
    left_ratio_x = calculate_eye_ratio_x(
        landmarks,
        LEFT_IRIS,
        LEFT_EYE_LEFT_CORNER,
        LEFT_EYE_RIGHT_CORNER,
        frame_w,
        frame_h
    )

    right_ratio_x = calculate_eye_ratio_x(
        landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_LEFT_CORNER,
        RIGHT_EYE_RIGHT_CORNER,
        frame_w,
        frame_h
    )

    raw_x = (left_ratio_x + right_ratio_x) / 2.0

    _, left_iris_y = iris_center(landmarks, LEFT_IRIS, frame_w, frame_h)
    _, right_iris_y = iris_center(landmarks, RIGHT_IRIS, frame_w, frame_h)

    raw_y = ((left_iris_y + right_iris_y) / 2.0) / frame_h

    return raw_x, raw_y


def calculate_head_offset(landmarks, frame_w, frame_h):
    nose = landmark_to_point(landmarks[NOSE_TIP], frame_w, frame_h)
    face_left = landmark_to_point(landmarks[FACE_LEFT], frame_w, frame_h)
    face_right = landmark_to_point(landmarks[FACE_RIGHT], frame_w, frame_h)

    face_center_x = (face_left[0] + face_right[0]) / 2
    face_width = abs(face_right[0] - face_left[0])

    if face_width < 1:
        return 0.0, "UNKNOWN"

    offset_x = (nose[0] - face_center_x) / face_width

    if abs(offset_x) < 0.10:
        status = "GOOD"
    elif offset_x > 0:
        status = "HEAD_RIGHT"
    else:
        status = "HEAD_LEFT"

    return offset_x, status


def draw_target(canvas, target_name):
    canvas[:] = (20, 20, 20)

    x, y = TARGET_POSITIONS[target_name]

    cv2.circle(canvas, (x, y), TARGET_RADIUS, (0, 255, 255), -1)
    cv2.circle(canvas, (x, y), TARGET_RADIUS + 8, (255, 255, 255), 3)

    cv2.putText(
        canvas,
        f"LOOK AT: {target_name}",
        (360, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3
    )

    cv2.putText(
        canvas,
        "Keep your head still. Look at the yellow circle.",
        (280, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2
    )


def draw_status(canvas, text, color=(255, 255, 255)):
    cv2.putText(
        canvas,
        text,
        (400, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )


# =========================
# MAIN
# =========================

def main():
    participant_id = input("Nhap participant_id, vi du P01_webcam: ").strip()

    if not participant_id:
        participant_id = "P_UNKNOWN"

    timestamp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"{participant_id}_{timestamp_name}.csv")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Co gang set do phan giai cao hon cho webcam ngoai
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        print("Hay sua CAMERA_INDEX = 0, 1, 2 trong file collect_gaze_data.py")
        return

    raw_x_buffer = deque(maxlen=SMOOTHING_WINDOW)
    raw_y_buffer = deque(maxlen=SMOOTHING_WINDOW)

    cv2.namedWindow("Gaze Data Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Data Collection", SCREEN_W, SCREEN_H)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "participant_id",
            "trial_id",
            "target_name",
            "phase",
            "timestamp",
            "raw_x",
            "raw_y",
            "smooth_raw_x",
            "smooth_raw_y",
            "mapped_x",
            "mapped_y",
            "left_ear",
            "right_ear",
            "head_offset_x",
            "head_pose_status",
            "fps",
            "valid_face",
            "camera_index"
        ])

        trial_id = 0

        print("Bat dau test.")
        print("Nhan Q de thoat som.")
        print(f"Data se luu vao: {csv_path}")

        for repeat in range(TRIALS_PER_TARGET):
            for target_name in TARGETS:
                trial_id += 1

                # Reset smoothing moi target de do sach hon
                raw_x_buffer.clear()
                raw_y_buffer.clear()

                # Countdown
                phase_start = time.time()

                while time.time() - phase_start < COUNTDOWN_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    canvas = cv2.resize(frame, (SCREEN_W, SCREEN_H))

                    draw_target(canvas, target_name)

                    remain = COUNTDOWN_SEC - (time.time() - phase_start)
                    draw_status(canvas, f"Get ready... {remain:.1f}s", (0, 200, 255))

                    cv2.imshow("Gaze Data Collection", canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"Da luu file: {csv_path}")
                        return

                # Record
                record_start = time.time()
                last_time = time.time()

                while time.time() - record_start < RECORD_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    fps = 1.0 / dt if dt > 0 else 0.0

                    frame = cv2.flip(frame, 1)
                    frame_h, frame_w = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    valid_face = 0

                    raw_x = None
                    raw_y = None
                    smooth_raw_x = None
                    smooth_raw_y = None
                    mapped_x = None
                    mapped_y = None
                    left_ear = None
                    right_ear = None
                    head_offset_x = None
                    head_pose_status = "NO_FACE"

                    if results.multi_face_landmarks:
                        valid_face = 1
                        landmarks = results.multi_face_landmarks[0].landmark

                        raw_x, raw_y = calculate_gaze_raw(landmarks, frame_w, frame_h)

                        raw_x_buffer.append(raw_x)
                        raw_y_buffer.append(raw_y)

                        smooth_raw_x = sum(raw_x_buffer) / len(raw_x_buffer)
                        smooth_raw_y = sum(raw_y_buffer) / len(raw_y_buffer)

                        mapped_x = int(smooth_raw_x * SCREEN_W)
                        mapped_y = int(smooth_raw_y * SCREEN_H)

                        left_ear = calculate_ear(landmarks, LEFT_EYE_EAR, frame_w, frame_h)
                        right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR, frame_w, frame_h)

                        head_offset_x, head_pose_status = calculate_head_offset(
                            landmarks,
                            frame_w,
                            frame_h
                        )

                    writer.writerow([
                        participant_id,
                        trial_id,
                        target_name,
                        "RECORD",
                        now,
                        raw_x,
                        raw_y,
                        smooth_raw_x,
                        smooth_raw_y,
                        mapped_x,
                        mapped_y,
                        left_ear,
                        right_ear,
                        head_offset_x,
                        head_pose_status,
                        fps,
                        valid_face,
                        CAMERA_INDEX
                    ])

                    canvas = cv2.resize(frame, (SCREEN_W, SCREEN_H))
                    draw_target(canvas, target_name)

                    progress = time.time() - record_start
                    draw_status(canvas, f"Recording... {progress:.1f}/{RECORD_SEC:.1f}s", (0, 255, 0))

                    if smooth_raw_x is not None and smooth_raw_y is not None:
                        cv2.putText(
                            canvas,
                            f"raw_x={smooth_raw_x:.3f} raw_y={smooth_raw_y:.3f}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            canvas,
                            f"head={head_pose_status} offset={head_offset_x:.3f}",
                            (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            canvas,
                            f"camera_index={CAMERA_INDEX}",
                            (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )
                    else:
                        cv2.putText(
                            canvas,
                            "NO FACE DETECTED",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2
                        )

                    cv2.imshow("Gaze Data Collection", canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"Da luu file: {csv_path}")
                        return

        print("Hoan thanh test.")
        print(f"Da luu file: {csv_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()