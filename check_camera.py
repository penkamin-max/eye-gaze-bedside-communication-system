import cv2

print("Dang kiem tra camera...\n")

for i in range(6):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            h, w = frame.shape[:2]
            print(f"Camera index {i}: OK - resolution {w}x{h}")
        else:
            print(f"Camera index {i}: Mo duoc nhung khong doc duoc frame")
    else:
        print(f"Camera index {i}: NOT FOUND")

    cap.release()

print("\nNeu co 2 camera OK thi thuong:")
print("0 = webcam laptop")
print("1 = webcam ngoai")
print("Neu khong chac, hay doi CAMERA_INDEX trong collect_gaze_data.py roi test.")