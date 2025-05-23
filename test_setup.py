import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("All Libraries Imported Successfully! :)")

video_path = "C:\\Users\\IzzatHamizan\\Documents\\WorkoutFormChecker\\Exercise-Form-Checker-main\\Assets\\test_video.mp4"

print("Checking OpenCV window...")

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Test", 640, 480)
cv2.moveWindow("Test", 100, 100)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video or error reading frame.")
        break

    cv2.imshow("Test", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
