from ultralytics import YOLO
import cv2
import winsound
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

TARGET_OBJECT = "person"   # change to "person", "cell phone", etc.
last_beep_time = 0
BEEP_COOLDOWN = 2         # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=320, conf=0.25, verbose=False)

    detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == TARGET_OBJECT:
                detected = True

    annotated = results[0].plot()

    # beep without repeating too fast
    now = time.time()
    if detected and now - last_beep_time > BEEP_COOLDOWN:
        winsound.Beep(1000, 300)  # frequency, duration ms
        last_beep_time = now

    cv2.imshow("YOLO Detection with Sound", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()