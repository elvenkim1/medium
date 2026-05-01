from ultralytics import YOLO
import cv2
import time
import pyttsx3

# Load YOLO model
model = YOLO("yolov8n.pt")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

TARGET_OBJECT = "apple"

apple_was_detected = False
is_speaking = False
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, imgsz=320, conf=0.25, verbose=False)

    detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == TARGET_OBJECT:
                detected = True

    # Draw boxes
    annotated_frame = results[0].plot()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Speak safely (NO threading)
    if detected and not apple_was_detected and not is_speaking:
        is_speaking = True
        engine.say("apple")
        engine.runAndWait()
        is_speaking = False
        apple_was_detected = True

    if not detected:
        apple_was_detected = False

    cv2.imshow("YOLO Voice Safe", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()