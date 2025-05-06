import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

# === Konfiguration ===
arduino = serial.Serial('COM3', 9600, timeout=1)
model = YOLO("yolo11n.pt")
model.conf = 0.5
model.classes = [0]

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

move_threshold = 20  # Minimale X-Abweichung in Pixeln, bevor Daten gesendet werden
prev_positions = {}
last_sent_x = None  # Letzte gesendete X-Position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    current_positions = {}

    for i, box in enumerate(results.boxes):
        if box.conf[0] < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        prev_center = prev_positions.get(i)
        if prev_center:
            dx = abs(center_x - prev_center[0])
            dy = abs(center_y - prev_center[1])

            # Bewegung erkannt?
            if dx > 10 or dy > 10:
                if last_sent_x is None or abs(center_x - last_sent_x) <= move_threshold:
                    try:
                        serial_data = f"{center_x}, {center_y}\n"
                        arduino.write(serial_data.encode())
                        last_sent_x = center_x
                    except Exception as ex:
                        print(f"[Serial Error] {ex}")

        current_positions[i] = (center_x, center_y)

        # Bounding Box und Mittelpunkt anzeigen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        label = f"Person {i+1}: X={center_x}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        print(f"Person {i+1}: X-Koordinate = {center_x}")

    prev_positions = current_positions

    cv2.imshow("YOLO11n - Mehrere Personen + Bewegung", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
