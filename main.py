import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

arduino = serial.Serial('COM3', 9600, timeout=1)

model = YOLO("yolo11n.pt")
model.conf = 0.5
model.classes = [0]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results= model(frame)

    for box in results:
        x1, y1, x2, y2 = map(int, box.boxes.xyxy[0].tolist())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        try:
            serial_data = f"{center_x}, {center_y} \n"
            arduino.write(serial_data.encode())
        except Exception as ex:
            print(f"[Serial Error] {ex}")
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    cv2.imshow("YOLO11n - Windows PyThorch Version", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyWindow()