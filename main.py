#Start /  Code für Windows

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
cap.set(3, 1280)
cap.set(4, 720)

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
        cv2.circle(frame, (center_x, center_y), 5,(255, 0, 0), -1)
        
    cv2.imshow("YOLO11n - Windows PyThorch Version", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyWindow()

#Ende

#Start / Code für Pi

# import time
# import cv2
# import serial
# import torch
# from picamera2 import Picamera2
# from ultralytics import YOLO

# arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# picam2 = Picamera2()
# picam2 = configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)}))
# picam2.start()

# model = YOLO("yolo11n.pt")

# prev_time = time.time()

# while True:
#     frame = picam2.capture_array()

#     results = model(frame, imgsz=640, conf=0.5)[0]

#     for box in results:
#         x1, y1, x2, y2 = map(int, box.boxes.xyxy[0].tolist())
#         center_x = int((x1 + x2) / 2)
#         center_y = int((y1 + y2) / 2)

#         try:
#             arduino.write(f"{center_x}, {center_y}\n".encode())
#         except Exception as ex:
#             print(f"[Serial Error] {ex}")
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.circle(frame, (center_x, center_y), 5,(255, 0, 0), -1)

#         cv2.imshow("Raspberry YOLO", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# picam2.stop()
# arduino.close()
# cv2.destroyWindow()

#Ende