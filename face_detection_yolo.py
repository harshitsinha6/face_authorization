
import torch
import numpy as np
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'crowdhuman_yolov5m.pt', force_reload=True)


def get_face_locations(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    # print(results.pandas().xyxy[0], "\n\n")
    # print(f"===================================")
    w, h = 640, 480
    face_locations = []
    for result in results.xyxy[0]:
        if result[-1] == 0:
            continue
        x1, y1, x2, y2, conf, class_id = result
        # print(x1, y1, x2, y2, conf, class_id)
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        face_locations.append((int(x1), int(y1), int(x2), int(y2)))
    print(face_locations)
    return frame, face_locations

