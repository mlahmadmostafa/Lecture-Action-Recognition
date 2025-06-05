from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

from yololstm import YOLO_LSTM
import torch
import pandas as pd
from collections import defaultdict, deque
import numpy as np
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from IPython.display import clear_output

df_action = pd.read_csv('../Dataset_2/class_number_map.csv').drop('Unnamed: 0', axis=1)

yolo_model_action = YOLO('../Models/yoloactiondata2.pt') 
yolo_model_human = YOLO('../Models/yolohuman.pt')
backbone = yolo_model_action.model.model[:10]  # nn.ModuleList
yolo_lstm_action = YOLO_LSTM(yolo_backbone=backbone, hidden_size = 512, num_classes=7)
yolo_lstm_action.load_state_dict(torch.load('../Models/best_yololstm2.pth'))
yolo_lstm_action.eval()


yolo_model_action = yolo_model_action.to('cuda')
yolo_model_human = yolo_model_human.to('cuda')
yolo_lstm_action = yolo_lstm_action.to('cuda')




tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(0) # change to for example: "../test_images/exam2.mp4" or the number of your webcam
frame_buffer = defaultdict(lambda: deque(maxlen=8))
track_action_map = {}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    results = yolo_model_human.predict(source=frame, conf=0.3)[0]
    detections = []
    results = results.to("cuda")
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf.item()
        cls = int(box.cls.item())
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Store cropped person
        person_crop = frame[y1:y2, x1:x2]
        frame_buffer[track_id].append(person_crop)

        # If 8 frames collected, run action classification
        if len(frame_buffer[track_id]) == 8:
            clips = []
            for f in frame_buffer[track_id]:
                if f.size == 0:
                    continue
                resized = cv2.resize(f, (640, 640))
                clips.append(resized)

            if len(clips) < 8:
                continue  # skip if not enough valid frames
            clips = np.stack(clips).transpose(0, 3, 1, 2)  # (8, 3, 640, 640)
            clips = torch.tensor(clips, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, 8, 3, 640, 640)
            clips = clips.to("cuda")
            with torch.no_grad():
                pred = yolo_lstm_action(clips)
                class_id = pred.argmax(dim=1).item()
                track_action_map[track_id] = class_id  # store

        # Get action label
        class_id = track_action_map.get(track_id, None)
        if class_id is not None:
            action_class = df_action.loc[df_action['number'] == class_id, 'class'].values
            action_class = action_class[0] if len(action_class) > 0 else "Unknown"
        else:
            action_class = "Loading..."

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        # Draw label
        label = f'ID {track_id}: {action_class}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (255,0,0), -1)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Tracking + Action", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    clear_output(wait=True)

cap.release()
cv2.destroyAllWindows()
