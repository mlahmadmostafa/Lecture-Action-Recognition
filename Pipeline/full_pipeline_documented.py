"""
Full Pipeline for Real-time Human Action Recognition
=================================================

This script implements a complete pipeline for real-time human detection, tracking, and action recognition.
It combines YOLO for human detection, DeepSort for tracking, and a YOLO-LSTM hybrid model for action recognition.

Key Features:
- Real-time human detection and tracking
- Action recognition using temporal information
- GPU-accelerated processing
- Real-time visualization with bounding boxes and labels

Author: [Your Name]
Date: [Current Date]
"""

# Standard library imports
from collections import defaultdict, deque  # For efficient frame buffering and tracking

# Third-party imports
from ultralytics import YOLO  # YOLO implementation for detection
from deep_sort_realtime.deepsort_tracker import DeepSort  # Real-time object tracking
import cv2  # OpenCV for video processing
import torch  # PyTorch for deep learning operations
import pandas as pd  # For handling class mapping
import numpy as np  # For numerical operations
from IPython.display import clear_output  # For clearing output in Jupyter notebooks

# Load class mapping from CSV
# This maps numerical class IDs to human-readable action names
df_action = pd.read_csv('../Dataset_2/class_number_map.csv').drop('Unnamed: 0', axis=1)

# Initialize Models
# ================

# Load YOLO models for action recognition and human detection
yolo_model_action = YOLO('../Models/yoloactiondata2.pt')  # Action recognition model
yolo_model_human = YOLO('../Models/yolohuman.pt')  # Human detection model

# Extract backbone layers from action model for LSTM
backbone = yolo_model_action.model.model[:10]  # First 10 layers as feature extractor

# Initialize YOLO-LSTM model for action recognition
yolo_lstm_action = YOLO_LSTM(
    yolo_backbone=backbone,
    hidden_size=512,  # LSTM hidden layer size
    num_classes=7     # Number of action classes
)

# Load pre-trained weights for LSTM model
yolo_lstm_action.load_state_dict(torch.load('../Models/best_yololstm2.pth'))
yolo_lstm_action.eval()  # Set to evaluation mode

# Move models to GPU for faster processing
yolo_model_action = yolo_model_action.to('cuda')
yolo_model_human = yolo_model_human.to('cuda')
yolo_lstm_action = yolo_lstm_action.to('cuda')

# Initialize Tracking and Video Capture
# ====================================

# Initialize DeepSort tracker with maximum age of 30 frames
tracker = DeepSort(max_age=30)

# Initialize video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)  # Change to video file path if needed

# Initialize frame buffer for each tracked person
# Uses defaultdict to automatically create new buffers for new tracks
# Each buffer stores up to 8 frames (required for action recognition)
frame_buffer = defaultdict(lambda: deque(maxlen=8))

# Dictionary to store action predictions for each track
track_action_map = {}

# Main Processing Loop
# ===================
while True:
    # Read frame from video source
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))  # Resize to standard resolution
    if not ret:
        break

    # Human Detection
    # --------------
    # Detect humans in the frame with confidence threshold of 0.3
    results = yolo_model_human.predict(source=frame, conf=0.3)[0]
    detections = []
    results = results.to("cuda")  # Move results to GPU

    # Convert detections to DeepSort format
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert coordinates to integers
        conf = box.conf.item()  # Get confidence score
        cls = int(box.cls.item())  # Get class ID
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracks with new detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process each tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        # Get track information
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Store cropped person in frame buffer
        person_crop = frame[y1:y2, x1:x2]
        frame_buffer[track_id].append(person_crop)

        # Action Recognition
        # -----------------
        # Process action recognition when 8 frames are collected
        if len(frame_buffer[track_id]) == 8:
            clips = []
            for f in frame_buffer[track_id]:
                if f.size == 0:
                    continue
                resized = cv2.resize(f, (640, 640))  # Resize for model input
                clips.append(resized)

            # Skip if not enough valid frames
            if len(clips) < 8:
                continue

            # Prepare clips for model input
            clips = np.stack(clips).transpose(0, 3, 1, 2)  # (8, 3, 640, 640)
            clips = torch.tensor(clips, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize
            clips = clips.to("cuda")

            # Run action recognition
            with torch.no_grad():
                pred = yolo_lstm_action(clips)
                class_id = pred.argmax(dim=1).item()
                track_action_map[track_id] = class_id  # Store prediction

        # Get action label for display
        class_id = track_action_map.get(track_id, None)
        if class_id is not None:
            action_class = df_action.loc[df_action['number'] == class_id, 'class'].values
            action_class = action_class[0] if len(action_class) > 0 else "Unknown"
        else:
            action_class = "Loading..."

        # Visualization
        # ------------
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        # Draw label with background
        label = f'ID {track_id}: {action_class}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0], text_y + 5), (255,0,0), -1)
        cv2.putText(frame, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Display processed frame
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Tracking + Action", frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Clear output in Jupyter notebook
    clear_output(wait=True)

# Cleanup
# =======
cap.release()
cv2.destroyAllWindows() 