from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd 

# Tracker class for object detection and tracking
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox":x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20 
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15 
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect+12 
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame, 
                f"{track_id}", 
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 
                (0, 0, 0), 
                2
            )

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y], 
            [x - 10, y - 20], 
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = [] 
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() 

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players 
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))
            
            # Draw referees 
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control 
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

"""
Tracker Class for Object Detection and Tracking
===============================================

This class uses the YOLO model for detecting objects in frames and tracks their movement using the ByteTrack tracker.
Additionally, it provides utilities to annotate video frames with ellipses and triangles to represent detected objects.

Dependencies:
- ultralytics.YOLO: For object detection.
- supervision.ByteTrack: For tracking objects across frames.
- cv2 (OpenCV): For frame processing and annotation.
- pickle: For saving/loading tracking data.
- numpy: For numerical operations and annotations.

Methods:
--------
1. __init__(self, model_path): 
   Initializes the tracker with a YOLO model loaded from the specified path.

2. detect_frames(self, frames): 
   Detects objects in batches of frames using the YOLO model and returns the detections.

3. get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
   Tracks objects (players, referees, ball) across frames and optionally saves/loads tracks to/from a pickle file.

4. draw_ellipse(self, frame, bbox, color, track_id=None):
   Draws an ellipse to represent a player/referee and optionally adds a track ID label.

5. draw_triangle(self, frame, bbox, color):
   Draws a triangle to represent the detected ball.

6. draw_annotations(self, video_frames, tracks):
   Draws annotations (ellipses and triangles) for players, referees, and ball on a sequence of video frames.

Key Features:
-------------
- Detection: Uses the YOLO model for object detection.
- Tracking: Tracks detected objects (players, referees, and ball) across frames using ByteTrack.
- Annotation: Draws annotated overlays (ellipses for players/referees, triangles for ball) on video frames.
- Customization: Supports team-specific colors for players.
- Efficiency: Handles batch processing for detection and optional stub-based track persistence.

Detailed Comments:
------------------

1. detect_frames():
   - Divides frames into batches of size `batch_size` (default: 20) to optimize the YOLO model's inference.
   - Uses the model's `predict()` method with a confidence threshold (`conf=0.1`) for detections.

2. get_object_tracks():
   - Reads precomputed tracking data from a stub file if `read_from_stub` is True and `stub_path` exists.
   - Converts detections to the supervision `Detections` format.
   - Tracks players, referees, and the ball separately.
   - Saves the generated tracking data to a stub file if a path is provided.

3. draw_ellipse():
   - Draws an ellipse at the bottom-center of the bounding box (bbox) to represent the tracked object.
   - Optionally adds a track ID label above the ellipse.

4. draw_triangle():
   - Draws a filled triangle at the top-center of the bounding box to represent the detected ball.

5. draw_annotations():
   - Iterates through video frames and applies `draw_ellipse` and `draw_triangle` to annotate players, referees, and ball.

Edge Cases Handled:
-------------------
- Goalkeeper class: Reassigns "goalkeeper" to "player" for uniformity in tracking.
- Missing detection: Uses empty dictionaries for missing detections of players, referees, or ball in specific frames.

Notes:
------
- YOLO model is trained to detect required classes (e.g., "player", "referee", "ball").
- Adjust confidence thresholds and parameters (e.g., `conf`) for specific use cases.
"""
