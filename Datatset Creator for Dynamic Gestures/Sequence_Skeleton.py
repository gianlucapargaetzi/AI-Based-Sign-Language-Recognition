import cv2
import mediapipe as mp
from time import sleep
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import numpy as np

AMOUNT_HAND_LANDMARKS = 21
AMOUNT_BODY_LANDMARKS = 8

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def getPoseLandmarks(poseLandmarks):
    out = np.zeros((AMOUNT_BODY_LANDMARKS,3))
    out[0,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z])
    out[1,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z])
    out[2,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z])
    out[3,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z])
    out[4,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z])
    out[5,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z])
    out[6,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z])
    out[7,:] = np.array([poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y,
                        poseLandmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z])
    return out

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(title='Select Output Folder')

# Define the codec and create VideoWriter object


if file_path == '':
    print('No directory selected! Abort Program ...')
    quit()
else:
    print('Selected output directory: ',file_path)

# For webcam input:
cap = cv2.VideoCapture(file_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('test.mp4', fourcc, 25.0, (frame_width,  frame_height))

sequence_vec = []

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=1,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        out.write(image)
        
cap.release()
out.release()
cv2.destroyAllWindows()