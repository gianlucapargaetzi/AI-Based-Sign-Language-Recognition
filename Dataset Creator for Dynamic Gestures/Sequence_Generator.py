import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import shutil
import csv
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import os
import time


font                   = cv2.FONT_HERSHEY_SIMPLEX
position               = (15,70)
position2              = (15, 130)
fontScale              = 2
fontColorSel           = (0,255,0)
fontColorRec           = (0,0,255)
fontColorFixed         = (255,0,0)
fontThickness          = 5

root = tk.Tk()
root.withdraw()

SEQUENCE_LENGTH = 45

class SequenceRecorder:
    def open(self, capture_stream=1, frame_size=(1280, 720)):
        last_file = ''
        self.file_path = filedialog.askdirectory(title='Select Output Folder')

        if self.file_path == '':
            print('No directory selected! Abort Recorder ...')
            return ''
        else:
            print('Recorder output directory: ', self.file_path)

        self.cap = cv2.VideoCapture(capture_stream)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # self.out = cv2.VideoWriter('cv2_camera_output.mp4', fourcc, 25.0, (1920, 1080))

        cur_path = ''
        cnt = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            key = cv2.waitKey(1) & 0xFF

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return ''

            frame = cv2.flip(frame, 1)

            cam_frame = frame.copy()

            if (key >= 0x61) and (key <= 0x7A) and not self.recording:
                self.key_class = chr(key).upper()
                """
                if self.key_class == 'Z' or self.key_class == 'J':
                    self.fixed_video_length = True
                else:
                    self.fixed_video_length = False
                """
                cur_path = os.path.join(self.file_path, self.key_class)
                if not os.path.exists(cur_path):
                    os.mkdir(cur_path)
                print('Selected letter: ', self.key_class)

            elif key == 0x20 and self.key_class != '':
                if self.recording and not self.fixed_video_length:
                    self.recording = False
                    self.out.release()

                    print('Sequence "' + self.file_name + '"saved!')
                    last_file = cur_path + '/' + self.file_name + '.mp4'
                elif not self.recording:
                    self.recording = True
                    cnt = 0

                    print('Recording Sequence for letter "' + self.key_class + '" started!')
                    now = datetime.now()
                    current_time = now.strftime("%Y%m%d_%H%M%S")
                    self.file_name = self.key_class + current_time

                    self.out = cv2.VideoWriter(cur_path + '/' + self.file_name + '.mp4', fourcc, 25.0,
                                               (frame.shape[1], frame.shape[0]))
            elif key % 256 == 27 and not self.recording:
                # ESC pressed
                print("Escape hit, closing recorder ...")
                break

            if self.recording:
                # write the flipped frame
                if self.fixed_video_length and cnt == self.seq_len:
                    self.recording = False
                    seq_len = 0
                    self.out.release()
                    print('Sequence "' + self.file_name + '"saved!')
                    last_file = cur_path + '/' + self.file_name + '.mp4'
                    self.seq_len = int((1.5*np.random.random() + 1)*SEQUENCE_LENGTH)
                else:
                    self.out.write(frame)
                    cnt += 1
                cv2.putText(cam_frame, self.key_class, position, font, fontScale, fontColorRec, thickness=fontThickness)

            elif self.fixed_video_length:
                cv2.putText(cam_frame, self.key_class + '; ' + str(self.seq_len), position, font, fontScale, fontColorFixed, thickness=fontThickness)
            else:
                cv2.putText(cam_frame, self.key_class, position, font, fontScale, fontColorSel,
                            thickness=fontThickness)
            cv2.imshow('Camera Input', cam_frame)

        self.close()
        return last_file

    def close(self):
        # Release everything if job is finished
        if self.cap != None:
            self.cap.release()
        if self.out != None:
            self.out.release()
        cv2.destroyAllWindows()

    def __init__(self, fixed_length=False):
        self.fixed_video_length = fixed_length
        self.file_path = ''
        self.key_class = ''
        self.recording = False
        self.file_name = ''
        self.cap = None
        self.out = None
        self.seq_len = int(1.5*SEQUENCE_LENGTH)



seqgen = SequenceRecorder(fixed_length=True)
l_f = seqgen.open()

print('File: ', l_f)