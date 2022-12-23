import random

import tensorflow as tf

import numpy as np
import cv2

import os
import time
from datetime import datetime
import csv
import mediapipe as mp

from tqdm import tqdm

from PIL import Image

AMOUNT_HAND_LANDMARKS = 21
AMOUNT_BODY_LANDMARKS = 33
NUM_POINTS = (2*AMOUNT_HAND_LANDMARKS + AMOUNT_BODY_LANDMARKS)*3
SEQUENCE_LENGTH = 25
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0']


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


class CSVReader():
    def csv_read(self, file_path, val=0.0):
        self.file_path = file_path

        self.data = np.genfromtxt(file_path, delimiter=';')

        self.labels = self.data[:, 0].astype(np.uint8)

        self.data = self.data[:, 1:(SEQUENCE_LENGTH * NUM_POINTS + 1)]

        self.data = self.data.reshape((self.data.shape[0], SEQUENCE_LENGTH, NUM_POINTS), order='C')

        print('Batch Size: ' + str(self.data.shape[0]) + ', Sequence Length: ' + str(self.data.shape[1]) +
              ', Feature Size: ' + str(self.data.shape[2]))

        val_size = int(val * self.data.shape[0])

        self.val_data = np.zeros((val_size,SEQUENCE_LENGTH,NUM_POINTS))
        self.val_labels = np.zeros((val_size,))

        for i in range(val_size):
            idx = np.random.randint(self.data.shape[0])
            self.val_data[i,:,:] = self.data[idx, :, :]
            self.val_labels[i] = self.labels[idx]
            self.data = np.delete(self.data, idx, 0)
            self.labels = np.delete(self.labels,idx)
        return self.labels, self.data, self.val_data, self.val_labels

    def augment_csv_random_zero(self, percentage=20):
        if percentage > 100:
            percentage = 100

        amount = int(float(percentage)/100 * self.data.shape[0])
        length = int(SEQUENCE_LENGTH / 5)

        for k in range(amount):
            idx = np.random.randint(self.data.shape[0])

            frame_amount = np.random.randint(length)

            frames_idx = np.random.randint(SEQUENCE_LENGTH-frame_amount)

            tmp = self.data[idx,:,:]
            tmp[frames_idx:(frames_idx+frame_amount),:] = 0
            self.data = np.append(self.data, tmp)
            self.labels = np.append(self.labels, self.labels[idx])

    def get_training_data(self):
        print('Training Data:', self.data.shape[0])
        return self.data, self.labels

    def get_validation_data(self):
        print('Validation Data:', self.val_data.shape[0])
        return self.val_data, self.val_labels

    def clear(self):
        self.file_path = ''
        self.data = np.ndarray((0,))
        self.labels = np.ndarray((0,))
        self.val_data = np.ndarray((0,))
        self.val_labels = np.ndarray((0,))

    def __init__(self):
        self.data = np.ndarray((0,))
        self.labels = np.ndarray((0,))
        self.val_data = np.ndarray((0,))
        self.val_labels = np.ndarray((0,))
        self.file_path = ''


class SkeletonProcessor:
    def annotate(self, frame, result):
        annotated = frame.copy()

        mp_drawing.draw_landmarks(
            annotated,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            annotated,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        mp_drawing.draw_landmarks(
            annotated,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        return annotated

    def process_single(self, results):
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(AMOUNT_BODY_LANDMARKS*3)

        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(AMOUNT_HAND_LANDMARKS*3)

        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(AMOUNT_HAND_LANDMARKS*3)
        return np.concatenate([pose, lh, rh])

    def process_sequence(self, sequence):
        isValid = True
        skeleton_sequence = []
        annotated_sequence = []

        with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                smooth_landmarks=True,
                model_complexity=1,
                min_tracking_confidence=0.5) as holistic:
            for k in tqdm(range(len(sequence)), desc="Skeleton Processing"):
                frame = sequence[k]
                results = holistic.process(frame)

                vec = self.process_single(results)

                skeleton_sequence.append(vec)
                if self.annotate_sequence:
                    annotated_sequence.append(self.annotate(frame, results))

        return isValid, skeleton_sequence, annotated_sequence

    def initialize(self, annotate_sequence=True):
        self.annotate_sequence = annotate_sequence

    def __init__(self, annotate_sequence=True):
        self.annotate_sequence = annotate_sequence


class SequencePreprocessor:
    def reset(self):
        self.used_only_flip = False

    def augment(self, seq):
        if self.flip:
            if self.last_flip:
                f = False
            else:
                f = True
            self.last_flip = f
        else:
            f = False

        rot = int(self.random_param(self.rotate))
        trans_x = self.random_param(self.translate_x)
        trans_y = self.random_param(self.translate_y)

        print('Flip: ' + str(f) + ', Rotation: ' + str(rot) + ', Translation: '
              + f'{trans_y*100:.0f}%' + '/' + f'{trans_x*100:.0f}%')
        time.sleep(0.05)

        out = []
        for k in tqdm(range(len(seq)),  desc="Sequence Augmentation"):
            frame = seq[k]
            if f:
                frame = self.mirror_frame(frame)
            frame = self.rotate_frame(frame, rot)
            frame = self.translate_frame(frame, trans_x, trans_y)

            out.append(frame)

        return out

    def mirror_frame(self, frame):
        return cv2.flip(frame, 1)

    def rotate_frame(self, frame, val):
        img = Image.fromarray(frame)
        out = np.array(img.rotate(val))

        return out

    def translate_frame(self, frame, val_x, val_y):
        height = frame.shape[0]
        width = frame.shape[1]
        hT = height * val_y
        wT = width * val_x
        T = np.float32([[1, 0, wT], [0, 1, hT]])

        return cv2.warpAffine(frame, T, (width, height))

    def random_param(self, param):
        return (np.random.random_sample()-0.5) * param * 2

    def initialize(self, scale=0.0, rotate=0, translate=0.0, flip=False):
        self.scale = scale
        self.rotate = rotate
        self.translate_x = translate
        self.translate_y = translate
        self.flip = flip

    def __init__(self, scale=0, rotate=0, translate=0.0, flip=False):
        self.scale = 0.0
        self.rotate = 0
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.flip = False
        self.last_flip = False
        self.used_only_flip = False
        self.initialize(scale=scale, rotate=rotate, translate=translate, flip=flip)


class SequenceProcessor:
    def get_video_info(self):
        return self.video_fps, self.video_length

    def set_sequence_limits(self, value):
        value1 = value[0]
        value2 = value[1]
        if self.video_loaded:
            if value2 >= value1:
                self.f2 = int(value2 * self.video_fps)
                self.f1 = int(value1 * self.video_fps)
            else:
                self.f1 = int(value2 * self.video_fps)
                self.f2 = int(value1 * self.video_fps)

            if self.f1 < 0:
                self.f1 = 0
            if self.f2 >= len(self.video_frames):
                self.f2 = len(self.video_frames)-1

            self.f_length = self.f2 - self.f1 + 1
            return 1, self.video_frames[self.f1], self.video_frames[self.f2]
        else:
            return 0, np.zeros((1, 1)), np.zeros((1, 1))

    def set_export_options(self, vs, seq, aug):
        self.export_video = vs[0]
        self.export_skeleton = vs[1]
        self.annotate_video = vs[1]

        ## self.exported_sequence_length = seq[0]
        self.amount_sequences = seq[1]
        self.augmentation = aug[0]

        self.amount_augmentations = aug[1]

    def load_video(self, file_path):
        if file_path == '':
            return False
        self.video_path = file_path
        file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.gesture_class = file_name[0]

        cap = cv2.VideoCapture(self.video_path)

        self.video_frames = []
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_size = (frame_width, frame_height)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.video_frames.append(image)

        if len(self.video_frames) == 0:
            return False
        else:
            self.video_length = float(len(self.video_frames) / self.video_fps)

        self.video_loaded = True
        print('Finished loading video' + file_name)
        return self.video_loaded, self.gesture_class

    def set_output_path(self, file_path):
        if file_path == '':
            return 0
        else:
            self.output_path = file_path
            return 1

    def select_whole_sequence(self):
        self.f1 = 0
        self.f2 = len(self.video_frames)-1

    def initialize_random_sequence(self, input_len, seed=0, step=1., reverse=0):
        deltaSeq = float(step)*self.exported_sequence_length
        if self.whole_sequence:
            return np.linspace(0, input_len-1, self.exported_sequence_length, dtype=int), True
        elif reverse == 0 and seed + deltaSeq < input_len-1:
            return np.linspace(seed, seed + deltaSeq, self.exported_sequence_length, dtype=int), True
        elif reverse == 1 and seed - deltaSeq > 0:
            return np.linspace(seed, seed - deltaSeq, self.exported_sequence_length, dtype=int), True
        else:
            return np.zeros((1, 1)), False

    def get_skeleton_sequence(self, skeleton_seq, idx):
        out = []
        for k in range(self.exported_sequence_length):
            out.append(skeleton_seq[idx[k]])
        return out

    def generate_skeleton_sequence(self, seq, step=1, seed=0, reverse=0):

        idx, isValid = self.initialize_random_sequence(len(seq), step=step, seed=seed, reverse=reverse)

        if isValid:
            skel_seq = self.get_skeleton_sequence(seq, idx)
            return 1, skel_seq
        else:
            return 0, []

    def get_write_vec(self, raw_skeleton_seq):
        write_data = np.zeros(raw_skeleton_seq.shape[0] * raw_skeleton_seq.shape[1] + 1)
        write_data[0] = self.class_label
        write_data[1:] = np.array(raw_skeleton_seq.flatten(order='C'))

        return write_data

    def export_data(self):
        if (not self.video_loaded) or self.output_path == '':
            return 0
        else:
            if self.export_video:
                cur_path = os.path.join(self.output_path, self.gesture_class)
                if not os.path.exists(cur_path):
                    os.mkdir(cur_path)
                now = datetime.now()
                current_time = now.strftime("%Y%m%d_%H%M%S")
                save_file_name = self.gesture_class + current_time
                print('Export video: ' + save_file_name)

                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(cur_path + '/' + save_file_name + '.mp4', fourcc, self.video_fps, self.video_frame_size)
                print('Seqlen:', len(self.video_frames))
                seq = self.video_frames[self.f1:self.f2]
                self.skeleton_processor.initialize(annotate_sequence=True)
                if self.annotate_video:
                    print('')
                    print('********************************************************')
                    print('Exporting Skeleton Video:')
                    _, _, seq = self.skeleton_processor.process_sequence(seq)

                for k in range(len(seq)):
                    out.write(cv2.cvtColor(seq[k], cv2.COLOR_BGR2RGB))

            if self.export_skeleton:
                self.class_label = classes.index(self.gesture_class.upper())
                csv_file = self.output_path + '/data.csv'

                self.preprocessor.reset()

                if not os.path.exists(self.output_path + '/KeyPoints'):
                    os.mkdir(self.output_path + '/' + 'KeyPoints')
                if not os.path.exists(self.output_path + '/KeyPoints/' + self.gesture_class):
                    os.mkdir(self.output_path + '/KeyPoints/' + self.gesture_class)


                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    raw_seq = self.video_frames[self.f1:self.f2+1]
                    self.skeleton_processor.initialize(annotate_sequence=False)
                    if len(raw_seq) < self.exported_sequence_length:
                        print('Selected Sequence shorter than specified export sequence length')
                        return 0

                    if self.augmentation:
                        num = self.amount_augmentations + 1
                    else:
                        num = 1

                    for i in range(num):
                        sequence = []
                        if i == 0:
                            print('')
                            print('********************************************************')
                            print('Processing Raw Video Sequence (Unaugmented):')
                            sequence = raw_seq
                        else:
                            print('')
                            print('********************************************************')
                            print('Processing Augmentation ' + str(i) + ':')
                            sequence = self.preprocessor.augment(raw_seq)

                        cnt = 0
                        _, skeleton, _ = self.skeleton_processor.process_sequence(sequence)
                        if self.whole_sequence:
                            self.amount_sequences=1

                        for j in range(self.amount_sequences):
                            if j == 0:
                                ok, seq = self.generate_skeleton_sequence(skeleton)
                            else:
                                step = np.random.random_sample() * (2. - 0.5) + 0.5
                                ok, seq = self.generate_skeleton_sequence(skeleton, step=step,
                                                                          seed=np.random.randint(len(skeleton)-step*self.exported_sequence_length),
                                                                          reverse=0)

                            if ok:
                                now = datetime.now()
                                current_time = now.strftime("%Y%m%d_%H%M%S")
                                save_folder = self.gesture_class + current_time + '_' + str(j)
                                save_dir = self.output_path + '/Keypoints/' + self.gesture_class + '/' + save_folder
                                if not os.path.exists(save_dir):
                                    os.mkdir(save_dir)

                                for k in range(len(seq)):
                                    npy_path = os.path.join(save_dir, str(k))
                                    np.save(npy_path, seq[k])

                                #writer.writerow(self.get_write_vec(np.stack(seq)))
            return 1

    def __init__(self, sequence_length=SEQUENCE_LENGTH, amount_sequences=2, amount_augmentations=10, whole_sequence = False):
        self.video_loaded = False
        self.video_frames = []
        self.video_fps = 0
        self.video_frame_size = (0, 0)
        self.video_length = 1.0
        self.video_path = ''

        self.preprocessor = SequencePreprocessor(rotate=35, translate=0.3, flip=True)
        self.skeleton_processor = SkeletonProcessor(annotate_sequence=True)

        self.gesture_class = ''
        self.whole_sequence = whole_sequence
        self.class_label = 0

        self.output_path = ''

        self.f1 = 0
        self.f2 = 0
        self.f_length = 30


        self.annotate_video = True ## unused
        self.export_video = True
        self.augmentation = True

        self.exported_sequence_length = sequence_length
        self.amount_sequences = amount_sequences
        if amount_sequences > 0 and self.exported_sequence_length > 0:
            self.export_skeleton = True
        else:
            self.export_skeleton = False

        self.amount_augmentations = amount_augmentations

        if self.amount_augmentations > 0 and self.export_skeleton:
            self.augmentation = True
        else:
            self.augmentation = False



