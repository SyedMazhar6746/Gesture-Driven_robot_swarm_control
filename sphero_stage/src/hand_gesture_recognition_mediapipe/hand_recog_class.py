#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy
import argparse
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc

from model import KeyPointClassifier
from model import PointHistoryClassifier

from process_data_and_draw import select_mode, calc_bounding_rect, calc_landmark_list, pre_process_landmark, pre_process_point_history, logging_csv, draw_bounding_rect, draw_info_text, draw_point_history, draw_info, draw_landmarks

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    
    parser.add_argument("--no_of_hands", type=int, default=2)


    args = parser.parse_args()

    return args


class hand_landmark_recog:

    def __init__(self):
        # Argument parsing #################################################################
        
        self.landmark_list = []

        args = get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        no_of_hands = args.no_of_hands

        self.use_brect = True

        # Camera preparation ###############################################################
        self.cap = cv.VideoCapture(cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=no_of_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()

        self.point_history_classifier = PointHistoryClassifier()
        # Read labels ###########################################################

        # Get the directory path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Relative path to the CSV file
        keypoint_classifier_label_file_path = os.path.join(current_dir, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
        point_history_classifier_label_file_path = os.path.join(current_dir, 'model', 'point_history_classifier', 'point_history_classifier_label.csv')

        # with open('/home/syed_mazhar/c++_ws/src/aa_zagreb_repo/HRI_project/HRI-project/sphero_simulation-master/sphero_stage/src/hand_gesture_recognition_mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
        #         encoding='utf-8-sig') as f:
        with open(keypoint_classifier_label_file_path, encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [   # list of keypoint labels
                row[0] for row in self.keypoint_classifier_labels
            ]
        # with open(
        #         '/home/syed_mazhar/c++_ws/src/aa_zagreb_repo/HRI_project/HRI-project/sphero_simulation-master/sphero_stage/src/hand_gesture_recognition_mediapipe/model/point_history_classifier/point_history_classifier_label.csv',
        #         encoding='utf-8-sig') as f:
        with open(point_history_classifier_label_file_path, encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [   # list of point history labels
                row[0] for row in self.point_history_classifier_labels
            ]

        # FPS Measurement ########################################################
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # Finger gesture history ################################################
        self.finger_gesture_history = deque(maxlen=self.history_length)



    def run_recog(self):
        #  ########################################################################
        mode = 0

        while True:
            fps = self.cvFpsCalc.get()

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)  # for gathering data using k, h, number of labels, etc

            # Camera capture #####################################################
            ret, image = self.cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)
            # print('image size', debug_image.shape)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    
                    # print the landmark points in 3D for each point (Almost unuseful)
                    # print('hand_landmarks:', hand_landmarks)

                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    self.landmark_list = landmark_list

                    # print all the landmarks in pixels
                    # print('landmark_list:', landmark_list[8])
                    # print('landmark_list:', len(landmark_list))

                    # Conversion to relative coordinates with respect to palm point in meters or cm/ normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark( 
                        landmark_list)
                    
                    # if pre_processed_landmark_list is not None:
                    #     print('pre_processed_landmark_list:', pre_processed_landmark_list[8:10]) # len = 42 (21 * 2)

                    pre_processed_point_history_list = pre_process_point_history( # point history in meters or cms
                        debug_image, self.point_history) 
                    
                    # if pre_processed_point_history_list is not None: 
                    #     print('pre_processed_point_history_list:', pre_processed_point_history_list) # len = 32 (16 * 2) 2 * history length

                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        self.point_history.append(landmark_list[8])
                    else:
                        self.point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (self.history_length * 2):
                        finger_gesture_id = self.point_history_classifier(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    self.finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        self.finger_gesture_history).most_common()

                    # Drawing part
                    debug_image = draw_bounding_rect(self.use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)

                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        self.keypoint_classifier_labels[hand_sign_id],
                        self.point_history_classifier_labels[most_common_fg_id[0][0]],
                    )


            else:
                self.landmark_list = []
                self.point_history.append([0, 0])
            

            debug_image = draw_point_history(debug_image, self.point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)

            if self.landmark_list:
                yield self.landmark_list, self.keypoint_classifier_labels[hand_sign_id]
        
        self.cap.release()
        cv.destroyAllWindows()