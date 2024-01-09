#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

from hand_recog_without_face import hand_landmark_recog  # Update the class name according to your class
from gesture_manipulation import equidistant_points_func, rescale_landmark_resolution # for manippulation of the gestures


class run_hand_recognition():
    def __init__(self, hand_recognizer):
        self.hand_recognizer = hand_recognizer
        self.msg = Float64MultiArray()
        self.pub = rospy.Publisher('landmarks', Float64MultiArray, queue_size=10)

    def find_last_index_of_left_and_right(self, hand_history, landmark_list_history):
        # Convert the list to a NumPy array
        arr = np.array(hand_history)

        # Find the index of the last "Left"
        last_left_index = np.where(arr == "Left")[0][-1] if "Left" in arr else None
        last_right_index = np.where(arr == "Right")[0][-1] if "Right" in arr else None

        landmark_list_left = landmark_list_history[last_left_index]
        landmark_list_right = landmark_list_history[last_right_index]
        # print("Last Left Index:", last_left_index)
        # print("Last Right Index:", last_right_index)
        # print("landmark_list_left:", landmark_list_left)
        # print("landmark_list_right:", landmark_list_right)
        return landmark_list_left, landmark_list_right
    
    def publish_points(self, equidistant_points):
        flattened_array = np.array(equidistant_points).flatten()
        self.msg.data = flattened_array.tolist()
        self.pub.publish(self.msg)

    def receive_landmarks(self):
        while not rospy.is_shutdown():
            for landmark_list, symbol, point_history, hand_history, landmark_list_history in hand_recognizer.run_recog(): # landmark_list is a list of all landmarks in pixels
                
                if symbol=="Letter C" and "Left" in hand_history and "Right" in hand_history:
                    # print('land_mark', landmark_list_history)
                    landmark_list_left, landmark_list_right = self.find_last_index_of_left_and_right(hand_history, landmark_list_history)
                    landmark_left_and_right = landmark_list_left + landmark_list_right
                    rescaled_points = rescale_landmark_resolution(landmark_left_and_right) 
                    # print('rescaled_points', rescaled_points)
                    equidistant_points = equidistant_points_func(rescaled_points, symbol, hand_history)
                    self.publish_points(equidistant_points)
                    
                elif symbol == "Open" or symbol == "Letter C":
                    rescaled_points = rescale_landmark_resolution(landmark_list)
                    equidistant_points = equidistant_points_func(rescaled_points, symbol, hand_history)
                    self.publish_points(equidistant_points)

                elif symbol == "Pointer":
                    rescaled_points = rescale_landmark_resolution(point_history)
                    equidistant_points = equidistant_points_func(rescaled_points, symbol, hand_history)
                    is_any_nan = np.isnan(equidistant_points).any()
                    if not is_any_nan:
                        self.publish_points(equidistant_points)

                elif symbol == "Heart":
                    rescaled_points = rescale_landmark_resolution(landmark_list[7])
                    equidistant_points = equidistant_points_func(rescaled_points, symbol, hand_history)
                    self.publish_points(equidistant_points)
                
                elif symbol == "Closed":
                    rescaled_points = rescale_landmark_resolution(landmark_list[0])
                    equidistant_points = equidistant_points_func(rescaled_points, symbol, hand_history)
                    self.publish_points(equidistant_points)

if __name__ == '__main__':
    try:
        rospy.init_node('hand_recognition_node', anonymous=True)
        hand_recognizer = hand_landmark_recog()

        node = run_hand_recognition(hand_recognizer)
        node.receive_landmarks()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass