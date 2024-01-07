#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

from hand_recog_class import hand_landmark_recog  # Update the class name according to your class
from gesture_manipulation import rescale_and_equidistant_points # for manippulation of the gestures


class run_hand_recognition():
    def __init__(self, hand_recognizer):
        self.hand_recognizer = hand_recognizer
        self.msg = Float64MultiArray()
        self.pub = rospy.Publisher('landmarks', Float64MultiArray, queue_size=10)

    def publish_points(self, equidistant_points):
        flattened_array = np.array(equidistant_points).flatten()
        self.msg.data = flattened_array.tolist()
        self.pub.publish(self.msg)

    def receive_landmarks(self):
        while not rospy.is_shutdown():

            for landmark_list, symbol in hand_recognizer.run_recog(): # landmark_list is a list of all landmarks in pixels

                equidistant_points = rescale_and_equidistant_points(landmark_list)
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