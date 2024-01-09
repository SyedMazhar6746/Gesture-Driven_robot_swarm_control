#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

from hand_recog_class import hand_landmark_recog  # Update the class name according to your class
from gesture_manipulation import equidistant_points_func, rescale_landmark_resolution # for manippulation of the gestures


class run_hand_recognition():
    def __init__(self, hand_recognizer):
        self.hand_recognizer = hand_recognizer

        self.bridge = CvBridge()
        rospy.Subscriber('camera_feed_1', Image, self.callback_stream)
        rate = rospy.Rate(10)
        rate.sleep()
        self.person_name = "unknown"
        self.actual_name = "Mazhar-Boss"   # change the name according to face recognition
        self.person_name_history = deque(maxlen=3)
        # self.person_name = "Mazhar-Boss"
        self.msg = Float64MultiArray()
        self.pub = rospy.Publisher('landmarks', Float64MultiArray, queue_size=10)
        self.subs = rospy.Subscriber('recognized_names', String, self.callback)

    def callback_stream(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def publish_points(self, equidistant_points):
        flattened_array = np.array(equidistant_points).flatten()
        self.msg.data = flattened_array.tolist()
        self.pub.publish(self.msg)

    def callback(self, msg):
        self.person_name_history.append(msg.data)
        if self.actual_name in self.person_name_history:
            self.person_name = self.actual_name
        else:
            self.person_name = "unknown"
        print("person", self.person_name)

    def receive_landmarks(self):
        while not rospy.is_shutdown():
            for landmark_list, symbol in hand_recognizer.run_recog(self.cv_image, self.person_name): # landmark_list is a list of all landmarks in pixels
                # print('hand', symbol)
                rescaled_points = rescale_landmark_resolution(landmark_list)
                if symbol == "Open":
                    equidistant_points = equidistant_points_func(rescaled_points, symbol)
                    self.publish_points(equidistant_points)

                elif symbol == "Letter C":
                    equidistant_points = equidistant_points_func(rescaled_points, symbol)
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