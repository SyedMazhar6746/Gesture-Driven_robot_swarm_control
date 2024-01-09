#!/usr/bin/python3

import os
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import face_recognition
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self):

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        # rospy.Subscriber('camera_feed_1', Image, self.callback)
        self.stream_pub = rospy.Publisher('camera_feed_1', Image, queue_size=10)

        self.publisher = rospy.Publisher('recognized_names', String, queue_size=10)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))

        images_folder = os.path.join(current_dir, 'face_model', 'train_images')

        # obama_image = face_recognition.load_image_file("/home/syed_mazhar/c++_ws/src/aa_zagreb_repo/HRI_project/face_recognition/Train/obama.png")
        obama_image = face_recognition.load_image_file(images_folder + "/obama.png")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        biden_image = face_recognition.load_image_file(images_folder + "/biden.png")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        ozge_yagiz_image = face_recognition.load_image_file(images_folder + "/ozge_yagiz.jpeg")
        ozge_yagiz_face_encoding = face_recognition.face_encodings(ozge_yagiz_image)[0]

        mazhar_image = face_recognition.load_image_file(images_folder + "/Mazhar.png")
        mazhar_face_encoding = face_recognition.face_encodings(mazhar_image)[0]

        self.known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding,
            ozge_yagiz_face_encoding,
            mazhar_face_encoding
        ]

        self.known_face_names = [
            "Barack Obama",
            "Joe Biden",
            "Ozge Yagiz",
            "Mazhar-Boss"
        ]

        print('Learned encoding for', len(self.known_face_encodings), 'images.')

    # def callback(self, data):
    #     self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def recognize_faces(self, continuous):

        # video_capture = cv2.VideoCapture(0)
        process_this_frame = True

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                image_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.stream_pub.publish(image_message)
            # frame = frame

            if process_this_frame:
                name = "Unknown"

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left * 4, bottom * 4 - 35), (right * 4, bottom * 4), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    flipped_name = name[::-1]
                    cv2.putText(frame, name, (left * 4 + 6, bottom * 4 - 6), font, 1.0, (255, 255, 255), 1)
                self.publisher.publish(name)

            if not continuous:
                process_this_frame = not process_this_frame
            frame = cv2.flip(frame, 1)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    try:
        rospy.init_node('face_recognition_node', anonymous=True)
        face_recognizer = FaceRecognizer()
        face_recognizer.recognize_faces(continuous=False) # if continuous detection is needed, set it to True

    except rospy.ROSInterruptException:
        pass
