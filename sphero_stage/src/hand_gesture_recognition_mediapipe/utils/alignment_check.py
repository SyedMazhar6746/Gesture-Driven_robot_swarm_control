#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import re
import tf2_ros
from nav_msgs.msg import Odometry 

class Alignment:


    def odom_callback(self, odom):
        # Define a regular expression pattern to match the number
        pattern = re.compile(r"/robot_(\d+)/odom")
        frame_id = odom.header.frame_id
        # Use the pattern to search for a match in the string
        match = pattern.search(frame_id)
        robot_num = int(match.group(1))
        # Store position and velocity of all robots
        p = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
        self.positions[robot_num] = p
        v = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y])
        self.velocities[robot_num] = v



    def neighbours(self, robot_num):
        # Find neighbours of robot_num
        neighbours = []
        for i in range(len(self.positions)):
            if i != robot_num:
                try:
                    # Get the transform from robot_num to robot i
                    transform = self.tfBuffer.lookup_transform("robot_{}/base_footprint".format(robot_num),
                                                                "robot_{}/base_footprint".format(i), rospy.Time(0),
                                                                  rospy.Duration(1.0))
                    # print("Transform from 'source_frame' to 'target_frame':", transform.transform.translation.x, transform.transform.translation.y)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print("Error:", e)
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                distance = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                if distance < self.neighbours_dist and (theta <= self.neighbours_angle
                                                         and theta >= -self.neighbours_angle):
                    neighbours.append(i)
        return neighbours
    

    def separation(self):
        # find the neighbours in the neighborhood and move away from them
        for i in range(self.num_of_robots):
            neighbours = self.neighbours(i)
            force = np.zeros(2)
            for neighbour in neighbours:
                difference = self.positions[i] - self.positions[neighbour]
                if np.linalg.norm(difference) < self.separation_dist:
                    force += (difference/max(np.linalg.norm(difference), 0.001))
            
            force /= max(1, len(neighbours))
            vx = force[0]
            vy = force[1]
            v = np.array([vx, vy])
            self.separation_vel[i] = v

    def weighted_velocities(self, event): 
        # compute the weighted velocities and publish them
        self.separation()
        for i in range(self.num_of_robots):
            self.weighted_vel[i] = (self.separation_vel_weights*self.separation_vel[i])
        return self.weighted_vel


    def __init__(self):
        self.num_of_robots = rospy.get_param("/num_of_robots")


        # Parameters for field of view
        self.neighbours_dist = 0.8# 1.2
        # self.neighbours_angle = 2*np.pi/3
        self.neighbours_angle = np.pi
        
        # separation distance to be maintained
        self.separation_dist = 0.1 # 0.1 works best

        # velocity vector weights for each behaviour
        self.separation_vel_weights = 10 # 10 works best

        # listen to the transforms between robots to compute their relative transformations
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # arrays to store robot postions and velocities
        self.positions = np.zeros((self.num_of_robots, 2))
        self.velocities = np.zeros((self.num_of_robots, 2))

        # arrays to store the velocities for each of the behaviours
        self.separation_vel = np.zeros((self.num_of_robots, 2))
        
        
        # subscribers to get the positions and velocities of each robot
        [rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.odom_callback)
          for i in range(self.num_of_robots)]
        

        rospy.Timer(rospy.Duration(0.05), self.weighted_velocities)
        self.weighted_vel = [0]*self.num_of_robots