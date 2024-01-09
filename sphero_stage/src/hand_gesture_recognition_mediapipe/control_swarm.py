#!/usr/bin/env python3

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

from utils.boid import Boid, pixels_to_meters

from utils.alignment_check import Alignment

class Robot_1_move:

    def publish_vel(self, frame_id, Total_velocity):
            Total_velocity_cmd = Twist()
            Total_velocity_cmd.linear.x = Total_velocity.x
            Total_velocity_cmd.linear.y = Total_velocity.y 

            self.cmd_vel_pub[frame_id].publish(Total_velocity_cmd)

    def odom_callback(self, msg):

        frame_id = msg.header.frame_id
        
        if len(frame_id) == 13: 
            id = frame_id[7] # string
            if frame_id not in self.cmd_vel_pub:
                self.cmd_vel_pub[frame_id] = rospy.Publisher("{}cmd_vel".format(frame_id[:9]), Twist, queue_size=10)
        else:
            id = frame_id[7:9] # string
            if frame_id not in self.cmd_vel_pub:
                self.cmd_vel_pub[frame_id] = rospy.Publisher("{}cmd_vel".format(frame_id[:10]), Twist, queue_size=10)


        self.agent= self.agents[int(id)]
        if any(self.targets):
            nav_velocity_v = self.agent.arrive(msg, self.targets[int(id)])

            Total_velocity = self.nav_velocity_v_weight*nav_velocity_v 
            Total_velocity.x += self.weighted_vel[int(id)][0]
            Total_velocity.y += self.weighted_vel[int(id)][1]

            self.publish_vel(frame_id, Total_velocity)



    def landmark_callback(self, msg):

        # Reshape the received flattened data into a 2D list
        rows = len(msg.data) // 2  # Assuming each sublist has 2 elements
        self.landmarks = [[msg.data[i*2], msg.data[i*2+1]] for i in range(rows)]

        converted_targets = []
        for point in self.landmarks:
            converted_point = pixels_to_meters(point)
            converted_targets.append(converted_point)
        
        self.targets = converted_targets

    def __init__(self, vel, max_speed, slowing_radius):
        self.num_of_robots = rospy.get_param("/num_of_robots")

        self.weighted_vel = [0]*self.num_of_robots
        self.weighted_vel = vel.weighted_velocities(2)

        self.max_speed = max_speed
        self.slowing_radius = slowing_radius
        
        self.targets = [0] * self.num_of_robots
        self.velocity = [0.0, 0.0]  # Initial velocity
        self.landmarks = [0] * 20
        self.cmd_vel_pub = {}
        self.nav_velocity_v_weight = 3.0

        self.agents = [Boid(initial_velocity_x=0.0, initial_velocity_y=0.0, max_speed_mag=self.max_speed, 
                            slowing_radius=self.slowing_radius) for _ in range(self.num_of_robots)]
        rospy.Subscriber('landmarks', Float64MultiArray, self.landmark_callback)
        rate = rospy.Rate(0.5)
        rate.sleep()
        
        [rospy.Subscriber("robot_{}/odom".format(i), Odometry, self.odom_callback) for i in range(self.num_of_robots)]

if __name__ == '__main__':

    try:
        rospy.init_node('rotate_robot_circularly')
        vel = Alignment()
        robot = Robot_1_move(vel, max_speed=0.6, slowing_radius=1.0)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass