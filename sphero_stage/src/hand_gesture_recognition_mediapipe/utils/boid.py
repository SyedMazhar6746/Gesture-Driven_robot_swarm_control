#!/usr/bin/env python3


import numpy as np
from geometry_msgs.msg import Twist

from utils.vector2 import Vector2, get_agent_position

class Boid(object):

    # use twist type for velocity and pose type message for position

    def __init__(self, initial_velocity_x, initial_velocity_y, max_speed_mag, slowing_radius): # , wait_count, start_count, frequency
        """Create an empty boid and update parameters."""
        self.position = Vector2()
        self.velocity = Vector2()
        self.max_speed_mag = max_speed_mag #
        self.slowing_radius = slowing_radius #

        # Set initial velocity
        self.initial_velocity = Twist()
        self.initial_velocity.linear.x = initial_velocity_x
        self.initial_velocity.linear.y = initial_velocity_y

    
    def arrive(self, agent_msg, target):

        target_v = Vector2(target[0], target[1])
        desired_velocity_v = Vector2()
        self.position_v = get_agent_position(agent_msg) # agent position

        target_offset_v = target_v - self.position_v
        distance = target_offset_v.norm() 
        
        ramped_speed = (distance / self.slowing_radius)
        
        if distance < 1e-3:
            # print(f'position reached')
            return Vector2()
        else:
            # tanh path
            # desired_velocity_v.x = np.tanh((ramped_speed / distance) * target_offset_v.x )
            # desired_velocity_v.y = np.tanh((ramped_speed / distance) * target_offset_v.y )

            # Straight path
            desired_velocity_v.x = (ramped_speed / distance) * target_offset_v.x 
            desired_velocity_v.y = (ramped_speed / distance) * target_offset_v.y 

            if target_offset_v.norm() > self.max_speed_mag:
                desired_velocity_v.set_mag(self.max_speed_mag)

            return desired_velocity_v



    def meters_to_pixels(self, meter_coordinates, map_size=200, map_range=10.0): # meter_coordinates = [x, y]
        
        # Calculate the middle pixel (center of the map)
        middle_pixel = map_size // 2
        
        # Calculate the pixel scale per meter
        pixels_per_meter = map_size / map_range
        
        # Convert meters to pixels
        x_meters, y_meters = meter_coordinates
        x_pixels = int((x_meters  * pixels_per_meter)+ middle_pixel)
        y_pixels = int(-(y_meters  * pixels_per_meter) + middle_pixel)

        return [y_pixels, x_pixels]
    
def pixels_to_meters(pixel_coordinates, map_size=200, map_range=10.0):
    # Calculate the middle pixel (center of the map)
    middle_pixel = map_size // 2
    
    # Calculate the meters per pixel
    meters_per_pixel = map_range / map_size
    
    # Convert pixels to meters
    y_pixels, x_pixels = pixel_coordinates
    y_meters = (middle_pixel - y_pixels) * meters_per_pixel
    # x_meters = (x_pixels - middle_pixel) * meters_per_pixel
    x_meters = -(x_pixels - middle_pixel) * meters_per_pixel
    
    x_meter = -y_meters
    y_meter = x_meters

    return [x_meter, y_meter]

