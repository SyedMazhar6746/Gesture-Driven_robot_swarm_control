#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

# Define the points: 
# heart
# x = np.array([4, 2.0, 1, 4, 7, 6, 4])
# y = np.array([0, 2.0, 5, 4, 5, 2, 0])

# Letter C

# x = np.array([4, 0, 0, 4])
# y = np.array([0, 0, 4, 4])

def rescale_landmark_resolution(landmark_list): #No further changes required
    landmark_array = np.array(landmark_list)

    # Original pixel resolution
    original_resolution = (640, 480)
    # New scale
    new_resolution = (200, 200)

    # Calculate scaling factors for x and y dimensions
    scale_factors = (new_resolution[0] / original_resolution[0], new_resolution[1] / original_resolution[1])

    # Scale the hand landmarks to the new resolution
    scaled_hand_landmarks = (landmark_array * scale_factors).astype(int)

    return list(scaled_hand_landmarks)

def draw_curve_and_points(x, y): # No further changes required
    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=3, s=0.2) for coords in [x, y]]

    # Compute the spline for the asked distances:
    alpha = np.linspace(0, 1, 1000)  # Increase the number of points for more accuracy
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T

    # Get 20 equidistant points on the curve
    equidistant_indices = np.linspace(0, len(points_fitted) - 1, 20, dtype=int)
    equidistant_points = points_fitted[equidistant_indices]
    return points_fitted, equidistant_points


def Letter_c(hand_landmarks):
    # actual letter C
    # hand_landmarks = np.array([[469, 340], [408, 318], 
    #                         [359, 290], [315, 287], [277, 290], 
    #                         [401, 196], [362, 161], [326, 154], [294, 157], 
    #                         [414, 193], [359, 152], [319, 151], [286, 161], 
    #                         [424, 202], [370, 161], [326, 157], [290, 163], 
    #                         [424, 220], [376, 192], [341, 179], [311, 171]])

    # Given order
    order = [4, 3, 2, 5, 6, 7, 8]

    # Extracting points in the given order
    ordered_points = [hand_landmarks[idx] for idx in order]

    # Separating x and y points
    x = np.array([point[0] for point in ordered_points])
    y = np.array([point[1] for point in ordered_points])
    return x, y



def rescale_and_equidistant_points(hand_landmarks):
    scaled_landmark = rescale_landmark_resolution(hand_landmarks)
    x, y = Letter_c(scaled_landmark)
    points_fitted, equidistant_points = draw_curve_and_points(x, y)
    return equidistant_points

    # print(equidistant_points)
    # plot(x, y, points_fitted, equidistant_points)



















def plot(x, y, points_fitted, equidistant_points): # No further changes required
    # Graph:
    plt.plot(x, y, 'ok', label='original points')
    plt.plot(points_fitted[:, 0], points_fitted[:, 1], '-r', label='fitted spline k=3, s=0.2')
    plt.scatter(equidistant_points[:, 0], equidistant_points[:, 1], color='red', label='Equidistant points')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

