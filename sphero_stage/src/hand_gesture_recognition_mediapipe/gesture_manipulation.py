#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

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
    equidistant_indices = np.linspace(0, len(points_fitted) - 1, 21, dtype=int)
    equidistant_points = points_fitted[equidistant_indices]
    return points_fitted, equidistant_points

def heart_draw_curve_and_points(x, y, left, fix):
    # Create an interpolation function for x and y separately
    interp_x = interp1d(np.arange(len(x)), x, kind='cubic')
    interp_y = interp1d(np.arange(len(y)), y, kind='cubic')

    # Generate the interpolated points with higher resolution
    alpha = np.linspace(0, len(x) - 1, 1000)
    points_fitted = np.column_stack((interp_x(alpha), interp_y(alpha)))
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points_fitted, axis=0) ** 2, axis=1))
    
    # Calculate cumulative distance along the curve
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    # Calculate equidistant points based on length
    if fix:
        target_distances = np.linspace(0, cumulative_distances[-10], 21)
    elif left:
        target_distances = np.linspace(0, cumulative_distances[-1], 11)
    else:
        target_distances = np.linspace(0, cumulative_distances[-1], 10)

    equidistant_indices = np.searchsorted(cumulative_distances, target_distances, side='right')
    equidistant_indices = np.clip(equidistant_indices, 0, len(points_fitted) - 1)
    equidistant_points = points_fitted[equidistant_indices]
    
    return points_fitted, equidistant_points


def Letter_c(hand_landmarks):

    # Given order
    order = [4, 3, 2, 5, 6, 7, 8]

    # Extracting points in the given order
    ordered_points = [hand_landmarks[idx] for idx in order]

    # Separating x and y points
    x = np.array([point[0] for point in ordered_points])
    y = np.array([point[1] for point in ordered_points])
    return x, y

def open_hand(hand_landmarks):
    hand_landmarks = hand_landmarks[1:]
    return hand_landmarks


def Heart(hand_landmarks):

    hand_landmarks_left = hand_landmarks[:21]
    hand_landmarks_right =hand_landmarks[21:]

    # Given order
    order = [4, 5, 6, 7, 8]

    # Extracting points in the given order
    ordered_points_left = [hand_landmarks_left[idx] for idx in order]
    ordered_points_right = [hand_landmarks_right[idx] for idx in order]
    
    # Separating x and y points
    x_left = np.array([point[0] for point in ordered_points_left])
    y_left = np.array([point[1] for point in ordered_points_left])

    x_right = np.array([point[0] for point in ordered_points_right])
    y_right = np.array([point[1] for point in ordered_points_right])

    return x_left, y_left, x_right, y_right

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def Heart_fixed(hand_landmarks):
    # Assuming hand_landmarks contains a single 2D point [x, y]
    x, y = hand_landmarks
    
    # Adjust the heart shape relative to the provided hand_landmarks
    heart_points = [
        [clamp(x, 0, 200), clamp(y, 0, 200)],                           # Hand landmark point
        [clamp(x - 20, 0, 200), clamp(y - 20, 0, 200)],                 # 80% of the landmark point
        [clamp(x - 15, 0, 200), clamp(y - 30, 0, 200)],                 # 85% of the landmark point
        [clamp(x - 5, 0, 200), clamp(y - 25, 0, 200)],                  # 95% of the landmark point
        [clamp(x, 0, 200), clamp(y - 20, 0, 200)],                  # 95% of the landmark point
        [clamp(x + 5, 0, 200), clamp(y - 25, 0, 200)],                  # 105% of the landmark point
        [clamp(x + 15, 0, 200), clamp(y - 30, 0, 200)],                 # 115% of the landmark point
        [clamp(x + 20, 0, 200), clamp(y - 20, 0, 200)],                 # 120% of the landmark point
        [clamp(x, 0, 200), clamp(y, 0, 200)]                            # Back to the hand landmark point to complete the heart
    ]
    x, y = zip(*heart_points)
    return x, y

# Function to generate equidistant points on a circle
def equidistant_points_on_circle(center, radius, num_points=21):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points = np.array([center[0] + radius * np.cos(angles),
                              center[1] + radius * np.sin(angles)]).T
    return circle_points


def equidistant_points_func(hand_landmarks, symbol, hand_history):
    # scaled_landmark = rescale_landmark_resolution(hand_landmarks) # No further changes required
    if symbol=="Letter C" and "Left" in hand_history and "Right" in hand_history:
        x_left, y_left, x_right, y_right = Heart(hand_landmarks)
        points_fitted, equidistant_points_left = heart_draw_curve_and_points(x_left, y_left, left=True, fix=False)
        points_fitted, equidistant_points_right = heart_draw_curve_and_points(x_right, y_right, left=False, fix=False)
        # equidistant_points = equidistant_points_left + equidistant_points_right
        equidistant_points = np.concatenate((equidistant_points_left, equidistant_points_right), axis=0)

    elif symbol == "Letter C":
        x, y = Letter_c(hand_landmarks)
        points_fitted, equidistant_points = draw_curve_and_points(x, y)

    elif symbol == "Open":
        equidistant_points = hand_landmarks
    
    elif symbol == "Pointer":
        x, y = zip(*hand_landmarks)
        points_fitted, equidistant_points = draw_curve_and_points(x, y)

    elif symbol == "Heart":
        x, y = Heart_fixed(hand_landmarks)
        points_fitted, equidistant_points = heart_draw_curve_and_points(x, y, left=False, fix=True)

    elif symbol == "Closed":
        center_point = np.array([hand_landmarks[0], hand_landmarks[1]-10])
        circle_radius = 20
        equidistant_points = equidistant_points_on_circle(center_point, circle_radius)

    return equidistant_points

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










# def plot(x, y, points_fitted, equidistant_points): # No further changes required
#     # Graph:
#     plt.plot(x, y, 'ok', label='original points')
#     plt.plot(points_fitted[:, 0], points_fitted[:, 1], '-r', label='fitted spline k=3, s=0.2')
#     plt.scatter(equidistant_points[:, 0], equidistant_points[:, 1], color='red', label='Equidistant points')
#     plt.axis('equal')
#     plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()

