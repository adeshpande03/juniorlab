import cv2
import numpy as np
from skimage import io
from scipy.optimize import linear_sum_assignment
from pprint import *

def detect_dots(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []


def compute_displacement(dots1, dots2, frame_num):
    results = []
    
    if len(dots1) == 0 or len(dots2) == 0:
        return results
    
    # Compute pairwise distance matrix
    distance_matrix = np.linalg.norm(dots1[:, np.newaxis, :2] - dots2[np.newaxis, :, :2], axis=2)
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    for r, c in zip(row_ind, col_ind):
        displacement = distance_matrix[r, c]
        results.append({
            'dot_id': r,
            'frame': frame_num,
            'position': tuple(dots2[c][:2]),
            'displacement': displacement
        })
    
    return results

def track_dot_displacement(tif_file):
    images = io.imread(tif_file)
    all_results = []

    prev_dots = detect_dots(images[0])

    for idx in range(1, len(images)):
        current_dots = detect_dots(images[idx])
        frame_results = compute_displacement(prev_dots, current_dots, idx)
        all_results.extend(frame_results)
        prev_dots = current_dots

    return all_results



tif_file = "Lab_3_Brown/Data/Day 1 Trial 1 - 1 micron - 10x magnification/image_0.tif"
displacements = track_dot_displacement(tif_file)
# print(displacements)
# np.save(f"{tif_file[:-4]}.txt", displacements)
with open(f"{tif_file[:-4]}.txt", "w") as f:
    for line in displacements:
        f.write(f"{line}\n")

pprint(displacements)
