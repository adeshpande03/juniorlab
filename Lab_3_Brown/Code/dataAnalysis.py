import cv2
import numpy as np
from skimage import io
from scipy.optimize import linear_sum_assignment
from pprint import *


def detect_dots(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=3, maxRadius=20
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]
    return []


# def compute_displacement(dots1, dots2, frame_num):
#     results = []

#     if len(dots1) == 0 or len(dots2) == 0:
#         return results
#     # dots1[0, :, 0] - dots2[0, :, 0]
#     # Compute pairwise distance matrix
#     distance_matrix = np.linalg.norm(
#         dots1[0, :, :2] - dots2[0, :, :2], axis=2
#     )

#     # Use Hungarian algorithm to find optimal assignment
#     row_ind, col_ind = linear_sum_assignment(distance_matrix)

#     for r, c in zip(row_ind, col_ind):
#         displacement = distance_matrix[r, c]
#         results.append(
#             {
#                 "dot_id": r,
#                 "frame": frame_num,
#                 "position": tuple(dots2[c][:2]),
#                 "displacement": displacement,
#             }
#         )

#     return results


def subtract_background(tif_file, background):
    images_original = io.imread(tif_file, cv2.IMREAD_GRAYSCALE)
    background = io.imread(background, cv2.IMREAD_GRAYSCALE)
    background_avg = background[0]
    for i in range(len(background)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            background_avg = cv2.addWeighted(
                background[i], alpha, background_avg, beta, 0.0
            )
    background_avg = cv2.bitwise_not(background_avg)
    images = [
        cv2.bitwise_not(
            cv2.subtract(cv2.bitwise_not(images_original[i]), background_avg)
        )
        for i in range(len(images_original))
    ]
    return images

def track_dots(tif_file, background):
    images = subtract_background(tif_file, background)
    trackers = cv2.MultiTracker_create()
    circles = detect_dots(images[0])
    frame = images[0]
    for circle in circles:
        (x, y, r) = [int(v) for v in circle]
        # box = cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), (0, 255, 0), 2)
        box = (x-r, y-r, 2*r, 2*r)
        tracker = cv2.TrackerCSRT_create()
        trackers.add(tracker, frame, box)
        # cv2.circle(frame, (x,y), r, (0, 255, 0), 2)
    # cv2.imshow("Frame", frame)
    # cv2.waitKey(0)
    for i in range(len(images)):
        frame = images[i]
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if key == ord("s"):
        #     box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        #     tracker = cv2.TrackerCSRT_create()
        #     trackers.add(tracker, frame, box)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


# def track_dot_displacement(tif_file, background):
#     images = subtract_background(tif_file, background)
#     all_results = []

#     prev_dots = detect_dots(images[0])

#     for idx in range(1, len(images)):
#         current_dots = detect_dots(images[idx])
#         frame_results = compute_displacement(prev_dots, current_dots, idx)
#         all_results.extend(frame_results)
#         prev_dots = current_dots

#     return all_results


tif_file = "Lab_3_Brown/Data/Day 2 Trial 1 - 2 micron - 10x magnification/image_2.tif"
background = "Lab_3_Brown/Data/Day 2 Trial 1 - 2 micron - 10x magnification/image_3.tif"
track_dots(tif_file, background)
# displacements = track_dot_displacement(tif_file, background)
# # print(displacements)
# # np.save(f"{tif_file[:-4]}.txt", displacements)
# with open(f"{tif_file[:-4]}.txt", "w") as f:
#     f.write("[")
#     for line in displacements:
#         f.write(f"{line},\n")
#     f.write("]")
# pprint(displacements)
