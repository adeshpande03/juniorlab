import cv2
import numpy as np
from skimage import io
from scipy.optimize import linear_sum_assignment
from pprint import *


def detect_dots(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=10, minRadius=1, maxRadius=5
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]
    return []

ix = -1
def click_event(event, x, y, flags, params):
    global ix
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
    
def get_scale(ruler, background):
    image = subtract_background(ruler, background)
    x = []
    for i in range(2):
        cv2.imshow("Frame", image[0])
        cv2.setMouseCallback("Frame", click_event)
        cv2.waitKey(0)
        x.append(ix)
    cv2.destroyAllWindows()
    return x[1] - x[0]

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
    # cv2.imshow("Frame", images_original)
    # cv2.waitKey(0)
    # cv2.imshow("Frame", background_avg)
    # cv2.waitKey(0)
    if len(images_original.shape) == 2:
        images = [
            cv2.bitwise_not(
                cv2.subtract(cv2.bitwise_not(images_original), background_avg)
            )
            for i in range(len(images_original))
        ]
    else :
        images = [
            cv2.bitwise_not(
                cv2.subtract(cv2.bitwise_not(images_original[i]), background_avg)
            )
            for i in range(len(images_original))
        ]
    return images

def track_dots(tif_file, background):
    all_positions = []
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
        positions = []
        frame = images[i]
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            positions.append((x+w/2, y+h/2))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        all_positions.append(positions)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if key == ord("s"):
        #     box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        #     tracker = cv2.TrackerCSRT_create()
        #     trackers.add(tracker, frame, box)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    array = np.array(all_positions)
    np.save("Lab_3_Brown/array", array)
    return array


def compute_displacement(tif_file, background):
    all_positions = track_dots(tif_file, background)
    # all_positions = np.load("Lab_3_Brown/array.npy")
    all_results = []
    for i in range(1, len(all_positions)):
        all_results.append(all_positions[i] - all_positions[0])
        # current_dots = detect_dots(images[idx])
        # frame_results = compute_displacement(prev_dots, current_dots, idx)
        # all_results.extend(frame_results)
        # prev_dots = current_dots
    return np.array(all_results)

tif_file = "Lab_3_Brown/Data/Day 2 Trial 3 - 1 micron - 10x magnification/image_0.tif"
background = "Lab_3_Brown/Data/Day 2 Trial 3 - 1 micron - 10x magnification/image_2.tif"
ruler = "Lab_3_Brown/Data/Day 2 Trial 3 - 1 micron - 10x magnification/image_4.tif"
scale = 50e-6/get_scale(ruler, background)
displacements = scale*compute_displacement(tif_file, background)
e_x2 = []
for i in displacements:
    e_x2.append(np.average(np.linalg.norm(i, axis=1)**2))
e_x2 = np.array(e_x2)
# print(e_x2)
# print(len(e_x2))
time = (np.arange(0, len(e_x2))+1)*0.2
print(np.average(np.gradient(e_x2, time)))
# displacements = track_dot_displacement(tif_file, background)
# # print(displacements)
# # np.save(f"{tif_file[:-4]}.txt", displacements)
# with open(f"{tif_file[:-4]}.txt", "w") as f:
#     f.write("[")
#     for line in displacements:
#         f.write(f"{line},\n")
#     f.write("]")
# pprint(displacements)
