import cv2
import numpy as np
from skimage import io
from scipy.optimize import curve_fit
from pprint import *
import matplotlib.pyplot as plt


def detect_dots(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    circles = cv2.HoughCircles(
        # for 2 micron
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=4, maxRadius=15
        # for 1 micron
        # gray, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=13, minRadius=1, maxRadius=5
        # ignore this one
        # gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=5, minRadius=1, maxRadius=5
    )
    # circles = circles.reshape(1, circles.shape[1], circles.shape[2])
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
    # image = subtract_background(ruler, background)
    image = [2*io.imread(ruler, cv2.IMREAD_GRAYSCALE)]
    x = []
    for i in range(2):
        cv2.imshow("background", image[0])
        cv2.setMouseCallback("background", click_event)
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
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    for i in range(len(images)):
        # topHat = cv2.morphologyEx(images[i], cv2.MORPH_TOPHAT, kernel)
        # blackHat = cv2.morphologyEx(images[i], cv2.MORPH_BLACKHAT, kernel)
        # images[i] = images[i] + topHat - blackHat
        # images[i] = cv2.equalizeHist(images[i])
        # images[i] = cv2.convertScaleAbs(images[i], alpha=1.01, beta=0)
        images[i] = 2*images[i]
        images[i] = cv2.medianBlur(images[i], 1)
        # ret, thresh = cv2.threshold(images[i], 205, 255, cv2.THRESH_BINARY)
        # images[i] = thresh
    return images

def track_dots(tif_file, background):
    all_positions = []
    images = subtract_background(tif_file, background)
    trackers = cv2.MultiTracker_create()
    circles = detect_dots(images[0])
    frame = images[0]
    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(0) 
    cv2.imwrite("Lab_3_Brown/afterfilter.png", frame)
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
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,50), 1)
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
    # np.save("Lab_3_Brown/array", array)
    return array


def compute_displacement(tif_file, background):
    all_positions = track_dots(tif_file, background)
    # all_positions = np.load("Lab_3_Brown/array.npy")
    all_results = []
    for i in range(len(all_positions)):
        all_results.append(all_positions[i] - all_positions[0])
        # current_dots = detect_dots(images[idx])
        # frame_results = compute_displacement(prev_dots, current_dots, idx)
        # all_results.extend(frame_results)
        # prev_dots = current_dots
    return np.array(all_results)

def center_displacements(displacements):
    centered = []
    for i in displacements:
        means = i.mean(0)
        # means = np.repeat(means, repeats=len(i), axis=1)
        means = np.tile(means, reps=(len(i),1))
        centered.append(i - means)
    centered = np.array(centered)
    return centered

def linear_fit(time, a):
    return a*time

tif_file = "Lab_3_Brown/Data/Day 2 Trial 1 - 2 micron - 10x magnification/image_2.tif"
background = "Lab_3_Brown/Data/Day 2 Trial 1 - 2 micron - 10x magnification/background.tif"
ruler = "Lab_3_Brown/Data/Day 2 Trial 1 - 2 micron - 10x magnification/ruler.tif"
scale = 50e-6/get_scale(ruler, background)
a_values = []
a_errs = []
for j in range(1):
    displacements = scale*compute_displacement(tif_file, background)
    centered_displacements = center_displacements(displacements)
    print(len(centered_displacements[0]))
    e_x2 = []
    for i in centered_displacements:
        e_x2.append(np.average(np.linalg.norm(i, axis=1)**2))
    e_x2 = np.array(e_x2)
    # print(e_x2)
    # print(len(e_x2))
    time = (np.arange(0, len(e_x2)))*0.2
    plt.scatter(time, e_x2*1e12)
    params, covariance = curve_fit(linear_fit, time, e_x2)
    a = params[0]
    perr = np.sqrt(np.diag(covariance))
    aerr = perr[0]
    plt.plot(time, 1e12*linear_fit(time, a), color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Square Displacement ($\mu$m$^2$)")
    plt.savefig(f"Lab_3_Brown/Images/trial1_{j}_1.png")
    # plt.show()
    plt.clf()
    a_values.append(a)
    a_errs.append(aerr)
a_values = np.array(a_values)
a_errs = np.array(a_errs)
print(np.mean(a_values), np.linalg.norm(a_errs)/np.sqrt(len(a_errs)))
