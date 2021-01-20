import cv2
import argparse
import copy
import numpy as np


def show_circles(img_):
    img = copy.deepcopy(img_)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=300, param2=40, minRadius=30, maxRadius=300)
    if circles is None:
        print("no circle found")
    else:
        print("circle found")
        # print(circles)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow('circles', img)


def get_green_mask(frame):
    # convert to hsv space
    # dst = cv2.addWeighted(frame, 1, frame, 1, 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower = np.array([36, 25, 25], dtype=np.uint8)
    upper = np.array([70, 255,255], dtype=np.uint8)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)

    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    return res


def erode_dilate(img_, size1=(5, 5), size2=(5, 5), iterations=30):

    # perform erosion + dilation for noise suppresion 
    img = copy.deepcopy(img_)
    kernel1 = np.ones(size1, np.uint8)
    kernel2 = np.ones(size2, np.uint8)
    erosion = cv2.erode(img, kernel=kernel1,iterations=iterations)
    dilation = cv2.dilate(erosion, kernel=kernel2, iterations=iterations)
    return dilation
 

if __name__ == "__main__":
    
    # command line arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to source video")
    Args = ap.parse_args()
    
    # load the video
    cap = cv2.VideoCapture(Args.path)

    # to keep track of the frame count
    count = 1
    print("fps of video --> {}".format(cap.get(cv2.CAP_PROP_FPS)))

    prev = None
    # iterate through all the frames
    while cap.isOpened():
 
        success, frame = cap.read()
        if not success:
            print("invalid frame\n")
            break
        
        # cv2.imshow('frame num - {}'.format(count), frame)
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
        if prev is not None:
            cv2.imshow('diff', edges-prev)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
        prev = edges
        # if count%20 == 0:
    cv2.destroyAllWindows()


    cv2.destroyAllWindows()