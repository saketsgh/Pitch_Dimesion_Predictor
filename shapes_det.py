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


def detectBlobs(img_):
    
    img = copy.deepcopy(img_)
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 800
    
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # Set inertial filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw blobs on our image as red circles 
    blank = np.zeros((1, 1))  
    blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255), 
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return blobs

def get_green_mask(frame):
    # convert to hsv space
    dst = cv2.addWeighted(frame, 1, frame, 1, 0)
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

    # pick the frame with the max lines
    Max = -1
    n_frame = 0
    Line = []
    top_five = {}

    # iterate through all the frames
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            print("invalid frame\n")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_green = get_green_mask(frame)
        frame_green = cv2.GaussianBlur(frame_green, (5,5), 0)
        
        # blobs = detectBlobs(frame_green)
        # frame = cv2.addWeighted(frame, 1, frame, 1, 0)
        # probabilistic Hough Transform for line detection
        
        gray = cv2.cvtColor(frame_green, cv2.COLOR_BGR2GRAY)
        #--- First obtain the threshold using the greyscale image ---
        ret, th = cv2.threshold(gray, 127, 255, 0)

        #--- Find all the contours in the binary image ---
        _, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours
        big_contour = []
        Mx = 0
        for i in cnt:
            area = cv2.contourArea(i)
            #--- find the contour having biggest area ---
            if(area > Mx):
                Mx = area
                big_contour = i 

        frame_green = cv2.drawContours(frame_green, big_contour, -1, (255, 0, 0), 3)
        # cv2.imshow('final', final)

        '''edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minLineLength = 500
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 400, minLineLength, maxLineGap)
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        '''
        # Create default parametrization LSD
        fld = cv2.ximgproc.createFastLineDetector(230, 1.2, 50, 50, 3, True)

        # Detect lines in the image
        lines = fld.detect(gray)
        
        # pick the frame with max num of lines 
        if lines is not None:
            if len(lines) > Max:
                Max = len(lines)
                n_frame = count
                Line = lines
                Frame = copy.deepcopy(frame_green)
            top_five[count] = len(lines)

        # Draw detected lines in the image
        frame = fld.drawSegments(frame, lines)
        
        print("frame {}".format(count))
        # cv2.imshow('frame', frame)
        cv2.imshow('frame', frame_green)
        # cv2.imshow('canny', edges)
        count += 1
    
    # print(Max, n_frame)
    cap.release()

    # cv2.imshow('frame with max lines', Frame)
    edges = cv2.Canny(cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=3)

    # Draw detected lines in the image
    res = fld.drawSegments(Frame, Line)
    cv2.imshow('frame with max lines()', res)

    # for l in lines:
    #     x1, y1 = l[0][0], l[0][1]
    #     x2, y2 = l[0][2], l[0][3]
    #     res = cv2.line(Frame, (y1, x1), (y2, x2), (0, 0, 255),  5) 
    # cv2.imshow('frame with max lines()', res)

    five_frames = []
    for i in range(10):
        Max = max(top_five.items(), key=lambda x: x[1]) 
        five_frames.append(Max[0])
        del top_five[Max[0]] 
    
    cap = cv2.VideoCapture(Args.path)
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            num = five_frames.index(count)
            frame = cv2.resize(frame, (np.shape(frame)[1]//2, np.shape(frame)[0]//2))
            cv2.imshow('top frame num - {}'.format(num), frame)
        except ValueError:
            pass 
        count += 1

    # cv2.imshow('frame with max lines+edges', edges)
    
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
