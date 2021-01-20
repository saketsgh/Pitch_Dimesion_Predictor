import cv2
import argparse
import numpy as np
import os
import random
from glob import glob


def source_imgs(Args):
    '''
    for saving frames of souce video
    '''
    if not os.path.exists("./data"):
        os.mkdir("./data")

    # load the video(s)
    for i, file_name in enumerate(os.listdir(Args.path)): 
        cap = cv2.VideoCapture(os.path.join(Args.path, file_name))

        # get video properties
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        rand_frames = random.choices(list(range(1, length)), k=15) 

        for index in rand_frames:
            # load random frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)
            res, frame = cap.read()

            # save it 
            if not os.path.exists("./data/frames/{}".format(file_name[:-4])):
                os.mkdir("./data/frames/{}".format(file_name[:-4]))
            cv2.imwrite("./data/frames/{}/frame_{}.png".format(file_name[:-4], index), frame)


def select_roi(img, num):
    '''
    uses event handling to select ROI
    '''
    r = cv2.selectROI("Image", img, fromCenter=False)
     
    # Crop image
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # store the cropped image, to be used for training gmm
    if not os.path.exists("./data/train/"):
        os.mkdir("./data/train/")
    cv2.imwrite("./data/train/img_{}.png".format(num), imCrop)



if __name__ == "__main__":
    
    # command line arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to source video")
    Args = ap.parse_args()

    # source_imgs(Args)
    # create the training set
    count = 54
    for folder in os.listdir("./data/frames/"):
        path = os.path.join("./data/frames/", folder)
        path = path + "/*.png"
        for file_name in glob(path):
            img = cv2.imread(file_name)
            select_roi(img, count)
            count += 1

    if cv2.waitKey(0):  
        cv2.destroyAllWindows()