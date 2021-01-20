import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import os
import time
from sklearn import mixture
from sklearn.cluster import KMeans
# model = mixture.GaussianMixture(n_components=3, covariance_type='full', random_state=2)
# model = KMeans(n_clusters=3, max_iter=800, random_state=2, algorithm='full')

def load_model():
    '''
    loads the model parameters obtained after fitting GMM on training data
    '''
    save_model_path = "./gmm_model"
    gmm_name = "gmmGreen"
    means = np.load(save_model_path + '/' + gmm_name + '_means.npy')
    covar = np.load(save_model_path + '/' + gmm_name + '_covariances.npy')
    model = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    model.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    model.weights_ = np.load(save_model_path + '/' + gmm_name + '_weights.npy')
    model.means_ = means
    model.covariances_ = covar
    return model
    

if __name__ == "__main__":
    
    fld = cv2.ximgproc.createFastLineDetector(30, 1.23, True)
    # fld = cv2.ximgproc.createFastLineDetector()
    model = load_model()
    # print(model.converged_)
    # vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/Accrington v Ipswich Smaller Clip.mp4"
    # vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/Accrington v Ipswich Smaller Clip_Trim.mp4"
    # vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/England v Croatia Smaller Clip 2.mp4"
    # vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/City v Reading Smaller Clip_Trim.mp4"
    # vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/QPR v Blackburn Smaller Clip.mp4"
    vid_path = "D:/StatsBomb_Interview/Pitch_Dimension_Predicter/data/videos/City v Reading Smaller Clip.mp4"
    cap = cv2.VideoCapture(vid_path)
    start = time.time()
    count = 1

    Max = -1
    n_frame = -1
    Line = []
    top_five = {}

    while (cap.isOpened()):
        success, frame = cap.read()
        if success == False:
            break
        print("frame num - {}".format(count))
        frame = cv2.resize(frame, (80, 45), interpolation=cv2.INTER_LINEAR)
        image = cv2.GaussianBlur(frame, (5, 5), 0)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        # image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        stack = []
        ch = image.shape[2]
        nx = image.shape[0]
        ny = image.shape[1]
        # get the green channel
        # image = np.reshape(image, (nx*ny, 2))
        # image = image[:, :, 0] 
        h = image[:, :, 0]
        s = image[:, :, 1]
        image = np.zeros((nx, ny, 2), np.uint8)
        image[:, :, 0] = h
        image[:, :, 1] = s
        image = np.reshape(image, (nx*ny, 2))
        for i in range(image.shape[0]):
            stack.append(image[i,:])
            # stack.append([image[i]])
        print("shape of stack {}".format(np.shape(stack)))
        # GMM stuff ----
        cluster = model.predict_proba(stack)
        # cluster = cluster.reshape((nx, ny))
        # print(np.shape(cluster))
        cluster = np.max(cluster, axis=1)
        # plt.imshow(cluster)
        # print(np.max(cluster, axis=1))
        cluster = np.where(cluster == 1, 0, 255)
        cluster = cluster.reshape((nx, ny))
        # plt.imshow(cluster, cmap='gray')
        green_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        green_mask[:, :] = cluster
        th_frame = cv2.bitwise_and(frame, frame, mask=green_mask)
        # plt.pause(0.0000000000000003)
        cv2.imshow('thresholded frame', th_frame)
        # Create default parametrization LSD
        # -------------
        # KMeans stuff ----
        # cluster = model.predict(stack)
        # cluster = cluster.reshape((nx, ny))
        # plt.imshow(cluster)
        # plt.show()
        # break

        # Detect lines in the image
        lines = fld.detect(cv2.cvtColor(th_frame, cv2.COLOR_BGR2GRAY))
        # pick the frame with max num of lines 
        if lines is not None:
            print('detected')
            if len(lines) > Max:
                Max = len(lines)
                n_frame = count
                Line = lines
            top_five[count] = len(lines)
        # cv2.imshow('final', line_det)
        # line_det = fld.drawSegments(frame, lines)
        # cv2.imshow('final', line_det)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    # plt.show()
    # Create default parametrization LSD

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    five_frames = []
    for i in range(100):
        Max = max(top_five.items(), key=lambda x: x[1]) 
        five_frames.append(Max[0])
        del top_five[Max[0]]

    # print(five_frames)
    print("\ntime elapsed - {}".format(time.time()-start))
    # five_frames = [1248, 1574, 2079, 1238, 2081, 2082, 1241, 2080, 2087, 2103]
    cap = cv2.VideoCapture(vid_path)
    for num, index in enumerate(five_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        # lines = fld.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # res = fld.drawSegments(frame, lines)
        cv2.imshow('top frame - {}'.format(num), frame)
        print("top frame num - {}".format(index))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()