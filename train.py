import numpy as np
import cv2
import os 
import argparse
from sklearn import mixture


class TrainGMM:

    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.train_data = np.array([])

    def load_data(self,):
        '''
        returns flattend list of pixels of all the training imgs  
        '''
        train_dir = self.train_dir
        stack = []
        for filename in os.listdir(train_dir):
            image = cv2.imread(os.path.join(train_dir, filename))
            
            # pre-processing 
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.bilateralFilter(image, 9, 75, 75)
            image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            ch = image.shape[2]
            nx = image.shape[0]
            ny = image.shape[1]

            # extract the H, S channels 
            h = image[:, :, 0]
            s = image[:, :, 1]
            
            # create a new image consisting of only H, S channels
            image = np.zeros((nx, ny, 2))
            image[:, :, 0] = h
            image[:, :, 1] = s
            image = np.reshape(image, (nx*ny, 2))
            
            # return the flattened list of pixels 
            for i in range(image.shape[0]):
                stack.append(image[i,:])

        self.train_data = stack

    def fit(self,):
        '''
        function to train Gaussian Mixture Model for green color segmentation 
        '''            
        # train
        self.load_data()
        model = mixture.GaussianMixture(n_components=3, covariance_type='full', random_state=2)
        model.fit(self.train_data)

        # save model parameters 
        gmm_name = "gmmGreen"
        save_model_path = "./gmm_model"
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        np.save(save_model_path + '/' + gmm_name + '_weights', model.weights_, allow_pickle=False)
        np.save(save_model_path + '/' + gmm_name + '_means', model.means_, allow_pickle=False)
        np.save(save_model_path + '/' + gmm_name + '_covariances', model.covariances_, allow_pickle=False)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--path", help="path to train dir")
    args = argparser.parse_args() 
    train_dir = args.path 
    
    # train and save the model
    gmm_model = TrainGMM(train_dir)
    gmm_model.fit()

    print("model parameters stored in folder 'gmm_model'.....")