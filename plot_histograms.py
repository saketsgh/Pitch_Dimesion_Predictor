import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def plot_data():
    hue = []
    sat = []
    val = []
    count = 0 
    for filename in os.listdir("./data/train"):
        image = cv2.imread(os.path.join("./data/train", filename))
        image = cv2.resize(image,(40, 40), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height = image.shape[1]
        width = image.shape[0]
        channels = 3
        image = np.reshape(image, (width*height, channels))
        hue += list(image[:, 0])
        sat += list(image[:, 1])
        # print(np.max(image[:, 1]))
        val += list(image[:, 2])
    
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    n, bins, patches = axs[0].hist(hue, 20)
    n, bins, patches = axs[1].hist(sat, 20)
    n, bins, patches = axs[2].hist(val, 20)
    fig.suptitle('histogram plots of h, s, v')
    plt.show()


if __name__ == "__main__":
    
    plot_data()
    