import numpy as np
import cv2
from os.path import join
from os import listdir
from matplotlib.pyplot import clf, figure, subplot, plot, imshow
from sklearn.decomposition import PCA

pca = PCA(n_components=20)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
data_path = "../data/whale_fluke_data"
test_path = join(data_path, "test")
test_names = listdir(test_path)