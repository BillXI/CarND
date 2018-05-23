import numpy as np
import cv2
import glob
import time
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import sklearn.svm as svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib

from helper import *

#helper function to extract features from files
def get_features(files, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    features = []
    for file in files:
        
        img = mpimg.imread(file)
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        
        features.append(img_features)
    return features




def get_all_features(data_file, pickle_file, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	# Get all the data
	with open(data_file, mode='rb') as f:
		data = pickle.load(f)
		vehicles_train = data['vehicles_train']
		nonvehicles_train = data['nonvehicles_train']
		vehicles_val = data['vehicles_val']
		nonvehicles_val = data['nonvehicles_val']
		vehicles_test = data['vehicles_test']
		nonvehicles_test = data['nonvehicles_test']
	
	vehicles_train_feat = get_features(vehicles_train, color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
	vehicles_val_feat   = get_features(vehicles_val,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
	vehicles_test_feat  = get_features(vehicles_test,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

	nonvehicle_train_feat = get_features(nonvehicles_train,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
	nonvehicle_val_feat   = get_features(nonvehicles_val,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
	nonvehicle_test_feat  = get_features(nonvehicles_test,color_space, spatial_size,hist_bins, orient, 
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

	# Create an array stack of feature vectors that collect all the features.
	X = np.vstack((vehicles_train_feat,
				vehicles_val_feat,
				vehicles_test_feat,
				nonvehicle_train_feat,
				nonvehicle_val_feat,
				nonvehicle_test_feat)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Find index to resize back to the train, val, test sets.
	l1,l2,l3,l4,l5,l6 = len(vehicles_train_feat), len(vehicles_val_feat), len(vehicles_test_feat), len(nonvehicle_train_feat), len(nonvehicle_val_feat), len(nonvehicle_test_feat)

	ix1 = l1
	ix2 = ix1 + l2
	ix3 = ix2 + l3
	ix4 = ix3 + l4
	ix5 = ix4 + l5

	# get back features from scalled X
	vehicles_train_feat, vehicles_val_feat, vehicles_test_feat     = scaled_X[:ix1], scaled_X[ix1:ix2], scaled_X[ix2:ix3]
	nonvehicle_train_feat,nonvehicle_val_feat,nonvehicle_test_feat = scaled_X[ix3:ix4],scaled_X[ix4:ix5],scaled_X[ix5:]

	feature_dict = {
		"vehicles_train_feat": scaled_X[:ix1],
		"vehicles_val_feat": scaled_X[ix1:ix2],
		"vehicles_test_feat": scaled_X[ix2:ix3],
		"nonvehicle_train_feat": scaled_X[ix3:ix4],
		"nonvehicle_val_feat": scaled_X[ix4:ix5],
		"nonvehicle_test_feat": scaled_X[ix5:],
		"X_scaler": X_scaler
	}

	# dump to pickle
	try:
		with open(pickle_file, "wb") as pfile:
			pickle.dump(
                feature_dict,
                pfile, 
                pickle.HIGHEST_PROTOCOL
            )
	except Exception as e:
		print("Unable to save data ", pickle_file, ": ", e)
		raise
		
	print('Data cached in pickle file: ', pickle_file)
	print('Finished')
	return feature_dict


	
