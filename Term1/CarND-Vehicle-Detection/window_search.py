import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
from helper import *

def search_all_scales(para_path, image_path, allwindows, alloverlaps, Y_pos, X_pos, jpg=False):

	with open(para_path, mode='rb') as f:
		data = pickle.load(f)
		svc = data['svc'] 
		color_space = data['color_space']
		spatial_size = data['spatial_size']
		hist_bins = data['hist_bins']
		orient = data['orient']
		pix_per_cell = data['pix_per_cell']
		cell_per_block = data ['cell_per_block']
		hog_channel = data['hog_channel']
		spatial_feat = data ['spatial_feat']
		hist_feat = data['hist_feat']
		hog_feat = data['hog_feat']
		X_scaler = data['X_scaler']
	
	w0, w1, w2, w3  = allwindows[0][0], allwindows[1][0], allwindows[2][0], allwindows[3][0]

	Y_start_stop = Y_pos 
	X_start_stop = X_pos

	hot_windows = []
	all_windows = []

	image = mpimg.imread(image_path)
	if jpg:
		image = image.astype(np.float32)/255

	for i in range(len(Y_start_stop)):
		windows = slide_window(image, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i], 
					xy_window=allwindows[i], xy_overlap=alloverlaps[i])
		all_windows += [windows]
		hot_windows += search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
	return hot_windows, all_windows

def search_all_scales_video(para_path, image, allwindows, alloverlaps, Y_pos, X_pos):
	
	with open(para_path, mode='rb') as f:
		data = pickle.load(f)
		svc = data['svc'] 
		color_space = data['color_space']
		spatial_size = data['spatial_size']
		hist_bins = data['hist_bins']
		orient = data['orient']
		pix_per_cell = data['pix_per_cell']
		cell_per_block = data ['cell_per_block']
		hog_channel = data['hog_channel']
		spatial_feat = data ['spatial_feat']
		hist_feat = data['hist_feat']
		hog_feat = data['hog_feat']
		X_scaler = data['X_scaler']
	
	w0, w1, w2, w3  = allwindows[0][0], allwindows[1][0], allwindows[2][0], allwindows[3][0]

	Y_start_stop = Y_pos 
	X_start_stop = X_pos

	hot_windows = []
	all_windows = []

	for i in range(len(Y_start_stop)):
		windows = slide_window(image, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i], 
					xy_window=allwindows[i], xy_overlap=alloverlaps[i])
		all_windows += [windows]
		hot_windows += search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
	return hot_windows, all_windows