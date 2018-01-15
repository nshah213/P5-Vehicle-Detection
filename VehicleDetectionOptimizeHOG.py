import pickle
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import threading
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
#from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version >= 0.18
# for scikit-learn <= 0.17 use:
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from scipy.ndimage.measurements import label


# Create a list of all vehicle files and non-vehicle files from the different available databases
# The folder categorization between vehicles and non-vehicles is done for the given folder name structure, this will have to change if file moved to different folder location
"""
vehicle_file_list = list()
non_vehicle_file_list = list()

for root, dirs, files in os.walk("./", topdown=False):
	#print(len(files))
	for fname in files:
		if (fname.endswith(".png") | fname.endswith(".jpg")):
			#print("Append")
			pathname = os.path.join(root,fname)
			if (pathname.split('/')[1] == "OwnCollection"):
				if (pathname.split('/')[2] == "vehicles"):		
					#print("Vehicle: "+ pathname)
					vehicle_file_list.append(pathname)
				if (pathname.split('/')[2] == "non-vehicles"):		
					#print("Non - Vehicle: "+ pathname)
					non_vehicle_file_list.append(pathname)
			else:
				if (pathname.split('/')[1] == "vehicles"):		
					#print("Vehicle: "+ pathname)
					vehicle_file_list.append(pathname)
				if (pathname.split('/')[1] == "non-vehicles"):		
					#print("Non - Vehicle: "+ pathname)
					non_vehicle_file_list.append(pathname)



print(len(vehicle_file_list))
print(len(non_vehicle_file_list))

# Read in cars and notcars

cars = []
notcars = []
for fname in vehicle_file_list:
	image = cv2.imread(fname)	
	cars.append(image)

for fname in non_vehicle_file_list:
	image = cv2.imread(fname)	
	notcars.append(image)

if (True):
	pickle_file = './pickle_dataset.p'
	print('Saving data to pickle file...')
	try:
		with open(pickle_file, 'wb') as pfile:
			pickle.dump(
                   	{
	                   'cars': cars,
                           'notcars': notcars
                   	},
                   	pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	print('Data cached in pickle file.')

data_file = './classifiers/pickle_dataset.p'

with open(data_file, mode='rb') as f:
    data = pickle.load(f)
    
cars = data['cars']
notcars =data['notcars']

#print(np.shape(cars))
#print(np.shape(notcars))
"""

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                  cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=True, 
                  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# Otherwise call with one output
	else:
		features = hog(img, orientations=orient, 
                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                  cells_per_block=(cell_per_block, cell_per_block), 
                  transform_sqrt=True, 
                  visualise=vis, feature_vector=feature_vec)
		return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256), num_channels = 3):
	# Compute the histogram of the color channels separately
	#image_hist = []
	#for i in range(num_channels):
	channel1_hist, bin_edges = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel2_hist, bin_edges = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	#channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.hstack((channel1_hist,channel2_hist))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(16, 16),
                        hist_bins=16, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for i in range(len(imgs)):
		file_features = []
		# Read in each one by one
		image = imgs[i]
		# apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
			elif color_space =='Custom':
				feature_image = np.dstack((cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:,:,0],cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1],cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,2]))
		else: 
			feature_image = np.copy(image)      
		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			file_features.append(spatial_features)
			#file_features = np.concatenate(file_features)
			#print("End of spatial: "+ str(np.shape(file_features)))
		if hist_feat == True:
			# Apply color_hist()
			num_channels = feature_image.shape[2]
			#print(num_channels)
			hist_features = color_hist(feature_image, nbins=hist_bins, num_channels = num_channels)
			#print(np.shape(hist_features))
			hist_features = hist_features.reshape((1,len(hist_features)))
			#print(hist_features)
			#print(np.shape(hog_features))
			file_features= np.hstack((file_features,hist_features))

			#file_features.append(hist_features)
			#print("End of hist: "+ str(np.shape(file_features)))
		if hog_feat == True:
			# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(
                                         feature_image[:,:,channel], 
                                         orient, pix_per_cell, cell_per_block, 
                                         vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:
				hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
				#plt.imshow(hog_image)
				#plt.show()
			# Append the new feature vector to the features list
			hog_features = np.array(hog_features)
			hog_features = hog_features.reshape((1,len(hog_features)))
			#print(np.shape(hog_features))
			#file_features= np.hstack((file_features,hog_features))
		#features.append(np.concatenate(file_features))			
		features.append(np.concatenate(hog_features))
		#print(np.shape(features))
			#print("End of hog: "+ str(np.shape(file_features)))
	# Return list of feature vectors
	return features


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def initial_slide_window(imgsize, y_start_stop=[None, None], xy_overlap=(0.5, 0.5), wscale = 1.0):
	#nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	#ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
	#print(nx_windows, ny_windows)
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	"""
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	"""
#	for ys in range(ny_windows):


#	ystep = 10
	#for i in range(ystart,ystop,ystep):
	ystart = y_start_stop[0]
	ystop = y_start_stop[1]
	i = ystart	
	OuterLoop = True
	while OuterLoop == True:
		window_size = int(((46457.67/((740. - i)))-120.)*wscale)
		xmid = int(imgsize[1]/2.)
		xstop = xmid + 6 * window_size
		xstart = xmid
		x_range = int(xstop - xstart)
		print(i)
		
		xcur = xmid
		woffset = int(window_size/2.)
		#wcur = woffset
		#startx = xcur - woffset
		#endx = xcur + woffset
		ystep = int(window_size * xy_overlap[1])
		#ystep = int(window_size/8.)
		i += ystep		
		if i > ystop:
			break

		starty = i - window_size
		endy = i
		wcur = window_size
		print(window_size)
		#window_list.append(((startx,starty),(endx,endy)))
		xLoop = True
		while xLoop == True:# - window_size/2):
			#print(xcur)
			# Calculate window position
			
			wcur = int((1. + ((xcur - xmid)/x_range*0.))* window_size)
			woffset = int(wcur/2.)
			
			startx = xcur - woffset
			endx = xcur + woffset
			
			#print(startx, woffset)
			if (endx > imgsize[1]):
				endx = imgsize[1]
			if (startx >= imgsize[1]*0.9):
				break
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
			xcur += int(wcur*xy_overlap[0])	
	# Return the list of windows
	print(window_list)	
	return window_list
	
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	#print(bboxes)
	clr = 20
	idx = 0
	color_offset = 31
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		#print(bbox)
		bbox = tuple(map(tuple, bbox))
		clr += color_offset
		
		if clr > 255:
			clr -= 256
		
		if (np.mod(idx,3)==0):
			color = (clr,0,0)
		if (np.mod(idx,3)==1):
			color = (0,clr,0)
		if (np.mod(idx,3)==2):
			color = (0,0,clr)
		idx += 1
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		#print(window)
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(feature_image, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	features =[]
	file_features = []
	# Read in each one by one
	#image = imgs[i]
	# apply color conversion if other than 'RGB'
	      
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		file_features.append(spatial_features)
		#file_features = np.concatenate(file_features)
		#print("End of spatial: "+ str(np.shape(file_features)))
	if hist_feat == True:
		# Apply color_hist()
		num_channels = feature_image.shape[2]
		#print(num_channels)
		hist_features = color_hist(feature_image, nbins=hist_bins, num_channels = num_channels)
		#print(np.shape(hist_features))
		hist_features = hist_features.reshape((1,len(hist_features)))
		#print(hist_features)
		#print(np.shape(hog_features))
		file_features= np.hstack((file_features,hist_features))
		#file_features.append(hist_features)
		#print("End of hist: "+ str(np.shape(file_features)))
	if hog_feat == True:
		# Call get_hog_features() with vis=False, feature_vec=True
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(
                                        feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                           pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			#hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        #   pix_per_cell, cell_per_block, vis=True, feature_vec=True)
			#plt.imshow(hog_image)
			#plt.show()
		# Append the new feature vector to the features list
		hog_features = np.array(hog_features)
		hog_features = hog_features.reshape((1,len(hog_features)))
		#print(np.shape(hog_features))
		#file_features= np.hstack((file_features,hog_features))
		#features.append(np.concatenate(file_features))
		features.append(np.concatenate(hog_features))
		#print("End of hog: "+ str(np.shape(file_features)))
	# Return list of feature vectors
	return features

def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes thexy_overlap form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		#print(box)
	# Return updated heatmap
	return heatmap# Iterate through list of bboxes

def add_heat_weighted(heatmap, bbox_list, weight):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes thexy_overlap form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight
		#print(box)
	# Return updated heatmap
	return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img


### TODO: Tweak these parameters and see how the results change.
color_space = 'Custom' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 8 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [420, 700] # Min and max in y to search in slide_window()
"""
sample_size = 500
select_index = np.linspace(0,(len(cars)-(np.mod(len(cars),sample_size))),sample_size)
select_index = select_index.astype(int)
#print(select_index[0])
#print(np.shape(select_index[0]))

carstmp =[]
notcarstmp = []

for i in range(sample_size):
	carstmp.append(cars[select_index[i]])
	notcarstmp.append(notcars[select_index[i]])

print(carstmp[0])
print(np.shape(carstmp[0]))
"""
"""
car_features = extract_features(cars, color_space=color_space,spatial_size=spatial_size, hist_bins=hist_bins,orient=orient, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,hog_channel=hog_channel, spatial_feat=spatial_feat,hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space,spatial_size=spatial_size, hist_bins=hist_bins,orient=orient, pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,hog_channel=hog_channel, spatial_feat=spatial_feat,hist_feat=hist_feat, hog_feat=hog_feat)

if (False):
	pickle_file = './pickle_features4.p'
	print('Saving data to pickle file...')
	try:
		with open(pickle_file, 'wb') as pfile:
			pickle.dump(
                   	{
	                   'car_features': car_features,
                           'notcar_features': notcar_features
                   	},
                   	pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	print('Data cached in pickle file.')
"""
features_file = './classifiers/pickle_features4.p'

with open(features_file, mode='rb') as f:
    data = pickle.load(f)

car_features = data['car_features']
notcar_features = data['notcar_features']

#print(np.shape(car_features))
#print(np.shape(notcar_features))

#print(np.shape(car_features))
#print(np.shape(notcar_features))

#print(np.shape(car_features))
#print(notcar_features)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
                  
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

#print(X_scaler.scale_)
#print(X_scaler.mean_)
#print(X_scaler.var_)
#print(np.shape(X_scaler.mean_))


# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

if (False):
	pickle_file = './pickle_scaler.p'
	print('Saving data to pickle file...')
	try:
		with open(pickle_file, 'wb') as pfile:
			pickle.dump(
                   	{
	                   'X_scaler': X_scaler
                   	},
                   	pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	print('Data cached in pickle file.')

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# parameter search for SVC
C_range = np.linspace(80, 120, 3)
gamma_range = np.linspace(0.000008, 0.000012, 3)

#print(C_range)
#print(gamma_range)

param_grid = dict(C=C_range, gamma= gamma_range)
#parameters = {'kernel':('linear','rbf'), 'C':[1,10,20],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
#parameters = {'kernel':('linear', 'rbf'), 'C':[1,5,10,20,30]}
#parameters = {'kernel':('linear', 'rbf')}

#tree = svm.
svr = svm.SVC()
clf = GridSearchCV(svr, param_grid)

# Use a linear SVC 
#svc = LinearSVC()
"""
# Check the training time for the SVC

t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC

# Check the prediction time for a single sample
t=time.time()

if (True):
	pickle_file = './pickle_SVC3.p'
	print('Saving data to pickle file...')
	try:
		with open(pickle_file, 'wb') as pfile:
			pickle.dump(
                   	{
	                   'clf': clf
                   	},
                   	pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	print('Data cached in pickle file.')
"""


training_file = './classifiers/pickle_SVC.p'

with open(training_file, mode='rb') as f:
    data = pickle.load(f)
    
clf = data['clf']
#print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
print(clf.best_params_)
print(clf.cv_results_)

def find_cars(img,xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat):
	#draw_img = np.copy(img)
	img = img.astype(np.float32)/255
	img_tosearch = img[ystart:ystop,xstart:xstop,:]
	#print("xstart - xstop = " + str(xstart)+" "+str(xstop))
	#print("ystart - ystop = " + str(ystart)+" "+str(ystop))
	#print(np.shape(img_tosearch))
	
	#ctrans_tosearch = np.dstack((cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)[:,:,0],cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HSV)[:,:,1],cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HLS)[:,:,2]))
	ctrans_tosearch = img_tosearch
	#print(np.shape(ctrans_tosearch), scale)
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	
	#if xStartPad > 0:
	padLeft = np.zeros([int(ctrans_tosearch.shape[0]), 32, 3])
	padRight= np.zeros([int(ctrans_tosearch.shape[0]), 32, 3])
		
	#print(np.shape(ctrans_tosearch), padLeft)
	ctrans_pad = np.concatenate((padLeft,ctrans_tosearch,padRight),axis=1)
	#print(np.shape(ctrans_pad))
	#else:
	#ctrans_pad = ctrans_tosearch
	ch1 = ctrans_pad[:,:,0]
	#ch2 = ctrans_tosearch[:,:,1]
	#ch3 = ctrans_tosearch[:,:,2]
	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = max(1,(ch1.shape[0] // pix_per_cell) - cell_per_block + 1)
	#print(ch1.shape, pix_per_cell, cell_per_block) 
	nfeat_per_block = orient*cell_per_block**2
	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 1  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = max((nyblocks - nblocks_per_window) // cells_per_step, 1)
	#print(nyblocks, pix_per_cell, cell_per_block)
	
	# Compute individual channel HOG features for the entire image
	hog1, hog_image = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis = True)
	#hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	#hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	plt.imshow(hog_image)
	plt.show()
	warm_windows = []
	print(nxsteps, nysteps)
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			#hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			#hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			#hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell
			# Extract the image patch
			#print(ytop,ytop+window,xleft,xleft+window)
			
			
			

						
			if spatial_feat == True:
				subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
				spatial_features = bin_spatial(subimg, size=spatial_size)
				#test_features.append(spatial_features)
				
				#file_features = np.concatenate(file_features)
				#print("End of spatial: "+ str(np.shape(file_features)))
			if hist_feat == True:
				# Apply color_hist()
				#num_channels = feature_image.shape[2]
				#print(num_channels)
				hist_features = color_hist(subimg, nbins=hist_bins)
				#print(np.shape(hist_features))
				hist_features = hist_features.reshape((1,len(hist_features)))
				#print(hist_features)
				#print(np.shape(hog_features))
				test_features= np.hstack((spatial_features.reshape(1, -1),hist_features.reshape(1, -1),hog_feat1.reshape(1, -1)))
				#file_features.append(hist_features)
				#print("End of hist: "+ str(np.shape(file_features)))
			else:
				test_features = hog_feat1


			# Scale features and make a prediction
			#test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			test_features = X_scaler.transform(test_features.reshape(1, -1))

			test_prediction = svc.predict(test_features)
			if test_prediction == 1:
				xbox_left = max(0,np.int((xleft - 32)*scale)+xstart)
				print(xstart)
				print(xleft, xbox_left)
				
				ytop_draw = min(1280,np.int(ytop*scale + ystart))
				win_draw = np.int(window*scale)
				xbox_right = max(0,xbox_left+win_draw)
				ybot_draw = min(1280,ytop_draw+win_draw )
				#cv2.rectangle(draw_img,(xbox_left, ytop_draw),(xbox_left+win_draw,ytop_draw+win_draw),(0,0,255),6)
				warm_windows.append(((xbox_left, ytop_draw),(xbox_right,ybot_draw)))
			
	return warm_windows 

with open('./camera_cal/camera.p', mode='rb') as f:
    data = pickle.load(f)
    
mtx, dist = data['CameraMatrix'], data['Distortion']
FirstTime = True
#for i in range(1,7):
def vehicleDetectionPipeline(image):
	global windows
	global mtx
	global dist
	global FirstTime
	
	#image = cv2.imread('./../test_images/test' + str(i)+'.jpg')
	#print(np.shape(image))
	undist=cv2.undistort(image, mtx, dist,None,mtx)
	#[h,w] = np.shape(image)[:2]
	t=time.time()
	feature_image = np.dstack((cv2.cvtColor(undist, cv2.COLOR_BGR2YCrCb)[:,:,0],cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)[:,:,1],cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)[:,:,2]))
	draw_image = np.copy(undist)
		
	heat = np.zeros_like(undist[:,:,0]).astype(np.float)
	window_image = np.zeros_like(undist).astype(np.float)
#	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#	luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
	# Uncomment the following line if you extracted training
	# data from .png images (scaled 0 to 1 by mpimg) and the
	# image you are searching is a .jpg (scaled 0 to 255)
	#image = image.astype(np.float32)/255

	color_space = 'Custom' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 18  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 8 # HOG cells per block
	hog_channel = 0 # Can be 0, 1, 2, or "ALL"
	spatial_size = (16, 16) # Spatial binning dimensions
	hist_bins = 16    # Number of histogram bins
	spatial_feat = False # Spatial features on or off
	hist_feat = False # Histogram features on or off
	hog_feat = True # HOG features on or off
	xy_overlap=(0.1, 0.1)
	ystart = 420
	ystop = 720/1.11
	wscale = 1.5
	imgsize = np.shape(heat)
	tmp_img = np.copy(undist)
	#print(np.shape(tmp_img))
	i = ystart	
	OuterLoop = True
	all_hot_windows = []
	idx = 0
	while OuterLoop == True:
		window_size = min(300,int(((46457.67/((820. - 1.1*i)))-120.)*wscale))
		#print(window_size)
		xmid = int(imgsize[1]/2.)
		xstop = xmid + 5 * window_size
		xstart =  xmid -  5 * window_size
		x_range = int(xstop - xstart)
		#print(idx,i,window_size)
		xstart = max(0, xstart)
		xstop = min(1280,xstop)
		starty = i - window_size
		endy = i
		wcur = window_size
		scale = window_size/64.
		ystep = max(1,min(25,int(window_size * xy_overlap[1])))
		i += ystep		
		if i > ystop:
			OuterLoop = False

		#xcur = xmid
		#woffset = int(window_size/2.)
		#wcur = woffset
		#startx = xcur - woffset
		#endx = xcur + woffset

		#ystep = int(window_size/8.)
		#out_img = np.copy(image)

		if (np.mod(idx,1)==0):
			hot_windows = find_cars(undist, xstart, xstop, starty, endy, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat=spatial_feat, hist_feat=hist_feat)
			#print(np.shape(hot_windows))
			#all_hot_windows.append(hot_windows)
		# Visualize the windows we are currently running the search on
			drawVisImage = draw_boxes(tmp_img, hot_windows, color=(0, 0, 255), thick=2)
			tmp_img = np.copy(drawVisImage)
			if (False):
				plt.imshow(drawVisImage)
				plt.show()
		# Add heat to each box in box list
			heat = add_heat(heat,hot_windows)
		idx += 1
	#hot_windows = my_find_cars(image, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins , y_start_stop = [435, 680], xy_overlap = [0.1, 0.1])
	
	#plt.imshow(out_img)
	#plt.show()	
	#hot_windows = search_windows(feature_image, windows, clf, X_scaler, color_space='Custom', 
        #                spatial_size=spatial_size, hist_bins=hist_bins, 
        #                orient=orient, pix_per_cell=pix_per_cell, 
        #                cell_per_block=cell_per_block, 
        #                hog_channel=hog_channel, spatial_feat=spatial_feat, 
        #                hist_feat=hist_feat, hog_feat=hog_feat)                       
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to detect cars...')
	#all_hot_windows = np.concatenate((all_hot_windows))
	#

	
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,0)
	# 0 the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)

	draw1_img = draw_labeled_bboxes(np.copy(undist), labels)
	

	if (True):
		print(np.shape(draw1_img))
		fig = plt.figure()
		plt.subplot(131)
		plt.imshow(draw1_img)
		plt.title('Car Positions')
		plt.subplot(132)
		plt.imshow(heatmap, cmap='hot')
		plt.title('Heat Map')
		plt.subplot(133)
		plt.imshow(tmp_img)
		plt.title('Window Image')
		fig.tight_layout()               
		#plt.imshow(window_img)	
		plt.show()
	return draw1_img


class Vehicle_Subframe():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		self.xleftRel = 0
		self.ytopRel = 0
		self.xleftAbs = 0
		self.ytopAbs = 0
		#average x values of the fitted line over the last n iterations
		self.xwidth = 1 # pix      
		#polynomial coefficients averaged over the last n iterations
		self.ywidth = 1  # pix
		#polynomial coefficients for the most recent fit
		self.id = 255
		# try to sweep height of image scanned for the lines, some videos may do better if only lower part of image is used for lane detection in  windowing
		self.yOffsetSubframe = 0
		self.xOffsetSubframe = 0		
		self.windowScale = 1
		self.label = None



imagescale = []
allScales = []
binnedScales = []
def buildAllScales(image_size, wscale = 1.75, xy_overlap = [0.5, 0.5]):
	ybasemin = 420
	ybasemax = 720/1.1
	print(image_size)
	global allScales
	global binnedScales
	#wscale = 1.75
	i = ybasemin	
	OuterLoop = True
	all_hot_windows = []
	
	idx = 0
	while OuterLoop == True:
		#window_size = min(300,int(((46457.67/((820. - 1.1*i)))-120.)*wscale))
		window_size = min(250,int(((46457.67/((820. - 1.1*i)))-120.)*wscale))
		#print(window_size)
		xmid = int(image_size[1]/2.)
		xstop = xmid + 7 * window_size
		xstart =  xmid -  7 * window_size
		ystart = i - window_size
		ystop = i
		idx += 1
		allScales.append([xstart, xstop, ystart, ystop, window_size])
		ystep = max(1,min(75,int(window_size * xy_overlap[1])))
		i += ystep		
		if i > ybasemax:
			OuterLoop = False		
	idxCur = int(0)
	print(np.shape(allScales),allScales[idxCur][4])

	#OuterLoop = True
	#print(idxCur)
	#while(OuterLoop == True):
	#	print(len(allScales),idxCur)
	#	minScale = allScales[idxCur][4]
	#	for i in range((len(allScales))):
	#		xstart, xstop, ystart, ystop, window_size = allScales[i]
	#		if (allScales[i][4]> 1.1 * minScale)|(idxCur > len(allScales)):
	#			#ystartTmp = ystart
	#			break
	#	binnedScales.append((xstart,xstop,allScales[idxCur][2],ystop,allScales[idxCur][4]))
	#	idxCur += 1
	#	if(idxCur > len(allScales)-1):
	#		OuterLoop = False
	#print(np.shape(binnedScales))
	#print(binnedScales)
	# Build Threshold to account for 
#	return allScales, binnedScales

allWindows = []
allTest = []
thresh = np.ones([720,1280],dtype=np.float)
heatGlobal = np.zeros([720,1280],dtype=np.float)
mutex1_heat = threading.Lock()
mutex2_windows = threading.Lock()
def find_cars_threaded(img,xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat,threadId,):
	#draw_img = np.copy(img)
	global allWindows
	global allTest
	global thresh
	global heatGlobal
	global mutex1_heat
	global mutex2_windows
	t=time.time()
	img = img.astype(np.float32)/255
	yOffset = int(16*scale)
	ystart -= yOffset
	ystop += yOffset
	
	ystart = max(0, ystart)
	ystop = min(img.shape[0], ystop)
	xstart = max(0, xstart)
	xstop = min(img.shape[1],xstop)

	ctrans_tosearch = img[ystart:ystop,xstart:xstop,:]
	#print("xstart - xstop = " + str(xstart)+" "+str(xstop))
	#print("ystart - ystop = " + str(ystart)+" "+str(ystop))
	#print(np.shape(img_tosearch))
	#print(np.shape(ctrans_tosearch), scale)
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
	
	padLeft = np.random.random((int(ctrans_tosearch.shape[0]), 64, 3))
	padRight= np.random.random((int(ctrans_tosearch.shape[0]), 64, 3))
	ctrans_pad = np.concatenate((padLeft,ctrans_tosearch,padRight),axis=1)
	
	#padTop = np.random.random((16, int(ctrans_pad.shape[1]), 3))
	#padBot= np.random.random((16, int(ctrans_pad.shape[1]), 3))
	#ctrans_pad = np.concatenate((padTop,ctrans_pad,padBot),axis=0)	
	#print(np.shape(ctrans_pad))

	ch1 = None
	#ctrans_pad = ctrans_tosearch
	ch1 = ctrans_pad[:,:,0]
	#ch2 = ctrans_tosearch[:,:,1]
	#ch3 = ctrans_tosearch[:,:,2]
	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = max(1,(ch1.shape[0] // pix_per_cell) - cell_per_block + 1)
	#print(ch1.shape, pix_per_cell, cell_per_block) 
	nfeat_per_block = orient*cell_per_block**2
	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 1  # Instead of overlap, define how many cells to step
	nxsteps = int((nxblocks - nblocks_per_window) // cells_per_step)
	nysteps = int(max((nyblocks - nblocks_per_window) // cells_per_step, 1))
	#print(nyblocks, pix_per_cell, cell_per_block)
	
	# Compute individual channel HOG features for the entire image
	hog1, hog_image = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis = True)
	#hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	#hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	
	plt.imshow(hog_image)
	plt.axis('off')
	plt.show()

	warm_windows = []
	#allWindows = []
	allTest =[]
	cnt = 0
	#print(nxsteps, nysteps)
	#thresh = np.zeros_like(ch1)
	
	test_windows = []
	for xb in range(nxsteps):
		for yb in range(nysteps):
			
			ypos = int(yb*cells_per_step)
			xpos = int(xb*cells_per_step)
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			
			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell
			# Extract the image patch
			
			if spatial_feat == True:
				subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
				spatial_features = bin_spatial(subimg, size=spatial_size)
				#test_features.append(spatial_features)
				
				#file_features = np.concatenate(file_features)
				#print("End of spatial: "+ str(np.shape(file_features)))
			if hist_feat == True:
				# Apply color_hist()
				#num_channels = feature_image.shape[2]
				#print(num_channels)
				hist_features = color_hist(subimg, nbins=hist_bins)
				#print(np.shape(hist_features))
				hist_features = hist_features.reshape((1,len(hist_features)))
				#print(hist_features)
				#print(np.shape(hog_features))
				test_features= np.hstack((spatial_features.reshape(1, -1),hist_features.reshape(1, -1),hog_feat1.reshape(1, -1)))
				#file_features.append(hist_features)
				#print("End of hist: "+ str(np.shape(file_features)))
			else:
				test_features = hog_feat1
			
			# Scale features and make a prediction
			#test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			test_features = X_scaler.transform(test_features.reshape(1, -1))
			
			xbox_left = np.int((xleft - 64)*scale) + xstart
			ytop_draw = np.int((ytop)*scale) + ystart
			win_draw = np.int(window*scale)
			xbox_right = max(0,xbox_left+win_draw)
			ybot_draw = min(1280,ytop_draw+win_draw)
			xbox_left = max(0,xbox_left)
			ybot_draw = min(1280,ybot_draw)
			
			cnt += 1
			if(xbox_left<xbox_right)&(ytop_draw < ybot_draw):
				test_windows.append(((xbox_left, ytop_draw),(xbox_right,ybot_draw)))
				test_prediction = svc.predict(test_features)
				if test_prediction == 1:
					#cv2.rectangle(draw_img,(xbox_left, ytop_draw),(xbox_left+win_draw,ytop_draw+win_draw),(0,0,255),6)
					warm_windows.append(((xbox_left, ytop_draw),(xbox_right,ybot_draw)))
					#print(np.shape(warm_windows))
	
	if warm_windows != []:
		mutex2_windows.acquire()
		try:	
			allWindows.append(warm_windows)
		finally:
			mutex2_windows.release()
	
	
	print(np.shape(allWindows))
	allTest.append(test_windows)

	vehicles_subframe = []
	weight = (scale)
	thresh = add_heat(thresh, test_windows)
	
	
	
print_idx = 0
def vehicleDetectionPipelineThreaded(image_input):
	global windows
	global mtx
	global dist
	global FirstTime
	global allScales
	global allWindows
	global binnedScales
	global allTest
	global thresh
	global heatGlobal
	global mutex1_heat
	global mutex2_windows
	global print_idx
	#image = cv2.imread('./../test_images/test' + str(i)+'.jpg')
	#print(np.shape(image))
	undist=cv2.undistort(image_input, mtx, dist,None,None)
	#[h,w] = np.shape(image)[:2]
	t=time.time()
	feature_image = np.dstack((cv2.cvtColor(undist, cv2.COLOR_BGR2YCrCb)[:,:,0],cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)[:,:,1],cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)[:,:,2]))
	draw_image = np.copy(undist)
		
	heat = np.zeros_like(undist[:,:,0]).astype(np.float)
	window_image = np.zeros_like(undist).astype(np.float)
#	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#	luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
	# Uncomment the following line if you extracted training
	# data from .png images (scaled 0 to 1 by mpimg) and the
	# image you are searching is a .jpg (scaled 0 to 255)
	#image = image.astype(np.float32)/255

	color_space = 'Custom' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 18  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 8 # HOG cells per block
	hog_channel = 0 # Can be 0, 1, 2, or "ALL"
	spatial_size = (16, 16) # Spatial binning dimensions
	hist_bins = 16    # Number of histogram bins
	spatial_feat = False # Spatial features on or off
	hist_feat = False # Histogram features on or off
	hog_feat = True # HOG features on or off
	#xy_overlap=(0.1, 0.1)
	#ystart = 420
	#ystop = 720/1.3
	wscale = 1.5
	imgsize = np.shape(heat)
	tmp_img = np.copy(undist)
	threads = []
	thresh = np.ones_like(heat)
	#print(np.shape(allScales))
	for j in range(len(allScales)):
		xstop = allScales[j][1]
		xstart =  allScales[j][0]
		#print(xstart)
		x_range = int(xstop - xstart)
		ystart = allScales[j][2]
		ystop = allScales[j][3]
		window_size = allScales[j][4]
		#print(idx,i,window_size)
		scale = window_size/64.
		print_idx += 1
		if (np.mod(print_idx,3)==0):
			
			th = threading.Thread(target=find_cars_threaded, args=(undist, xstart, xstop, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat,j,))
		#draw_image =find_cars_optimized(undist, xstart, xstop, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, spatial_feat, hist_feat,j)
			threads.append(th)
			th.start()
	
	for j in range(len(threads)):
		threads[j].join()
	#hot_windows = ()
	#print(np.shape(hot_windows))
	#all_hot_windows.append(hot_windows)
	# Visualize the windows we are currently running the search on
	#for j in range(len(allWindows)):
	print(np.shape(allWindows))
	test_windows = np.concatenate((allTest))
	if allWindows != []:
		hot_windows = np.concatenate((allWindows))
	else: 
		hot_windows  = []
	print(np.shape(hot_windows))
	#print(np.shape(test_windows))
	allWindows = []
	drawVisImage = draw_boxes(tmp_img, hot_windows, color=(0, 0, 255), thick=4)
	#tmp_img = np.copy(drawVisImage)
	if (False):
		plt.imshow(heatGlobal, cmap = 'hot')
		plt.show()
	# Add heat to each box in box list
	heat = add_heat(heat,hot_windows)
	
	
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to detect cars...')
	# Apply threshold to help remove false positives
	relHeat = np.multiply(heat,thresh)
	relHeat = apply_threshold(relHeat,0)
	heatThd = apply_threshold(np.copy(heat),5)
	# 0 the heatmap when displaying    
	#relheatmap = np.clip(relHeat, 0, 0)
	heatmap = np.clip(heatThd, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)

	draw1_img = draw_labeled_bboxes(np.copy(undist), labels)
	

	if (False):
		#print(np.shape(draw1_img))
		fig = plt.figure(figsize = (8,6), dpi = 100)

		
		plt.subplot(221)
		plt.imshow(undist)
		plt.axis('off')
		plt.title('Input frame')
		plt.subplot(222)
		plt.imshow(drawVisImage)
		plt.title('Heat Image')
		plt.axis('off')
		plt.subplot(223)
		plt.imshow(heat, cmap='hot')
		plt.axis('off')
		plt.title('Heat Map')
		plt.subplot(224)
		plt.imshow(draw1_img)
		plt.title('Car Positions')
		plt.axis('off')
		
		fig.tight_layout()               
		plt.savefig("./output_images/Vehicle_detection_pipeline"+str(print_idx)+".png")
		#print_idx += 1	
		plt.show()
	allWindows = []
	return draw1_img




from moviepy.editor import VideoFileClip
fnames = []
fnames = glob.glob("*.mp4")

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#line_left = Line()
#line_right = Line()
#iteration = 0
windows = []
for file_num in range(1,len(fnames)):
	clip1 = VideoFileClip(fnames[file_num])
	print(fnames[file_num])
	#wscale = 1.75
	ystart, ystop = 420, 680
	imagesize = [720, 1280]
	wscale = 1.5
	xy_overlap = [0.1, 0.5]
	#print(window_size)
	#windows = initial_slide_window(imagesize, y_start_stop=[ystart,ystop], xy_overlap=(0.3, 0.1), wscale = wscale)
	buildAllScales(imagesize, wscale, xy_overlap)
	print(allScales)
	white_output = './Results/result5_'+str(file_num)+'.mp4'
#	white_clip = clip1.fl_image(vehicleDetectionPipelineThreaded)
	white_clip = clip1.fl_image(vehicleDetectionPipeline)
	white_clip.write_videofile(white_output, audio=False)


