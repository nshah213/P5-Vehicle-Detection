**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarNotCarExamples.png
[image2]: ./output_images/CarNotCarExamplesHOG.png
[image3]: ./output_images/Vehicle_detection_pipeline_frame_2.png
[image4]: ./output_images/Vehicle_detection_pipeline8.png
[image5]: ./output_images/Vehicle_detection_pipeline45.png
[image6]: ./output_images/Vehicle_detection_pipeline41.png
[image7]: ./output_images/Vehicle_detection_pipeline2.png
[image8]: ./output_images/Vehicle_detection_pipeline0.png


## Classifier

`VehicleDetectionOptimizedHOG.py` contains the best version of code to currently for all the next sections.

### Training images

The training images used were downloaded from a multiple datasets available online to download for research purposes. Overall the total number of images used for cars and non-cars are just more than 12,000. Code to read them all and create a single car and a single non-car lists is on lines 17 - 82. It saves the data into a pickle file, to avoid having to repeat the operation over again. Uncomment to re load new image set for cars and non-cars, will need to modify code based on data folder location and naming convention.

### Algorithm - Histogram of Oriented Gradients (HOG) with Support Vector Classifier (SVC)

#### 1. Extract HOG features from training images

The code for this step is contained on lines 98 - 113.

I started by reading in all the `vehicle` and `non-vehicle` images by loading the pickle file.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started by using HOG parameters in the range recommended by N. Dalal in his talk explaining his work. The problem they were working was detecting humans in a video, and we want to find vehicles, but I thought of it as a great starting point. By displaying the images returned for multiple car and non-car images, I was able to settle with the following parameters `orientations=18`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`

HOG output for the various cars and non-cars examples listed above looks like this - 

![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Line 575 - 584 for definition and training of the support vector classifier. Classifier implemented using `sklearn.tree.DecisionTreeClassifier()` and trained using from `sklearn.model_selection.GridSearchCV()`. I used a grid of C and gamma parameters using a logscale. The algorithms is able to get performance accuracy of 98.6% using only hog features of Y channel alone. Here are the best parameters of the SVC after training. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to try an approximate a hyperbolic relation between apparent vehicle height and the y position in the image. This was emperically tuned from height of the vehicles in the test video frames. At the start of the code `BuildAllScales()` function is called with some high level parameters that can help decide number of overlap eventual scales to be used later in the video detection pipeline. The function returns a list of x,y start and stop positons for the image area where the scale is useful and the window size roughly the size of the aparent vehicle height at that point.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.
As predictable, the pipeline accuracy for localization increases significantly if we use more scales and large overlap. Below is a visualization of the performance of the pipleline with finer resolution of scales.
![alt text][image4]

Drawback is that it takes a long time to process an image. I have used multithreadding to try and speed up the calculation and also leveraged the calculate once and sub sample windows to reduce total number of HOG calculations required.

Here is the performance of the pipeline with 1/3 rd the number of scales used and the performance improves significantly, by at least 5 times.
![alt text][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./Results/results4_1.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some more visualizations of the pipeline working. 
![alt text][image6]

![alt text][image7]

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Biggest concern is it is far from running real time. Need to understand methods for software and hardware optimizations to allow real time performance.
2. Classifier is using basic HOG on 1 channel only, HOG on all 3 channels of YCrCb will be beneficial. Selecting more features and running decision tree to select the features with the most information will help a lot.
3. Some current false positives especially on edges of the search frame will be reduced if we look at more spatial features along with HOG to help our classifier.

4. Training SVC with gridSearch is cumbersome, but performance on well selceted grid range i
