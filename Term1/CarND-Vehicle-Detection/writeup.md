# Vehicle Detection Project

By Xi Chen, Jan 7, 2018
---

Introduction
This project is going to use some hallmark techniques of classical computer vision (i.e. no deep learning) to see how far we can get in detecting and tracking vehicles. From the reading, I noticed there is an alternative state-of-art to implement the vehicle detection, YOLO (You Only Look Once) and I have implemented it, but I don't think it's necessary for this project, since it's a Conv Net approach.

The techniques used here are including:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/HOG.png
[image3]: ./images/FP.png
[image4]: ./images/FN.png
[image5]: ./images/sliding_windows.png
[image6]: ./images/Heatmap.png
[video1]: ./processed_project_video1.mp4
[video2]: ./processed_project_video2.mp4

---
## Data Processing and Exploration

The training dataset provided for this project ( vehicle and non-vehicle images) are in the .png format. All the images are 64x64 sized. I didn't used the datasets from the Udacity. The totaly image sizes are: 8792 images of vehicles and 8968 images of non-vehicles.

Using user-defined function to process data. The function is located in the data_process.py and the default training, validation and test data ration is 7:2:1. And the results will be saved as pickle file, data.p.

* Number of all samples in vehicles:  8792
* Number of all samples in non-vehicles:  8968
* Number of samples in vehicles training set:  6152
* Number of samples in nonvehicles training set:  6277
* Number of samples in vehicles validation set:  1759
* Number of samples in nonvehicles validation set:  1794
* Number of samples in vehicles test set:  881
* Number of samples in nonvehicles test set:  897

Here shows what the random images look like for car and non-car categories.

![alt text][image1]

## Histogram of Oriented Gradients (HOG)

From the reviewer's suggestion, I re-tried YUV and YCrCb two color-spaces, but YUV return errors and YCrCb doesn't improve the SVM more, also the false positive/negative rates, ironically increased.

For YCrCB, the results is here (with C parameter set as 0.0001, as suggested, however, to compare with original results I did return the default setting, which is roughly the same and not a lot of improvement):
* Validation  Accuracy of SVC =  0.9823
* Test Accuracy of SVC =  0.9809
* number of false positives 50
* number of false negatives 13

The original HLS color space return only 39 FP, and 20 FN, so I didn't follow the suggestion, I don't think the video implemetation will have a better results if there's no improvement than my previous model.

The code for extracting featuer is `get_all_features` function from `get_features.py` file.


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. And the parameters I set up are:

* color_space = 'HLS'
* spatial_size = (16, 16)
* hist_bins = 32
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
* spatial_feat = True
* hist_feat = True
* hog_feat = True

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The above figure I showed 4 images from the data of vehicle and nonvehicle. I illustrated that what the features underneath. The original figure is in the frist column, and the next 3 columns are the color channels, and the last 3 columns are the HOG of the each color channel.

The reason why I settled on my final choice of HOG parameters is through the SVM classification results. The current results without tuning `C` parameter in SVM gives me the following:

* Validation Accuracy of SVC =  0.9834
* Test Accuracy of SVC =  0.9826
* number of false positives 39
* number of false negatives 20

Later, I include the parameter `C` tuning and the best results is when `C=0.01` and the results are:

* Validation Accuracy of SVC =  0.9834
* Test Accuracy of SVC =  0.9826
* number of false positives 38
* number of false negatives 21

![alt text][image3]

![alt text][image4]

The best way to play with the parameter and the color-space is to a grid-search with a variety of combination, however, the training and testing of this searching will take a huge amount of time. I don't know whether I can finish it on time or the AWS creat amount allow me to do so. If I have time, I would like to test it parallelizely on a multicore machine or HPC. 

As I mentioned earlier, I trained a linear SVM (since rbt takes too long to train and too long to predict) using parameter `C= [0.01, 0.001, 0.0001]`, and the results are not diviate too much, and when `C=0.001` I got a better results, which was tested on the video results. I have less false positive boxes.

The code of SVM implementation is shown in the `Vehicle+Detection+Project.ipynb`

### Summary 1
From the above images, you can see, the false positive number is much bigger than the false negative, which I intended to do so. I tweak the parameter to have a low false negative, which when there's is vehicle and the classifier cannot idenfity so. Compare to the false positive, even though it is quite high, this implied my philosophy if my model, better false identify as vehicle when there are, than the opposite. This is the common tradeoff between false pos and neg. When facing a prediction that could lead to accident, I want a low falures. As for the opposite, I don't think it could harm much to "cautiously" predict the opposite.

## Sliding Window Search

Similiar to the lecture and the suggestion from the introduction, I segmented image into four zones for the lower half of the image. The windows size are suggested to be 240, 180, 120 and 70 pixels for four zones respectively. The overlapping is 75%, which is the best perform I can get.

I didn't use a random search, instead I set the y-axis strategically. First, I know that the cars will only show at the bottom half of the image, and as the distance get further, the cars should be smaller, so with smaller windows searching at a futher distance, and bigger one for closer distance. The y-axis for four windes are set as following

`
yi0,yi1,yi2,yi3 = 380,380,395,405                  
Y_pos =[[yi0,yi0+240/2],[yi1,yi1+180/2],[yi2,yi2+120/2],[yi3,yi3+70/2]]
`

And the sliding window search is as following. 

![alt text][image5]

### Summary 2
The reason why I set the overlapping is 75% is throught testing, I found out 75% gives me the best results. If less than 75%, there's a lot missing cars in the image, if more then 75%, it will take longer to search and results didn't improve much. From the above results, you can see the windows detection is quite good, and the next step is to testing the model on the video.



## Heatmap Function

The next step is to using the detected boxes to show a heatmap with higher value present more boxes detected (where car located). The code heatmap function is also in the ipython notebook, I showed heatmap results for the six test images given by the project.

![alt text][image6]

From the above figures, you can tell that the detection job is working, but there's something not as we expected. For example, some of the images have weird lighting, so it's really hard for a simple classifier to handle. I also notice that the power of the model is depends on the the window size, window number, scan region, and threshond of how many boxes overlopping in the heatmap function, I simply used four windows size, it is quite good, with a lot of false positive, which is great from my perspective, better safe than never. I have intend to make the false positive rate high with the trade-off flase negative rate low.

---

### Video Implementation

I corrected the mistake in the code, which I didn't use the heatmap function to filter out the false positive detections and the results is here: [video1](processed_project_video1.mp4). I also set the linear SVM parameter `C=0.01` to render another video: [video1](processed_project_video2.mp4). 

When implemnent the `C` parameter for SVM classifier, the resulting video has a slight better results, with less false positive detection.


## Discussion
I started with a linear SVM because the performance is not too bad comparing with the rbf svm module. However, since I trying to make a pipeline that allow fast response, the fastness of the linear model is really important. I intended to make the false positive higher due to the small size of the training data, this is the general approach from the bioinformatical research field, since we want to prevent accident so we don't want to miss any cars in the video. Higher false positive rate, in this case it is not severe, may be reduced with more training data. Also current technology is combine multiple predictions from different types of cameras, even LDR, so it might help to have an ensemble approach when I have more knowledges on other newst state of art.

The problems I have here is the pipeline is too slow, which could be caused by the HOG features computation for every single image. I think if there is a function that can predict the significant movement of the view (ignore a couple or a series of images when nothing happens) and compute HOG every other 2-5 images.

I've read a lot of people to use YOLO model, which is very software-environment-dependent, for example, to make YOLO work, you have to have certein version of tensorflow. I tried to implement, and it works, but I don't think it's necessary for this project.

Also the GPU/multi-core CPUs should be fully utilized, so it might be a great idea to parallize the module of window search by search different window sizes in different instances at the same time.


