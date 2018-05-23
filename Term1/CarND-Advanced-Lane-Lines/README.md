# Advanced Lane Finding

By Xi Chen
Jan 7, 2018

[//]: # (Image References)

[image1]: ./images/check_board.png "Check_board"
[image2]: ./images/real_world.png "Real-world image"
[image3]: ./images/binary.png "Binary Example"
[image4]: ./images/warped.png "Warp Example"
[image5]: ./images/ROI.png "ROI"
[image6]: ./images/ROI_nonstraight.png "ROI_nonstraight"
[image7]: ./images/Left_right.png 
[image8]: ./images/draw_line.png
[video1]: ./project_video_result.mp4 "Video"

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In the project resubmit, I borrowed and modified the code from this repo: https://github.com/ksakmann/CarND-Advanced-Lane-Lines/blob/master/stage1_test_image_pipeline.ipynb.

I did try to rewrite in my own, but some of them it's just too simple to rewrite to make more sense. I go through the author's code, and fully understand his rationale and implementation.

Since I didn't do well on the first submit, so I believe to compare with other student's implementation might help me.

Another reason y Why I learn from that repo is his code is more lighter and organized than mine. The one I implemented is too "heavy-weighted".

The code that I heavily borrowed is the class part, which is from the "Find lanes" section. However, I did correct mistakes and remove the redundant codes. As I mentioned earlier, I tried to rewrite, so I can claim it's mine, but a lot of the codes were also borrowed from the lecture, and even I tried, I don't think I could write it too differently. Therefore, instead of busy "rewriting", I was trying to "understand".


## Compute the camera calibration using chessboard images

In this section, I tried to make all the scripts of calibration using chessboard images in to individual functions, so I can re-use them. The check board results is here

![alt text][image1]

The procedure of calibration is by find object points from the world and image point from the image. Then using the cv2.calibrateCamera() function, it can automatically find the distortion coefficients, then apply them to the image to achieve undistortion. The output files are here
`
output_images/calibrationundistorted_calibration1.jpg
output_images/calibrationundistorted_calibration10.jpg
output_images/calibrationundistorted_calibration11.jpg
output_images/calibrationundistorted_calibration12.jpg
output_images/calibrationundistorted_calibration13.jpg
output_images/calibrationundistorted_calibration14.jpg
output_images/calibrationundistorted_calibration15.jpg
output_images/calibrationundistorted_calibration16.jpg
output_images/calibrationundistorted_calibration17.jpg
output_images/calibrationundistorted_calibration18.jpg
output_images/calibrationundistorted_calibration19.jpg
output_images/calibrationundistorted_calibration2.jpg
output_images/calibrationundistorted_calibration20.jpg
output_images/calibrationundistorted_calibration3.jpg
output_images/calibrationundistorted_calibration4.jpg
output_images/calibrationundistorted_calibration5.jpg
output_images/calibrationundistorted_calibration6.jpg
output_images/calibrationundistorted_calibration7.jpg
output_images/calibrationundistorted_calibration8.jpg
output_images/calibrationundistorted_calibration9.jpg
`

THe real-world application is by apply the distortion coefficients to the real-world image to achieve the undistortion. Here is an example:

![alt text][image2]

All the output of test images are located here

`
output_images/test_raw_undistorted_straight_lines1.jpg
output_images/test_raw_undistorted_straight_lines2.jpg
output_images/test_raw_undistorted_test1.jpg
output_images/test_raw_undistorted_test2.jpg
output_images/test_raw_undistorted_test3.jpg
output_images/test_raw_undistorted_test4.jpg
output_images/test_raw_undistorted_test5.jpg
output_images/test_raw_undistorted_test6.jpg
`

## Color-binary and channel
Here I will not re-write everything I did, I tested different channels from the previous submission, and for this submission I used the author's implementation: hls color space and l- and s- channels. I wouldn't disagree with the reviewer's suggestion to try the other two, but I found the output doesn't improve much, and since the author of the repo sense to have a better results, so I used his choices also.

The code for extract color channel is in the functions `hls_select` and `hsv_select` in the `Advanced Lane Finding Project.ipynb` file.

Then I applied Sobel X binary (`abs_sobel_thresh`)and combine the color-channel binary and Sobe X binary together in `binarize` in the same ipython notebook.

 The results is here: 

![alt text][image3]

## Perspective Transformation to Bird's-eye-view

In this part, I learned a trick to find the best corners for warp function `warp` by draw two lines in the original image and check whether the line is parallel and straight in the transformed image. I tweak the parameter and I found the following source gave me the best results, 
 Source        |   
|:-------------:| 
| 190,720      |
| 589,455      | 
| 695,455      | 
| 1120,720     |

and the destination is transformed by this function

`
corners=[[190,720],[589,455],[695,455],[1120,720]]
top_left=np.array([corners[0,0],0])
top_right=np.array([corners[3,0],0])
dst = np.float32([corners[0]+offset, top_left+offset, top_right-offset, corners[3]-offset])
`

The result is here:

![alt text][image4]

## Define region of interest

Applies an image mask and only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black.

And the results are here:
* For straight line
![alt text][image5]
* For non-straight line
![alt text][image6]


## Find lanes

This part I heavily borrowed from the author of the repo I listed above. But the original scripts have several bugs and couldn't work on the challange video of the project, so I correct them and tried to rewrite into my own.

In this part I seperated the left and right lanes, so it's more OOP. For `test1.jpg` image the results is here:

![alt text][image7]

The final `Lane` class is borrowed from the repo, which was a modification of the lecture scripts. The author has some good idea about the sanity check.

The output is like this:

![alt text][image8]

The author did use a averaged coefficiences, which is combining several images together to find the best results. However, the results of the challange video is roughly the same. But the required video work really good, and I like the fonts and color of the subtitles in the video.

## Video Processing Pipeline
Put everything together, I have this output video.
[Video](./project_video_result.mp4)

## Discussion
Comparing with the previous submission, I feel that the results didn't improve much, but the code is simpler and much lighter, which I prefer a lot. 

I still have issue with the chanllanged video, I don't now whether it's requirement or not. The reason why my pipeline cannot handel the video is the other lane was under a construction and the original lane lines was altered, so there's a very dark line in the middle to represent the original location. There should be some solution for it, but I didn't find any, I would like to some more concreted solution for it. I tried to average, add weight and use the different color channel as suggested, but the results did improve dramatically. Also I wonder what is the current state of art for even worse situation. 

The approach showed here is solely depends on the video camera, there's should be other types of cameras or technologies available. The visual approach (here) is limited to the x gradient and color channel transformation, and the bird's eye view curvature detection. And the pipeline was implementing by continuous update of detected line. I think it's also possible to use Radar, LIDAR, infrared camera, audio and GPS/MUM. Simply visual is too dangerous, even humans have issues, which can be illustrated from the car accidents. 





