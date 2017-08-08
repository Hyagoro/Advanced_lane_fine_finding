**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme/chessboard_corners.png "Chessboard corners"
[image2]: ./readme/chessboard_corners2.png "Chessboard corners 2"
[image3]: ./readme/convolution_test.png "Test convolution method"
[image4]: ./readme/dir_thr_select.png "Sobel threshold"
[image5]: ./readme/pipeline_edwarp_result.png "Pipeline warped"
[image6]: ./readme/pipeline_result.png "Pipeline res"
[image7]: ./readme/Sobel_thresh.png "Sobel thresh"
[image8]: ./readme/S_select.png "S thresh"
[image9]: ./readme/undistorded_chess.png "Undistorded chessboard"
[image10]: ./readme/undistorded_road.png "Warped road"
[image11]: ./readme/l_select.png "L select"
[image12]: ./readme/fit_poly.png "Poly fitted"
[image13]: ./readme/binary_top.png "Binary top"
[image14]: ./readme/final_result.png "Final result"

### Camera Calibration.

The code for this step is contained in the code cell n°3 of the Jupyter notebook located in "Advanced_line_finding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image9]

### Pipeline (single images).
#### Perspective transform.

From here, all steps of the pipeline use undistorded images.
The first step of the pipeline is to warp the image using `cv2.getPerspectiveTransform(src, dst)` to get an 'eye bird view'.

The code for my perspective transform includes a function which appears in code cell 8. I chose to hardcode the source and destination points in the following manner:

```python
offset1 = 280
src = np.float32([(575,464),(707,464), (1049,682), (258,682)])
dst = np.float32([[offset1, 0], [img_size[0]-offset1, 0], [img_size[0]-offset1, img_size[1]], [offset1, img_size[1]]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 280, 0        | 
| 707, 464      | 1000, 0       |
| 1049, 682     | 1000, 720     |
| 258, 682      | 280, 720      |

I verified that my perspective transform was working as expected by verifying that the lines appear parallel in the warped image.

![alt text][image10]

#### Transforms.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at note 9).
Here's an example of the 3 transformations I used (each transfomation is binarized) :

Sobel operator :

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
```

![alt text][image7]


S layer selection from 

```python
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
S = hls[:, :, 2]
```

![alt text][image8]

L layer selection:

```python
lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
lab_b = lab[:, :, 2]
```

![alt text][image11]

I finally merged all these tranformation and applied a median blur with kernel=7 to remove outliers.
```python
combined_binary = cv2.medianBlur(combined_binary,7)
```
![alt text][image13]

#### Fit polynomial

There is a function `find_lanes` to find main lane lines using sliding windows.
Then, each line from `find_lanes` is used to feed `fit_poly` function. This `fit_poly` function returns 2nd order polynomial using :
```python
np.polyfit(right_line.ally, right_line.allx, 2)
```

![alt text][image12]

#### Radius of curvature

I did this notes 26 through 28 in my code.
Meters per pixel for X = 30/720
Meters per pixel for Y = 3.7/700

#### End of the pipeline

I implemented this step in note 29 in my code in the function `global_process()`. This function groups all other functions and unwarps the image using the invert matrix, then draws a lane between the two polynomials over the original image. Here is an example of my result on a test image:

![alt text][image14]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_processed_met3.mp4)

Here's a [link to my challenge video result](./challenge_video_processed_met3.mp4)

---

### Discussion

During my work, I tried to use a method to find windows centroids using convolutions on a binary image, but the result wasn't satisfactory. The idea was to improve the function to find lines. I think it doesn't work because my pipeline was already adapted to the first function with sliding windows. 

One important thing was to find areas of interest for warp function, the pipeline was very sensitive to every small changes.

I spent also a lot of time to think how to correct frames without enough information. I implemented a function to create an history of n previous polynomials in order to "predict" when the information was missing.

My pipeline works correctly in favorable conditions, but when there is too much light, shadow or texture changes on the road, my pipeline fails. I think I must find better image transformations or improve the way I find the lanes on the road with this polynomial history I made. One way to improve my pipeline could be to implement a function to reduce noise using something like HDR to avoid fails when there is shadow or too much light on the image.
