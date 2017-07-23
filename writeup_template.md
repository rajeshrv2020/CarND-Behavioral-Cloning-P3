# **Behavioral Cloning** 

---

** Behavioral Cloning Project **

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* Youtube Link of the video : https://youtu.be/CbOlbDcfdYk


[//]: # (Image References)

[image1]: images/architecture.jpg "Architecture"
[image2]: images/center_driving.jpg "Center Driving"
[image3]: images/left_recovery_1.jpg "Left Recovery"
[image4]: images/left_recovery_2.jpg "Left Recovery"
[image5]: images/right_recovery_1.jpg "Right Recovery"
[image6]: images/unflipped_image.jpg "Unflipper Image"
[image7]: images/flipped_image.jpg "Dataset"
[image8]: images/dataset.jpg "Dataset"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* README.md which is same as writeup_report,md which you have viewing

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 
 * 5 Convolution layers with each made of

        * 5x5 stride with filter depth = 24 [ model.py line_no: 143  ]

        * 5x5 stride with filter depth = 36 [ model.py line_no: 145  ]

        * 5x5 stride with filter depth = 48 [ model.py line_no: 147  ]

        * 3x3 stride with filter depth = 64 [ model.py line_no: 149  ]

        * 3x3 stride with filter depth = 64 [ model.py line_no: 151  ]
 * Dropout layer sandwidth between the convolution layers [ model.py line_no: 144,146,168.150,152  ]
 * One Flattened layer  [ model.py line_no: 153  ]
 * A Dropout layer before fully connected layer [ model.py line_no: 154 ]
 * 3 Fully connected layers with depth of 100,50,10  [ model.py line_no: 155-157 ]
 * A Dropout layer before output layer [ model.py line_no: 158 ]
 * Finally , a dropout layer [ model.py line_no: 159 ]
        
The model includes RELU and ELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer [ model.py line_no: 174  ]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. (line numbers pointed out in the above section )

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually  [ model.py line_no: 181  ]

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to based on Nividia Architecture. 

My first step was to use a convolution neural network model similar to the LeNet Architecture. The MSE was high (>1) . So, i decided to try out Nividia Architecture. 

I thought this model might be appropriate because MSE was less than 0.1 during the first epoch.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropouts in between convolution layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. 

To address it, i collected more dataset at the failing spots. Example. Sharp Left turn.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network as seen below

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Tranining data is collected by driving the car for 3 laps and driving the lap in the reverse direction. 

During the initial testing, the Car was not steering and it always keeping staight. I realised that the 80% of the datasets has steering angle of 0.

![alt text][image8]

So, i trimmed out the dataset with only 70% of the images with steering angle==0.

After trimming the data, still the driving behaviour was not great. The car went off the road at sharp turns.  More data are collected from the track where the car was drifing away.. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery back to the road. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]

I keep adding the recovery images to the dataset untill car is able to recovery back to the road.

Also, to increase the dataset, i have used both left and right images with steering correction of 0.2.

To augment the data sat, I also flipped images and angles thinking that this would add more images to the dataset. For example, here is an image that has been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 22782 number of data points. After zero steeting filtering, the dataset was 13221. After Augmentation, the total dataset was 63510.
 I then preprocessed this data by 
  * Normalized the image
  * Cropped the images only to the area of interest.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

I used callbacks to save the best run. My Epoch was set to 10.The ideal number of epochs was 5 since the best model was reached by epoch 5. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.
