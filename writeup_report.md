# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_line_driving.jpg "Center Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes, 3x3 filter sizes and depths between 24 and 64 (model.py lines 90-102) 

The model includes RELU layers to introduce nonlinearity (code line 93-97), and the data is normalized in the model using a Keras lambda layer (code line 91). The image is also cropped from above and below so that redundant information will not be passed to the model (code line 92).

#### 2. Attempts to reduce overfitting in the model 

The model was trained and validated on different data sets to ensure that the model was not overfitting. After trying to train for multiple epochs, I observed that the model tended to overfit the training data. If I run the overfitted model in simulator, the steering angle will keep switching between positive and negative values, which is due to the driving behavior of the training dataset. After trial and error, I set the number of epoch to 1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In the beginning, I was using the provided default dataset. The driving behavior in the simulator was very human like, but it sometimes steered to left and went off the track. As a result, I augemented the flipped images and steering angles to the training dataset. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find out what are the desired layers for this application.

My first step was to use a convolution neural network model similar to the LeNet, and added dropout layers after each maxpooling layer. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. After training for multiple epochs, I found that the validation error was increasing, so I stopped the training after epochs 1. When I ran the simmulatorm, the vehicle was running smoothly on straight track but it failed at sharp turns.

After LeNet, I decided to try on Nvidia's end-to-end learning model architecture. The maxpooling layers were removed and `stride=(2,2)` in `Conv2D` was used to shrink the size. The result was even better then LeNet, now the vehicle can combat sharp turns and also drive like human. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-102) consisted of a convolution neural network with the following layers and layer sizes: 

|Layer (type)               |  Output Shape             | Param | note | 
|:-------------------------:|--------------------------:|------:| ----:|
|lambda_1 (Lambda)          |  (None, 160, 320, 3)      | 0     |      |    
|cropping2d_1 (Cropping2D)  |  (None, 66, 320, 3)       | 0     |      | 
|conv2d_1 (Conv2D)          |  (None, 31, 158, 24)      | 1824  | kernel=(5,5,24)|    
|conv2d_2 (Conv2D)          |  (None, 14, 77, 36)       | 21636 | kernel=(5,5,36)|
|conv2d_3 (Conv2D)          |  (None, 5, 37, 48)        | 43248 | kernel=(5,5,48)|   
|conv2d_4 (Conv2D)          |  (None, 3, 35, 64)        | 27712 | kernel=(3,3,64)|   
|conv2d_5 (Conv2D)          |  (None, 1, 33, 64)        | 36928 | kernel=(3,3,64)|   
|flatten_1 (Flatten)        |  (None, 2112)             | 0     |    |
|dense_1 (Dense)            |  (None, 100)              | 211300|    |
|dense_2 (Dense)            |  (None, 50)               | 5050  |    |
|dense_3 (Dense)            |  (None, 10)               | 510   |    |
|dense_4 (Dense)            |  (None, 1)                | 11    |    |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the sample dataset that consists multiple laps on track one using center lane driving. The total number of images is 8034. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would balance the intention to turn left and right.

After the collection process, I had 16068 number of data points including augmented data. I then preprocessed this data by applying an lambda layer in keras model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by increasing validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
