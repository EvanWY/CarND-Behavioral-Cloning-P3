# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model is based on this model by Nvidia [https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 

The model includes 5 convolution layers and 3 fully connected layer, as well as RELU layers to introduce nonlinearity and Doprout layers to decrease overfitting. The data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 70, 73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and right view of the lane when driving, recovering from the left and right sides of the road and specific part of the track, and mirroring of all these images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from simple.

My first step was to use a convolution neural network model similar to the Nvidia end-to-end architecture. I thought this model might be appropriate because it should be able to handle simulation because it's much simple environment compare to real world environment.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add dropout layers after all the convolution and fully connected layers.

Then everything breaks. The training dataset has a much higher mean squared error and wasn't able to imporve. I delete most of the dropout layers, only left 2 dropout layers after the 2nd and 4th convolution layers. And the result is much better, the validation set mean squared error is about the same as the trainning set mean squared error.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, crossing bridge, right after the bridge. To improve the driving behavior in these cases, I run simulator to generate data for these parts and added them to the trainning set.

However, after adding special case to the trainning set, the car will sometimes run out of track in normal parts of the track. Then I started some random experiment.

Finally, I found a very useful optimization, which is passing the grayscale image instead of color image to the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65 - 78) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer (type)                      | Output Shape          | Param #     
| --------------------------------- | --------------------- | -----------: 
| Cropping                          | (None, 90, 320, 1)    | 0           
| Normalization                     | (None, 90, 320, 1)    | 0           
| Convolution (5x5 stride 3) + ReLU | (None, 29, 106, 36)   | 936         
| Convolution (5x5 stride 2) + ReLU | (None, 13, 51, 48)    | 43248       
| Dropout                           | (None, 13, 51, 48)    | 0           
| Convolution (5x5 stride 2) + ReLU | (None, 5, 24, 64)     | 76864       
| Convolution (3x3 stride 1) + ReLU | (None, 3, 22, 96)     | 55392       
| Dropout                           | (None, 3, 22, 96)     | 0           
| Convolution (3x3 stride 1) + ReLU | (None, 1, 20, 96)     | 83040       
| Flatten                           | (None, 1920)          | 0           
| Fully Connected + ReLU            | (None, 240)           | 461040      
| Fully Connected + ReLU            | (None, 50)            | 12050       
| Fully Connected                   | (None, 1)             | 51          

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
